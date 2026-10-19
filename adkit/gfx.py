"""The 2.5D compositor: sprites, lights, bloom, ink, stage, particles and post.

Per frame, in this order (matches the method in SKILL.md):

    stage background -> particle field (as ink, under the UI)
    -> UI layer (premultiplied linear RGBA sprites) -> UI drop shadow -> composite UI
    -> light layer (bloomed) composited as ink (light stage) or glow (dark stage)
    -> camera (breath, shake, push) -> vignette -> gamma -> grain

All positions are in *design pixels*: a 1920x1080 frame for 16:9, 1080x1920 for
9:16, 1080x1080 for 1:1. ``S`` converts design pixels to output pixels, so the
same film renders at half resolution for previews and full resolution for the
final without touching scene code.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from .core import (Path, clamp, hex_srgb, lin, linear_to_srgb, srgb_to_linear)

LMAX = 2.0  # light energy that maps to 255 in the 8-bit AA raster
_LIN_LUT = srgb_to_linear(np.arange(256, dtype=np.float32) / 255.0).astype(np.float32)  # sRGB byte -> linear


# ---------------------------------------------------------------------------
# sprites
# ---------------------------------------------------------------------------

class Sprite:
    """Premultiplied linear-light RGBA image with a mip pyramid.

    ``ppd`` is pixels per design pixel of the base level. Drawing picks the
    smallest level that is still at least as sharp as the output needs, then
    does one bilinear warp, so scaled sprites stay clean without aliasing.
    """

    def __init__(self, img, ppd=1.0, baseline=None, meta=None):
        img = np.ascontiguousarray(img, dtype=np.float32)
        self.levels = [(img, float(ppd))]
        while min(img.shape[:2]) > 12:
            img = cv2.resize(img, (max(1, img.shape[1] // 2), max(1, img.shape[0] // 2)),
                             interpolation=cv2.INTER_AREA)
            self.levels.append((img, self.levels[-1][1] / 2))
        self.baseline = baseline  # design px from top, for text
        self.meta = meta or {}

    @property
    def img(self):
        return self.levels[0][0]

    @property
    def ppd(self):
        return self.levels[0][1]

    @property
    def w(self):
        return self.img.shape[1] / self.ppd

    @property
    def h(self):
        return self.img.shape[0] / self.ppd

    def pick(self, need):
        best = self.levels[0]
        for lv in self.levels:
            if lv[1] >= need * 0.999:
                best = lv
            else:
                break
        return best

    @classmethod
    def from_rgba8(cls, arr, ppd=1.0, supersample=1, **kw):
        """uint8 sRGB straight-alpha array -> Sprite (downsampled by ``supersample``)."""
        # Work in horizontal strips so a big supersampled sprite never needs a full float copy
        # (a 5000 px wide UI screenshot would otherwise briefly cost over a gigabyte).
        h, w = arr.shape[:2]
        ss = max(1, int(supersample))
        oh, ow = max(1, round(h / ss)), max(1, round(w / ss))
        out = np.empty((oh, ow, 4), np.float32)
        lut = _LIN_LUT
        rows_out = max(1, 4096 * 64 // max(1, ow))  # ~1M output pixels per strip
        for o0 in range(0, oh, rows_out):
            o1 = min(oh, o0 + rows_out)
            i0, i1 = o0 * ss, min(h, o1 * ss)
            strip = arr[i0:i1]
            a = strip[..., 3:4].astype(np.float32) * (1.0 / 255.0)
            f = np.empty(strip.shape[:2] + (4,), np.float32)
            f[..., :3] = lut[strip[..., :3]] * a
            f[..., 3:4] = a
            if ss > 1:
                f = cv2.resize(f, (ow, o1 - o0), interpolation=cv2.INTER_AREA)
            out[o0:o1] = f
        return cls(out, ppd, **kw)

    @classmethod
    def from_pil(cls, im, ppd=1.0, supersample=1, **kw):
        return cls.from_rgba8(np.asarray(im.convert("RGBA")), ppd, supersample, **kw)

    def tinted(self, hex_color, amount=1.0):
        """Recolour every opaque pixel toward one colour (keeps alpha)."""
        img = self.img.copy()
        c = lin(hex_color)
        img[..., :3] = img[..., :3] * (1 - amount) + c * img[..., 3:4] * amount
        return Sprite(img, self.ppd, self.baseline, dict(self.meta))


# ---------------------------------------------------------------------------
# the canvas
# ---------------------------------------------------------------------------

class Canvas:
    def __init__(self, W, H, S, stage_mode="light", slate="#5F6368"):
        self.W, self.H, self.S = int(W), int(H), float(S)
        self.stage_mode = stage_mode
        self.slate = lin(slate)
        self.bg = np.zeros((self.H, self.W, 3), np.float32)
        self.ui = np.zeros((self.H, self.W, 4), np.float32)
        self.under = np.zeros((self.H, self.W, 3), np.float32)  # particles: under the UI
        self.over = np.zeros((self.H, self.W, 3), np.float32)   # lights: over the UI
        self._groups = []
        self.camera = dict(zoom=1.0, x=0.0, y=0.0, rot=0.0)
        self.box = {"ui": None, "under": None, "over": None}  # touched regions, px
        self.off = (0.0, 0.0)  # design-px translation applied to everything drawn (whip pans)
        self.zoom = 1.0        # scale about the frame centre (push transitions)
        self.fade = 1.0        # opacity multiplier for everything drawn
        self.region = None     # (x0, x1) design px: only this vertical band is visible (wipes)

    def _tp(self, x, y):
        """Design point after the canvas camera (offset, then zoom about the centre)."""
        cx, cy = self.W / self.S / 2, self.H / self.S / 2
        return cx + (x + self.off[0] - cx) * self.zoom, cy + (y + self.off[1] - cy) * self.zoom

    def _region_mask(self, x0, x1):
        if self.region is None:
            return None
        r0, r1 = self.region
        xs = np.arange(x0, x1, dtype=np.float32) / self.S
        return np.clip(np.minimum(xs - r0, r1 - xs) / 2.0 + 0.5, 0, 1).astype(np.float32)

    def _touch(self, layer, x0, y0, x1, y1):
        b = self.box[layer]
        self.box[layer] = (x0, y0, x1, y1) if b is None else (min(b[0], x0), min(b[1], y0), max(b[2], x1),
                                                             max(b[3], y1))

    # -- coordinate helpers ------------------------------------------------
    def px(self, v):
        return v * self.S

    # -- layers --------------------------------------------------------------
    def _target(self):
        return self._groups[-1][0] if self._groups else self.ui

    def begin_group(self):
        """Start an offscreen layer (for motion blur / group fades)."""
        self._groups.append([np.zeros_like(self.ui), None])

    def end_group(self, alpha=1.0):
        buf, box = self._groups.pop()
        if box is None:
            return
        x0, y0, x1, y1 = box
        src = buf[y0:y1, x0:x1] * alpha
        dst = self._target()[y0:y1, x0:x1]
        dst *= (1 - src[..., 3:4])
        dst += src
        self._grow((x0, y0, x1, y1))

    def _grow(self, box):
        if not self._groups:
            self._touch("ui", *box)
            return
        g = self._groups[-1]
        if g[1] is None:
            g[1] = list(box)
        else:
            b = g[1]
            g[1] = [min(b[0], box[0]), min(b[1], box[1]), max(b[2], box[2]), max(b[3], box[3])]

    # -- sprite drawing ------------------------------------------------------
    def draw(self, sp: Sprite, x, y, scale=1.0, rot=0.0, alpha=1.0, anchor=(0.5, 0.5),
             blur=0.0, wipe=None, mode="over", sx=None, sy=None, dim=0.0, clip=None, band=None,
             grad=None, flip=False):
        """Draw a sprite with its anchor at design point (x, y).

        rot is in degrees. ``wipe=(x_edge, softness)`` reveals left-to-right in
        design px. ``mode='add'`` accumulates (for motion-blur sub-samples).
        ``dim`` darkens toward the stage (for 'others dim 60 %').
        ``clip=(x0, y0, x1, y1)`` hides everything outside a design rect.
        ``band=(c, width, angle_deg)`` keeps only a soft diagonal band (sheens).
        ``grad=(x0, y0, x1, y1)`` fades alpha linearly from p0 (1) to p1 (0).
        ``flip`` mirrors vertically (reflections).
        """
        if alpha <= 0.002 or scale <= 1e-4:
            return
        x, y = self._tp(x, y)
        alpha *= self.fade
        if alpha <= 0.002:
            return
        sxx = (scale if sx is None else sx) * self.zoom
        syy = (scale if sy is None else sy) * self.zoom
        need = self.S * max(abs(sxx), abs(syy))
        img, ppd = sp.pick(need)
        if flip:
            img = img[::-1]
        h, w = img.shape[:2]
        kx, ky = self.S * sxx / ppd, self.S * syy / ppd
        c, s = math.cos(math.radians(rot)), math.sin(math.radians(rot))
        ax, ay = anchor[0] * w, anchor[1] * h
        tx, ty = x * self.S, y * self.S
        M = np.array([[c * kx, -s * ky, 0.0], [s * kx, c * ky, 0.0]])
        M[:, 2] = [tx - (M[0, 0] * ax + M[0, 1] * ay), ty - (M[1, 0] * ax + M[1, 1] * ay)]
        corners = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]], np.float64) @ M.T
        pad = int(math.ceil(blur * self.S * 3)) + 2
        x0 = max(0, int(math.floor(corners[:, 0].min())) - pad)
        y0 = max(0, int(math.floor(corners[:, 1].min())) - pad)
        x1 = min(self.W, int(math.ceil(corners[:, 0].max())) + pad)
        y1 = min(self.H, int(math.ceil(corners[:, 1].max())) + pad)
        if x1 <= x0 or y1 <= y0:
            return
        Mr = M.copy()
        Mr[:, 2] -= (x0, y0)
        src = cv2.warpAffine(img, Mr, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        if src.ndim == 2:
            src = src[..., None]
        if blur > 0.05:
            src = cv2.GaussianBlur(src, (0, 0), blur * self.S)
        if wipe is not None:
            edge, soft = wipe
            xs = (np.arange(x0, x1, dtype=np.float32) / self.S)
            m = clamp((edge - xs) / max(1e-3, soft) + 0.5)
            src *= m[None, :, None]
        if clip is not None or band is not None or grad is not None:
            # back into the scene's own coordinates (undo zoom, then offset)
            ccx, ccy = self.W / self.S / 2, self.H / self.S / 2
            ys_ = (ccy + (np.arange(y0, y1, dtype=np.float32) / self.S - ccy) / self.zoom - self.off[1])[:, None]
            xs_ = (ccx + (np.arange(x0, x1, dtype=np.float32) / self.S - ccx) / self.zoom - self.off[0])[None, :]
            m = np.ones((y1 - y0, x1 - x0), np.float32)
            if clip is not None:
                cx0, cy0, cx1, cy1 = clip
                m *= ((xs_ >= cx0) & (xs_ < cx1) & (ys_ >= cy0) & (ys_ < cy1))
            if band is not None:
                c_, w_, ang = band
                ca, sa = math.cos(math.radians(ang)), math.sin(math.radians(ang))
                m *= np.exp(-(((xs_ * ca + ys_ * sa) - c_) / max(1e-3, w_)) ** 2)
            if grad is not None:
                gx0, gy0, gx1, gy1 = grad
                dx, dy = gx1 - gx0, gy1 - gy0
                L2 = dx * dx + dy * dy + 1e-6
                m *= np.clip(1 - ((xs_ - gx0) * dx + (ys_ - gy0) * dy) / L2, 0, 1)
            src *= m[..., None]
        rm = self._region_mask(x0, x1)
        if rm is not None:
            src *= rm[None, :, None]
        if dim > 0:
            # pull colour toward the stage while keeping coverage
            src[..., :3] = src[..., :3] * (1 - dim) + src[..., 3:4] * self._stage_lin * dim
        if alpha < 1:
            src *= alpha
        dst = self._target()[y0:y1, x0:x1]
        if mode == "add":
            dst += src
        else:
            dst *= (1 - src[..., 3:4])
            dst += src
        self._grow((x0, y0, x1, y1))

    _stage_lin = lin("#FAFAFB")

    def draw_motion(self, sp, fn, t, shutter, n=5, **kw):
        """Motion blur: average ``n`` sub-samples of fn(t) -> dict(x, y, scale, rot, alpha...)."""
        if shutter <= 0 or n <= 1:
            self.draw(sp, **{**kw, **fn(t)})
            return
        self.begin_group()
        for i in range(n):
            ti = t - shutter * (i / (n - 1))
            p = {**kw, **fn(ti)}
            p["alpha"] = p.get("alpha", 1.0) / n
            p["mode"] = "add"
            self.draw(sp, **p)
        self.end_group()

    def draw_quad(self, sp: Sprite, quad, alpha=1.0):
        """Perspective-warp a sprite onto 4 design points (tl, tr, br, bl)."""
        quad = np.array([self._tp(px_, py_) for px_, py_ in np.asarray(quad, np.float64)]) * self.S
        alpha *= self.fade
        # pick a mip level from the largest edge
        need_w = max(np.linalg.norm(quad[1] - quad[0]), np.linalg.norm(quad[2] - quad[3])) / sp.w
        img, ppd = sp.pick(need_w)
        h, w = img.shape[:2]
        x0 = max(0, int(math.floor(quad[:, 0].min())) - 2)
        y0 = max(0, int(math.floor(quad[:, 1].min())) - 2)
        x1 = min(self.W, int(math.ceil(quad[:, 0].max())) + 2)
        y1 = min(self.H, int(math.ceil(quad[:, 1].max())) + 2)
        if x1 <= x0 or y1 <= y0:
            return
        srcq = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        H = cv2.getPerspectiveTransform(srcq, np.float32(quad - (x0, y0)))
        src = cv2.warpPerspective(img, H, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        rm = self._region_mask(x0, x1)
        if rm is not None:
            src *= rm[None, :, None]
        if alpha < 1:
            src *= alpha
        dst = self._target()[y0:y1, x0:x1]
        dst *= (1 - src[..., 3:4])
        dst += src
        self._grow((x0, y0, x1, y1))

    # -- lights --------------------------------------------------------------
    def light_poly(self, pts, colors, intens, width=4.5, layer="over", occlude=0.0):
        """Draw a polyline of light with per-vertex colour (linear RGB) and intensity.

        Rasterised with OpenCV's 8-bit anti-aliased lines at 1/16 px precision
        into an ROI, then added to a float light layer.
        """
        pts = np.asarray(pts, np.float64)
        if len(pts) < 2:
            return
        cx, cy = self.W / self.S / 2, self.H / self.S / 2
        pts = np.stack([cx + (pts[:, 0] + self.off[0] - cx) * self.zoom,
                        cy + (pts[:, 1] + self.off[1] - cy) * self.zoom], 1) * self.S
        width = width * self.zoom
        if len(pts) < 2:
            return
        colors = np.asarray(colors, np.float32)
        if colors.ndim == 1:
            colors = np.repeat(colors[None], len(pts), 0)
        intens = np.asarray(intens, np.float32) * self.fade
        if intens.ndim == 0:
            intens = np.full(len(pts), float(intens), np.float32)
        th = max(1, int(round(width * self.S)))
        pad = th + 3
        x0 = max(0, int(pts[:, 0].min()) - pad)
        y0 = max(0, int(pts[:, 1].min()) - pad)
        x1 = min(self.W, int(pts[:, 0].max()) + pad + 1)
        y1 = min(self.H, int(pts[:, 1].max()) + pad + 1)
        if x1 <= x0 or y1 <= y0:
            return
        roi = np.zeros((y1 - y0, x1 - x0, 3), np.uint8)
        q = np.round((pts - (x0, y0)) * 16).astype(np.int64)
        vals = np.clip(colors * intens[:, None] / LMAX * 255.0, 0, 255)
        # thin lines lose energy to AA: compensate so half-res previews match
        gain = clamp(width * self.S / th, 0.35, 1.0)
        for i in range(len(q) - 1):
            v = vals[i + 1] * gain
            if v.max() < 0.5:
                continue
            cv2.line(roi, (int(q[i, 0]), int(q[i, 1])), (int(q[i + 1, 0]), int(q[i + 1, 1])),
                     (float(v[0]), float(v[1]), float(v[2])), th, cv2.LINE_AA, 4)
        f = roi.astype(np.float32) * (LMAX / 255.0)
        rm = self._region_mask(x0, x1)
        if rm is not None:
            f *= rm[None, :, None]
        if occlude > 0:
            f *= (1 - occlude * self.ui[y0:y1, x0:x1, 3:4])
        (self.over if layer == "over" else self.under)[y0:y1, x0:x1] += f
        self._touch("over" if layer == "over" else "under", x0, y0, x1, y1)

    def light_dot(self, p, color, energy=1.5, radius=4.0, layer="over"):
        x, y = self._tp(p[0], p[1])
        x, y = x * self.S, y * self.S
        r = max(1.0, radius * self.S * self.zoom)
        energy = energy * self.fade
        x0, y0 = int(max(0, x - r - 3)), int(max(0, y - r - 3))
        x1, y1 = int(min(self.W, x + r + 4)), int(min(self.H, y + r + 4))
        if x1 <= x0 or y1 <= y0:
            return
        roi = np.zeros((y1 - y0, x1 - x0, 3), np.uint8)
        v = np.clip(np.asarray(color) * energy / LMAX * 255, 0, 255)
        cv2.circle(roi, (int(round((x - x0) * 16)), int(round((y - y0) * 16))), int(round(r * 16)),
                   tuple(float(c) for c in v), -1, cv2.LINE_AA, 4)
        f = roi.astype(np.float32) * (LMAX / 255.0)
        rm = self._region_mask(x0, x1)
        if rm is not None:
            f *= rm[None, :, None]
        (self.over if layer == "over" else self.under)[y0:y1, x0:x1] += f
        self._touch("over" if layer == "over" else "under", x0, y0, x1, y1)

    def light_path(self, path: Path, head, tail=0.35, color="#3186FF", color_tail=None,
                   width=4.5, energy=1.0, head_glow=1.0, n=96, layer="over", occlude=0.0,
                   fade=1.0):
        """One travelling light: the head at arc fraction ``head`` with a fading tail.

        Tail intensity is w**2 so it dies out smoothly behind the head. ``head``
        may exceed 1: the tail then drains into the end point. ``color_tail``
        makes a hand-over (the colour changes along the tail).
        """
        if fade <= 0.003:
            return
        u0, u1 = head - tail, head
        us = np.linspace(u0, u1, n)
        w = (us - u0) / max(1e-6, tail)
        keep = us >= 0
        if path.closed:
            keep = np.ones_like(us, bool)
        us, w = us[keep], w[keep]
        if len(us) < 2:
            return
        if not path.closed:
            us = np.minimum(us, 1.0)
        pts = path.at(us)
        c1 = lin(color) if isinstance(color, str) else np.asarray(color, np.float32)
        if color_tail is not None:
            c0 = lin(color_tail) if isinstance(color_tail, str) else np.asarray(color_tail, np.float32)
            cols = c0[None] * (1 - w[:, None]) + c1[None] * w[:, None]
        else:
            cols = np.repeat(c1[None], len(us), 0)
        inten = energy * fade * (w ** 2)
        self.light_poly(pts, cols, inten, width, layer, occlude)
        if head_glow > 0 and head <= 1.0 + 1e-6:
            self.light_dot(path.at(min(head, 1.0)), c1, energy=1.25 * head_glow * fade,
                           radius=width * 0.8, layer=layer)

    # -- compositing ---------------------------------------------------------
    def composite(self, ui_shadow=(16, 10, 0.13)):
        """Collapse layers into one linear image (before camera / post)."""
        img = self.bg
        if self.box["under"]:
            self._ink(img, self.under, self.box["under"], bloom=False)
        ub = self.box["ui"]
        if ub:
            blur, off, op = ui_shadow if ui_shadow else (0, 0, 0)
            pad = int((blur * 3 + off) * self.S) + 8
            x0, y0 = max(0, ub[0] - pad), max(0, ub[1] - pad)
            x1, y1 = min(self.W, ub[2] + pad), min(self.H, ub[3] + pad)
            ui = self.ui[y0:y1, x0:x1]
            roi = img[y0:y1, x0:x1]
            if op > 0:
                f = 4
                hh, ww = max(1, (y1 - y0) // f), max(1, (x1 - x0) // f)
                small = cv2.resize(np.ascontiguousarray(ui[..., 3]), (ww, hh), interpolation=cv2.INTER_AREA)
                small = cv2.GaussianBlur(small, (0, 0), max(0.6, blur * self.S / f))
                sh = cv2.resize(small, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
                o = int(round(off * self.S))
                if o > 0:
                    sh = np.vstack([np.zeros((o, x1 - x0), np.float32), sh[:-o]])
                roi *= (1 - op * sh[..., None])
            roi *= (1 - ui[..., 3:4])
            roi += ui[..., :3]
        if self.box["over"]:
            self._ink(img, self.over, self.box["over"], bloom=True)
        return img

    def _bloom(self, L):
        S = self.S
        out = L.copy()
        for sigma, k in ((3.0, 1.0), (12.0, 0.75), (42.0, 0.5)):
            s = sigma * S
            f = 1
            while s / f > 6 and f < 16:
                f *= 2
            if f > 1:
                small = cv2.resize(L, (max(1, L.shape[1] // f), max(1, L.shape[0] // f)),
                                   interpolation=cv2.INTER_AREA)
                small = cv2.GaussianBlur(small, (0, 0), s / f)
                b = cv2.resize(small, (L.shape[1], L.shape[0]), interpolation=cv2.INTER_LINEAR)
            else:
                b = cv2.GaussianBlur(L, (0, 0), s)
            out += k * b
        return out

    def _ink(self, img, L, box, bloom=True):
        """Composite a light layer in place, only where it was drawn.

        On a light stage light becomes coloured ink (additive glow would vanish
        on white); on a dark stage it is added as glow."""
        pad = int(42 * self.S * 3) + 4 if bloom else 1
        x0, y0 = max(0, box[0] - pad), max(0, box[1] - pad)
        x1, y1 = min(self.W, box[2] + pad), min(self.H, box[3] + pad)
        if x1 <= x0 or y1 <= y0:
            return img
        Lr = L[y0:y1, x0:x1]
        B = self._bloom(Lr) if bloom else Lr
        roi = img[y0:y1, x0:x1]
        mx = B.max(-1)
        mask = mx > 1.5e-3
        frac = float(mask.mean())
        if frac == 0:
            return img
        if frac < 0.3:
            ys, xs = np.nonzero(mask)
            roi[ys, xs] = self._ink_px(roi[ys, xs], B[ys, xs], mx[ys, xs][:, None])
        else:
            roi[:] = self._ink_px(roi, B, mx[..., None])
        return img

    def _ink_px(self, r, b, m):
        if self.stage_mode == "dark":
            return 1 - (1 - r) * np.exp(-b)
        mn = b.min(-1, keepdims=True)
        A = 1 - np.exp(-2.3 * m)
        inv = 1.0 / np.maximum(m, 1e-6)
        sat = np.clip((m - mn) * inv * 1.8, 0, 1)
        ink = self.slate * (1 - sat) + b * inv * (0.82 * sat)
        return r * (1 - A) + ink * A


# ---------------------------------------------------------------------------
# stage, particles, post
# ---------------------------------------------------------------------------

def stage_background(cv: Canvas, t, base="#FAFAFB", tint="#3186FF", strength=0.16, pulse=0.0,
                     seed=7, dark=False):
    """Soft pastel blobs that drift and breathe: 62 % white + 38 % tint."""
    w, h = max(16, cv.W // 8), max(16, cv.H // 8)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    xx /= h
    yy /= h
    aspect = w / h
    r = np.random.default_rng(seed)
    field = np.zeros((h, w), np.float32)
    for i in range(5):
        cx0, cy0 = r.uniform(0.1, 0.9) * aspect, r.uniform(0.15, 0.85)
        ax, ay = r.uniform(0.05, 0.14), r.uniform(0.04, 0.12)
        fx, fy, ph = r.uniform(0.05, 0.11), r.uniform(0.04, 0.09), r.uniform(0, 6.28)
        cx = cx0 + ax * math.sin(t * fx * 6.28 + ph)
        cy = cy0 + ay * math.cos(t * fy * 6.28 + ph * 1.3)
        rad = r.uniform(0.28, 0.48) * (1 + 0.06 * math.sin(t * 0.5 + i))
        field += np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * rad * rad)) * r.uniform(0.6, 1.0)
    field = np.clip(field / 1.4, 0, 1)
    base_l = lin(base)
    k = clamp(strength + 0.10 * pulse, 0, 0.6)
    if dark:
        # near-black stage with soft coloured glows (added, never a wash)
        glow = lin(tint) * 0.22
        small = base_l[None, None] + glow[None, None] * (k * field[..., None] ** 1.6)
    else:
        blob = srgb_to_linear(0.62 * np.ones(3, np.float32) + 0.38 * hex_srgb(tint))
        small = base_l[None, None] * (1 - k * field[..., None]) + blob[None, None] * (k * field[..., None])
    cv.bg = cv2.resize(small.astype(np.float32), (cv.W, cv.H), interpolation=cv2.INTER_CUBIC)


class ParticleField:
    """Tiny streaks flying toward camera. ``speed(t)`` drives travel; it is
    integrated once on a fine grid so any frame is deterministic."""

    def __init__(self, W, H, duration, speed_fn, n=170, seed=11, colors=None):
        self.W, self.H = W, H
        r = np.random.default_rng(seed)
        self.p = np.stack([r.uniform(-1.6, 1.6, n), r.uniform(-1.0, 1.0, n), r.uniform(0.2, 4.0, n)], 1)
        self.e = r.uniform(0.18, 0.45, n)
        self.col = r.integers(0, 1 << 30, n)
        self.colors = colors or []
        self.dt = 1 / 240
        ts = np.arange(0, duration + 1, self.dt)
        self.travel = np.concatenate([[0], np.cumsum([speed_fn(x) * self.dt for x in ts])])[:len(ts)]
        self.ts = ts
        self.speed_fn = speed_fn

    def draw(self, cv: Canvas, t, fade=1.0, slate="#9AA0A6"):
        if fade <= 0.01:
            return
        tr = float(np.interp(t, self.ts, self.travel))
        v = self.speed_fn(t)
        z = np.mod(self.p[:, 2] - tr, 4.0) + 0.15
        z_prev = z + max(0.004, v * 0.03)
        f = 0.9 * self.H
        cx, cy = self.W / 2, self.H / 2
        sx, sy = cx + self.p[:, 0] * f / z, cy + self.p[:, 1] * f / z
        px, py = cx + self.p[:, 0] * f / z_prev, cy + self.p[:, 1] * f / z_prev
        near = clamp((4.0 - z) / 3.5)
        grey = lin(slate)
        for i in range(len(z)):
            if not (-50 < sx[i] < self.W + 50 and -50 < sy[i] < self.H + 50):
                continue
            col = grey
            if self.colors and self.col[i] % 3 == 0:
                col = lin(self.colors[self.col[i] % len(self.colors)])
            e = self.e[i] * near[i] * fade
            if e < 0.02:
                continue
            w = 1.2 + 2.2 * near[i]
            cv.light_poly([(px[i], py[i]), (sx[i], sy[i])], col, [e * 0.3, e], w, layer="under")


def post(cv: Canvas, img, t, frame_index, breath=0.008, shake=(0.0, 0.0), vignette=0.07,
         grain=0.005, grain_tiles=None):
    """Camera (breath + shake + push), vignette, gamma, grain -> uint8 RGB."""
    cam = cv.camera
    z = cam["zoom"] * (1 + breath * math.sin(t * 2 * math.pi / 7.0))
    dx = (shake[0] + cam["x"]) * cv.S
    dy = (shake[1] + cam["y"]) * cv.S
    if abs(z - 1) > 1e-5 or abs(dx) > 0.01 or abs(dy) > 0.01 or cam["rot"]:
        M = cv2.getRotationMatrix2D((cv.W / 2, cv.H / 2), cam["rot"], z)
        M[:, 2] += (dx, dy)
        img = cv2.warpAffine(img, M, (cv.W, cv.H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    if vignette > 0:
        img *= _vignette(cv.W, cv.H, vignette)
    out = linear_to_srgb(img)
    if grain > 0:
        out += _grain(cv.W, cv.H, frame_index, grain)
    out *= 255.0
    out += 0.5
    np.clip(out, 0, 255, out=out)
    return out.astype(np.uint8)


_CACHE = {}


def _vignette(W, H, amount):
    key = ("v", W, H, amount)
    if key not in _CACHE:
        yy, xx = np.ogrid[0:H, 0:W]
        r2 = ((xx - W / 2) / (W / 2)) ** 2 * 0.6 + ((yy - H / 2) / (H / 2)) ** 2 * 0.4
        _CACHE[key] = (1 - amount * r2).astype(np.float32)[..., None]
    return _CACHE[key]


def _grain(W, H, i, amount):
    """Fine monochrome film grain; 8 cached fields cycled per frame."""
    key = ("g", W, H, amount, i % 8)
    if key not in _CACHE:
        g = np.random.default_rng(1000 + i % 8).standard_normal((H, W)).astype(np.float32) * 0.33
        _CACHE[key] = (g * (amount * 3.0))[..., None]
    return _CACHE[key]
