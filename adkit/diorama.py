"""Diorama: the product's UI as a physical object on a table, shot with a real 3D camera.

A slab carrying the UI texture sits on a warm table. Every frame is ray-cast: each output
pixel's ray hits either the slab top or the table, so grazing camera angles, perspective and
depth are exact. On top of that: soft contact shadows, a practical key light, shallow depth of
field focused on the subject, lights lying on the UI (lanes, a playhead) and an optional 3D
mascot that hops across the interface with squash and stretch, a shadow and landing debris.

World units: the slab is ``SLAB_W`` wide, centred at the origin, on the z=0 table; +y points
"up the screen" of the UI, +z is up. Camera presets are functions of the scene's progress.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from .core import ease_in_out, ease_out, hex_srgb, lerp, lin, ramp, smooth, srgb_to_linear

SLAB_W = 16.0
SLAB_T = 0.32


def _poly_blend(img, p, col, a=1.0, blur=0):
    """Blend a filled polygon (pixel coords) into img, touching only its bounding box."""
    H, W = img.shape[:2]
    pad = 2 + blur * 3
    x0, y0 = max(0, int(np.floor(p[:, 0].min())) - pad), max(0, int(np.floor(p[:, 1].min())) - pad)
    x1, y1 = min(W, int(np.ceil(p[:, 0].max())) + pad + 1), min(H, int(np.ceil(p[:, 1].max())) + pad + 1)
    if x1 <= x0 or y1 <= y0 or a <= 0.002:
        return img
    m = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillConvexPoly(m, np.round((p - (x0, y0)) * 16).astype(np.int32), 255, cv2.LINE_AA, 4)
    mf = m.astype(np.float32) / 255
    if blur:
        mf = cv2.GaussianBlur(mf, (0, 0), blur)
    mf = (mf * a)[..., None]
    roi = img[y0:y1, x0:x1]
    if col is None:          # darken (shadows)
        roi *= (1 - mf)
    else:
        roi[:] = roi * (1 - mf) + np.asarray(col, np.float32) * mf
    return img


def _norm(v):
    v = np.asarray(v, np.float64)
    return v / (np.linalg.norm(v) + 1e-12)


class Camera:
    def __init__(self, eye, target, fov=42.0, W=1920, H=1080):
        self.eye = np.asarray(eye, np.float64)
        f = _norm(np.asarray(target, np.float64) - self.eye)
        r = _norm(np.cross(f, [0.0, 0.0, 1.0]))
        u = np.cross(r, f)
        self.f, self.r, self.u = f, r, u
        self.W, self.H = W, H
        self.fl = (W / 2) / math.tan(math.radians(fov) / 2)

    def project(self, P):
        """World points (n, 3) -> pixel coords (n, 2) and depth (n,). depth<=0 is behind."""
        P = np.atleast_2d(np.asarray(P, np.float64)) - self.eye
        x, y, z = P @ self.r, P @ self.u, P @ self.f
        zz = np.where(z > 1e-4, z, 1e-4)
        return np.stack([self.W / 2 + self.fl * x / zz, self.H / 2 - self.fl * y / zz], 1), z


# ---------------------------------------------------------------------------
# camera choreography (u = 0..1 over the scene)
# ---------------------------------------------------------------------------

def _keys(keys, u):
    """Smoothly interpolate keyframes [(u, eye, target, fov), ...] (smoothstep between keys)."""
    for (u0, e0, t0, f0), (u1, e1, t1, f1) in zip(keys, keys[1:]):
        if u <= u1:
            k = smooth(ramp(u, u0, u1))
            return lerp(np.array(e0, float), np.array(e1, float), k), lerp(np.array(t0, float), np.array(t1, float), k), \
                lerp(f0, f1, k)
    _, e, t, f = keys[-1]
    return np.array(e, float), np.array(t, float), f


def camera_path(preset, u, subject, slab_h):
    """Eye, target and fov for a preset. subject = the mascot (or focus) world point."""
    sx, sy = float(subject[0]), float(subject[1])
    hh = slab_h / 2
    if preset == "flyover":
        e = (lerp(-10, 9, u), -hh - 6.5, 3.2 + 0.6 * math.sin(u * math.pi))
        t = (lerp(-4, 13, u) * 0.7 + sx * 0.3, sy * 0.6, 0.4)
        return np.array(e), np.array(t), 44.0
    if preset == "orbit":
        a = math.radians(lerp(-48, 42, ease_in_out(u, 1.6)))
        R, z = 15.0, 7.5 - 2.0 * math.sin(u * math.pi)
        return np.array([R * math.sin(a), -R * math.cos(a), z]), np.array([sx * 0.4, sy * 0.4, 0.3]), 40.0
    if preset == "reveal":
        return _keys([(0.0, (sx - 2.4, sy - 5.4, 2.2), (sx + 0.6, sy + 1.2, 0.4), 46.0),
                      (0.55, (sx - 3.4, sy - 8.0, 3.6), (sx * 0.6, sy * 0.6, 0.3), 44.0),
                      (1.0, (0.0, -hh - 13.5, 11.5), (0.0, -0.6, 0.0), 40.0)], u)
    if preset == "follow":
        e = (sx - 4.0, sy - 8.0, 3.2)
        return np.array(e), np.array([sx + 1.0, sy + 1.4, 0.4]), 44.0
    # "swoop": high and wide, dive to grazing height tracking the subject, then pull back
    return _keys([(0.0, (-3.0, -hh - 13.0, 12.5), (0.0, 0.0, 0.0), 40.0),
                  (0.35, (sx - 4.6, sy - 8.8, 3.4), (sx + 1.0, sy + 1.6, 0.4), 42.0),
                  (0.72, (sx - 3.4, sy - 7.6, 2.8), (sx + 1.8, sy + 1.2, 0.4), 44.0),
                  (1.0, (2.5, -hh - 12.0, 9.5), (0.5, 0.0, 0.0), 40.0)], u)


# ---------------------------------------------------------------------------
# the set
# ---------------------------------------------------------------------------

class Diorama:
    def __init__(self, ui_rgba, W, H, table="#3A2216", key_light="#FFB37A", seed=1):
        """ui_rgba: uint8 RGBA (or RGB) image of the interface, any size."""
        self.W, self.H = int(W), int(H)
        img = np.asarray(ui_rgba)
        if img.shape[2] == 3:
            img = np.concatenate([img, np.full(img.shape[:2] + (1,), 255, np.uint8)], 2)
        self.tex = srgb_to_linear(img[..., :3].astype(np.float32) / 255.0)
        self.th, self.tw = self.tex.shape[:2]
        self.slab_w = SLAB_W
        self.slab_h = SLAB_W * self.th / self.tw
        self.table_tex = self._make_table(table, seed)
        self.key = lin(key_light)
        # pixel grid (half resolution; maps are smooth so they upsample cleanly)
        hw, hh = max(8, self.W // 2), max(8, self.H // 2)
        ys, xs = np.mgrid[0:hh, 0:hw].astype(np.float32)
        self.gx = (xs + 0.5) * (self.W / hw) - self.W / 2
        self.gy = (ys + 0.5) * (self.H / hh) - self.H / 2

    def _make_table(self, hexcol, seed, n=1024, span=72.0):
        """A warm wood-like table: low-frequency tone, long grain, fine noise."""
        r = np.random.default_rng(seed)
        base = srgb_to_linear(hex_srgb(hexcol))
        y, x = np.mgrid[0:n, 0:n].astype(np.float32) / n
        tone = cv2.GaussianBlur(r.standard_normal((n, n)).astype(np.float32), (0, 0), n / 18) * 6
        grain = cv2.GaussianBlur(r.standard_normal((n, n)).astype(np.float32), (0, 0), sigmaX=n / 6, sigmaY=1.2)
        grain = grain / (np.abs(grain).max() + 1e-6)
        fine = r.standard_normal((n, n)).astype(np.float32) * 0.03
        k = 1.0 + 0.18 * tone + 0.22 * grain + fine
        self.table_span = span
        return np.clip(base[None, None, :] * k[..., None], 0, 1).astype(np.float32)

    # -- rendering -------------------------------------------------------------
    def render(self, cam: Camera, focus_depth, dof=1.0, sheen=None):
        """Linear RGB image (H, W, 3) and depth map (H, W)."""
        e, f, r, u, fl = cam.eye, cam.f, cam.r, cam.u, cam.fl
        gx, gy = self.gx, self.gy
        dx = f[0] * fl + r[0] * gx - u[0] * gy
        dy = f[1] * fl + r[1] * gx - u[1] * gy
        dz = f[2] * fl + r[2] * gx - u[2] * gy
        hs = SLAB_T
        # slab top
        with np.errstate(divide="ignore", invalid="ignore"):
            ts = (hs - e[2]) / dz
        sx, sy = e[0] + ts * dx, e[1] + ts * dy
        uu = sx / self.slab_w + 0.5
        vv = 0.5 - sy / self.slab_h
        on_slab = (ts > 0) & (uu >= 0) & (uu <= 1) & (vv >= 0) & (vv <= 1)
        # table
        with np.errstate(divide="ignore", invalid="ignore"):
            tt = (0.0 - e[2]) / dz
        tx, ty = e[0] + tt * dx, e[1] + tt * dy
        hit_table = tt > 0
        depth = np.where(on_slab, ts * fl, np.where(hit_table, tt * fl, 1e6)).astype(np.float32)

        def up(a, interp=cv2.INTER_LINEAR):
            return cv2.resize(a.astype(np.float32), (self.W, self.H), interpolation=interp)

        mapx_s = up(np.clip(uu, 0, 1) * (self.tw - 1))
        mapy_s = up(np.clip(vv, 0, 1) * (self.th - 1))
        slab_m = up(on_slab.astype(np.float32))
        span = self.table_span
        n = self.table_tex.shape[0]
        mapx_t = up(np.mod((tx / span + 0.5) * n, n))
        mapy_t = up(np.mod((ty / span + 0.5) * n, n))
        D = up(np.minimum(depth, 1e5))
        ui = cv2.remap(self.tex, mapx_s, mapy_s, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        tab = cv2.remap(self.table_tex, mapx_t, mapy_t, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        # table lighting: warm key pool around the set, darkening with distance; contact shadow
        TX, TY = up(np.where(hit_table, tx, 1e3)), up(np.where(hit_table, ty, 1e3))
        rr = np.sqrt((TX / (self.slab_w * 0.9)) ** 2 + (TY / (self.slab_h * 1.3)) ** 2)
        pool = 0.18 + 1.1 * np.exp(-rr ** 2 * 1.6)
        ox = np.maximum(np.abs(TX) - self.slab_w / 2, 0)
        oy = np.maximum(np.abs(TY) - self.slab_h / 2, 0)
        dist = np.sqrt(ox ** 2 + oy ** 2)
        shadow = 1 - 0.78 * np.exp(-dist / 0.55)
        tab = tab * (pool * shadow)[..., None] * (0.55 + 0.45 * self.key)[None, None, :]
        # UI lighting: soft gradient + a moving sheen band
        U = mapx_s / max(1, self.tw - 1)
        V = mapy_s / max(1, self.th - 1)
        light = 0.74 + 0.22 * (1 - V) * (1 - 0.4 * U)
        ui = ui * light[..., None]
        if sheen is not None:  # a soft specular sweep, added (not multiplied) so whites don't clip hard
            band = 0.10 * np.exp(-(((U - V * 0.35) - sheen) / 0.06) ** 2)
            ui = ui + band[..., None] * self.key
        img = tab * (1 - slab_m[..., None]) + ui * slab_m[..., None]
        img = self._slab_sides(img, cam)
        # far fog into the dark
        fog = np.clip((D - 12 * fl) / (60 * fl), 0, 1)[..., None]
        img = img * (1 - 0.85 * fog)
        self.depth = D
        if dof > 0:
            img = self._dof(img, D, focus_depth, dof)
        return img.astype(np.float32)

    def _slab_sides(self, img, cam):
        """Dark edges of the slab (thickness) facing the camera."""
        w, h, t = self.slab_w / 2, self.slab_h / 2, SLAB_T
        faces = [((-w, -h), (w, -h), (0, -1)), ((w, -h), (w, h), (1, 0)), ((w, h), (-w, h), (0, 1)),
                 ((-w, h), (-w, -h), (-1, 0))]
        out = img
        for a, b, nrm in faces:
            if (cam.eye[0] - a[0]) * nrm[0] + (cam.eye[1] - a[1]) * nrm[1] <= 0:
                continue
            P = np.array([[a[0], a[1], t], [b[0], b[1], t], [b[0], b[1], 0.0], [a[0], a[1], 0.0]])
            p, z = cam.project(P)
            if (z <= 0.05).any():
                continue
            col = np.array([0.016, 0.014, 0.016], np.float32) * (1.4 if nrm[1] < 0 else 1.0)
            out = _poly_blend(out, p, col)
        return out

    def _dof(self, img, D, focus, amount):
        coc = np.abs(1.0 - focus / np.maximum(D, 1e-3)) * amount
        f = 2
        small = cv2.resize(img, (self.W // f, self.H // f), interpolation=cv2.INTER_AREA)
        b1 = cv2.resize(cv2.GaussianBlur(small, (0, 0), 2.2 * self.W / 1920 * 2 / f), (self.W, self.H))
        b2 = cv2.resize(cv2.GaussianBlur(small, (0, 0), 7.0 * self.W / 1920 * 2 / f), (self.W, self.H))
        c = np.clip(coc, 0, 1)[..., None]
        k1 = np.clip((c - 0.08) / 0.32, 0, 1)
        k2 = np.clip((c - 0.42) / 0.6, 0, 1)
        return img * (1 - k1) + b1 * k1 * (1 - k2) + b2 * k2

    # -- helpers for things living on the set --------------------------------------
    def world(self, uv, z=SLAB_T):
        """UI-normalised point (u, v in 0-1, v down) -> world xyz on the slab top."""
        return np.array([(uv[0] - 0.5) * self.slab_w, (0.5 - uv[1]) * self.slab_h, z])


# ---------------------------------------------------------------------------
# the mascot: a little 3D cube creature
# ---------------------------------------------------------------------------

def draw_cube_mascot(img, cam: Camera, pos, size, yaw, color_hex, squash=0.0, blink=False, depth=None):
    """Draw a shaded cube with eyes and feet at world pos (centre of its base) into img (linear RGB)."""
    col = lin(color_hex)
    s = size / 2
    sx_ = s * (1 + 0.28 * squash)
    sz_ = size * (1 - 0.38 * squash)
    c, sn = math.cos(yaw), math.sin(yaw)

    def W(px, py, pz):
        return np.array([pos[0] + px * c - py * sn, pos[1] + px * sn + py * c, pos[2] + pz])

    V = [W(x * sx_, y * sx_, z) for z in (0.08 * size, sz_ + 0.08 * size) for x, y in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    faces = [((0, 1, 2, 3), (0, 0, -1)), ((4, 5, 6, 7), (0, 0, 1)), ((0, 1, 5, 4), (0, -1, 0)),
             ((1, 2, 6, 5), (1, 0, 0)), ((2, 3, 7, 6), (0, 1, 0)), ((3, 0, 4, 7), (-1, 0, 0))]
    Lt = _norm([-0.5, -0.6, 0.85])
    p, z = cam.project(np.array(V))
    if (z <= 0.05).any():
        return img
    H, W_ = img.shape[:2]
    lo = np.floor(p.min(0)).astype(int) - 40
    hi = np.ceil(p.max(0)).astype(int) + 40
    x0, y0 = max(0, lo[0]), max(0, lo[1])
    x1, y1 = min(W_, hi[0]), min(H, hi[1])
    if x1 <= x0 or y1 <= y0:
        return img
    roi = img[y0:y1, x0:x1].copy()
    order = []
    for idx, n in faces:
        nw = np.array([n[0] * c - n[1] * sn, n[0] * sn + n[1] * c, n[2]])
        cen = np.mean([V[i] for i in idx], 0)
        if np.dot(nw, cam.eye - cen) <= 0:
            continue
        order.append((np.linalg.norm(cam.eye - cen), idx, nw))
    order.sort(key=lambda o: -o[0])
    for _, idx, nw in order:
        shade = 0.42 + 0.75 * max(0.0, float(np.dot(nw, Lt)))
        poly = np.round((p[list(idx)] - (x0, y0)) * 16).astype(np.int32)
        m = np.zeros(roi.shape[:2], np.uint8)
        cv2.fillConvexPoly(m, poly, 255, cv2.LINE_AA, 4)
        mf = m.astype(np.float32)[..., None] / 255
        roi = roi * (1 - mf) + (col * shade) * mf
    # face features on the front (-y in body space) when visible
    nw = np.array([0 * c - (-1) * sn, 0 * sn + (-1) * c, 0])
    if np.dot(nw, cam.eye - W(0, -sx_, sz_ / 2)) > 0:
        for ex in (-0.42, 0.42):
            eh = 0.05 if blink else 0.22
            pts = [W(ex * sx_ - 0.07 * size, -sx_ - 0.002, sz_ * 0.62 - eh * size / 2),
                   W(ex * sx_ + 0.07 * size, -sx_ - 0.002, sz_ * 0.62 - eh * size / 2),
                   W(ex * sx_ + 0.07 * size, -sx_ - 0.002, sz_ * 0.62 + eh * size / 2),
                   W(ex * sx_ - 0.07 * size, -sx_ - 0.002, sz_ * 0.62 + eh * size / 2)]
            q, _ = cam.project(np.array(pts))
            m = np.zeros(roi.shape[:2], np.uint8)
            cv2.fillConvexPoly(m, np.round((q - (x0, y0)) * 16).astype(np.int32), 255, cv2.LINE_AA, 4)
            mf = m.astype(np.float32)[..., None] / 255
            roi = roi * (1 - mf) + np.array([0.008, 0.006, 0.006], np.float32) * mf
    img[y0:y1, x0:x1] = roi
    return img


def draw_shadow(img, cam, pos, radius, strength=0.55):
    pts = np.array([[pos[0] + radius * math.cos(a), pos[1] + radius * 0.8 * math.sin(a), SLAB_T + 0.001]
                    for a in np.linspace(0, 2 * math.pi, 28)])
    p, z = cam.project(pts)
    if (z <= 0.05).any():
        return img
    bw = float(p[:, 0].max() - p[:, 0].min())
    return _poly_blend(img, p, None, strength, blur=max(1, int(bw * 0.12)))


def draw_debris(img, cam, origin, t_since, seed, color="#F2F2F2", n=12, size=0.16):
    """White chunks bursting from a landing, under gravity, fading out."""
    if t_since < 0 or t_since > 0.9:
        return img
    r = np.random.default_rng(seed)
    col = lin(color)
    H, W = img.shape[:2]
    for i in range(n):
        ang = r.uniform(0, 2 * math.pi)
        sp = r.uniform(1.8, 4.2)
        vz = r.uniform(2.0, 4.5)
        t = t_since
        P = np.array([origin[0] + math.cos(ang) * sp * t, origin[1] + math.sin(ang) * sp * t,
                      max(SLAB_T + 0.02, origin[2] + vz * t - 4.9 * t * t)])
        s = size * r.uniform(0.6, 1.3) * (1 - 0.6 * t / 0.9)
        rot = r.uniform(0, 6) + t * r.uniform(4, 9)
        quad = np.array([[P[0] + s * math.cos(rot + k * math.pi / 2), P[1] + s * math.sin(rot + k * math.pi / 2),
                          P[2] + s * 0.6 * math.sin(rot + k)] for k in range(4)])
        p, z = cam.project(quad)
        if (z <= 0.05).any():
            continue
        a = (1 - t / 0.9) ** 0.7
        img = _poly_blend(img, p, col * (0.65 + 0.35 * (i % 2)), a)
    return img
