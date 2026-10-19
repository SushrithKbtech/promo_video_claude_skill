"""The film: one continuous light, from the hook to the end card.

Skeleton (seconds at 100 BPM, bar = 2.4 s, downbeats D(k) = 0.60 + 2.4 k):

    0.00-D0   hook: particles rush, one light per logo facet sweeps in on a Bezier
    D0        SLAM: the logo lands, facets punch inward on a spring, shards, shake
    D0+0.02   the rim light starts its first lap and never stops
    D0+0.52   the icon slides into the lockup; a light writes the wordmark
    D1-0.38   peel: the first act's light leaves its facet and lands in act 1
    D1..      acts (scenes). Each light does one job; hand-overs recolour the tail
    HIT-1.1   race: every light leaves where it is and races back to its facet
    HIT       LOGO HIT on a downbeat (the song's drop): same slam, smaller
    HIT+0.6   lockup + written wordmark; HIT+1.95 fade; rim head becomes a dot
    HIT+2.4   end card: tri-colour line with a glint, credit or call to action

Picture and sound read the same Film object, so every cue lands on its frame.
"""
from __future__ import annotations

import json
import math
import os

import cv2
import numpy as np

from .brand import Logo, Typesetter, resolve_fonts
from .core import (Path, clamp, damped, ease_in, ease_in_out, ease_out, hash01, hex_srgb, lerp,
                   lin, mix_hex, pulse, ramp, rng, smooth, smoother, window)
from .gfx import Canvas, ParticleField, post, stage_background
from .ui import Theme

ASPECTS = {"16:9": (1920, 1080), "9:16": (1080, 1920), "1:1": (1080, 1080), "4:5": (1080, 1350)}

# A format is a film shape. It only fills in what the brief does not set itself.
FORMATS = {
    # logo first, the light does every job, races back into the logo (the Drive spec ad shape)
    "journey": dict(opening="slam", hook="converge", ending="race", transitions=["light"]),
    # straight into the product, calm, real UI, understated logo reveal at the very end
    "keynote": dict(opening="cold", hook="converge", ending="reveal", transitions=["light", "cut", "light"]),
    # energy: logo hit up front, whips and cuts on the beat, hard cut to the logo
    "launch": dict(opening="slam", hook="trace", ending="cut", transitions=["whip", "cut", "light", "cut"]),
    # human, voice-led: tension -> product -> resolution, the name is written at the end
    "story": dict(opening="cold", hook="converge", ending="write", transitions=["light", "light", "cut"]),
    # a physical product: dark, slow, the product first, logo revealed by light
    "showcase": dict(opening="cold", hook="trace", ending="reveal", transitions=["cut", "light"]),
    # vertical social reel: fast, captions, whips, hard cut to the logo
    "reel": dict(opening="slam", hook="trace", ending="cut", transitions=["whip", "cut"], captions="on"),
}
ENDINGS = ("race", "reveal", "cut", "write")


class Film:
    def __init__(self, project_dir, scale=1.0, fps=60, brief=None):
        self.dir = os.path.abspath(project_dir)
        if brief is None:
            with open(os.path.join(self.dir, "brief.json"), encoding="utf-8") as f:
                brief = json.load(f)
        fmt = FORMATS.get(brief.get("format", ""), {})
        brief = {**{k: v for k, v in fmt.items() if k != "transitions"}, **brief}
        self.format = brief.get("format", "custom")
        self.format_transitions = fmt.get("transitions")
        self.brief = b = brief
        self.fps = fps
        self.DW, self.DH = ASPECTS.get(b.get("aspect", "16:9"), (1920, 1080))
        self.S = float(scale)
        self.W = int(round(self.DW * self.S / 2)) * 2
        self.H = int(round(self.DH * self.S / 2)) * 2
        self.S = self.W / self.DW
        self.CX, self.CY = self.DW / 2, self.DH / 2
        self.portrait = self.DH > self.DW * 1.1

        st = b.get("stage", {})
        self.stage_mode = st.get("mode", "light")
        self.stage_color = st.get("color", "#FAFAFB" if self.stage_mode == "light" else "#05060A")
        self.ink = b.get("ink", {})
        self.text_color = self.ink.get("text", "#1F1F1F" if self.stage_mode == "light" else "#F5F6F7")
        self.text2 = self.ink.get("secondary", "#5F6368" if self.stage_mode == "light" else "#9AA0A6")
        self.slate = self.ink.get("slate", "#5F6368" if self.stage_mode == "light" else "#C9CDD2")

        self.fonts = resolve_fonts(b.get("font", {}), self.dir)
        self.type = Typesetter(self.fonts, self.S)
        brand = b.get("brand", {})
        self.name = brand.get("name", "Product")
        self.wordmark = brand.get("wordmark", self.name)
        self.wordmark_weight = int(brand.get("wordmark_weight", 400))

        lg = brand.get("logo")
        if not lg:
            raise ValueError("brief.brand.logo is required (SVG preferred, PNG ok)")
        lg_path = lg if os.path.isabs(lg) else os.path.join(self.dir, lg)
        self.logo = Logo(lg_path, design_h=float(brand.get("logo_height", 380)),
                         facets=brand.get("facets", "auto"), k=brand.get("facet_count"),
                         cache_dir=os.path.join(self.dir, ".cache"))
        if brand.get("facet_lights"):
            for f_, c_ in zip(self.logo.facets, brand["facet_lights"]):
                f_["color"] = c_
            self.logo.colors = [f_["color"] for f_ in self.logo.facets]
        self.facet_colors = self.logo.colors
        palette = b.get("lights") or self.facet_colors
        if len(palette) < 2:
            palette = list(palette) + brand.get("palette", ["#3186FF", "#0EBC5F", "#FEC700"])
        self.palette = palette
        self.primary = brand.get("primary", palette[0])
        self.theme = Theme.from_brand(brand.get("ui_accent", self.primary), b.get("ui", {}),
                                      dark=self.stage_mode == "dark")
        self._sprites = {}

        # ---- timing grid -------------------------------------------------
        self.bpm = float(b.get("bpm", 100))
        self.bar = 240.0 / self.bpm
        self.beat = self.bar / 4
        self.d0 = 0.60
        dur = float(b.get("duration", 21.8))
        self.hit_bar = max(4, int(round((dur - 4.4 - self.d0) / self.bar)))
        self.slam = self.D(0)
        self.hit = self.D(self.hit_bar)
        self.signature = b.get("signature", "none")      # the shape the light takes (motifs.py)
        self.ending = b.get("ending", "race")
        if self.ending not in ENDINGS:
            self.ending = "race"
        # how long the stage is clear before the logo: the race needs time, a cut needs none
        self.race_t = self.hit - {"race": 1.1, "cut": 0.0, "reveal": 0.55, "write": 0.45}[self.ending]
        self.drop = self.D(3) if self.hit_bar >= 6 else self.D(2) if self.hit_bar >= 5 else self.D(1) + self.bar
        self.lock2 = self.hit + 0.6
        self.fade_t = self.hit + 1.95
        self.end_t = self.hit + 2.4
        self.duration = self.end_t + float(b.get("end_card", {}).get("hold", 2.0))
        self.nframes = int(round(self.duration * fps))
        self.opening = b.get("opening", "slam")          # "slam" (logo first) | "cold" (logo only at the end)
        self.cold = self.opening == "cold"
        self.acts_start = self.d0 if self.cold else self.D(1)

        # ---- scenes ------------------------------------------------------
        from .scenes import build_scenes
        self.scenes = build_scenes(self, b.get("acts", []))
        self._lockup_layout()
        self.particles = ParticleField(self.DW, self.DH, self.duration + 1, self._particle_speed,
                                       colors=self.facet_colors, n=150 if not self.portrait else 170)
        self.voice = self._load_voice()

    # ------------------------------------------------------------------
    def D(self, k):
        return self.d0 + k * self.bar

    def sprite(self, key, fn):
        if key not in self._sprites:
            self._sprites[key] = fn()
        return self._sprites[key]

    @property
    def ppd(self):
        return self.S

    def light_color(self, i):
        return self.palette[i % len(self.palette)]

    # ------------------------------------------------------------------
    # anchors: named moments the voice and sound can be pinned to
    # ------------------------------------------------------------------
    def anchors(self):
        a = {"slam": self.slam, "lockup": self.slam + 1.2, "drop": self.drop, "race": self.race_t,
             "hit": self.hit, "end": self.end_t}
        for i, sc in enumerate(self.scenes):
            for k, v in sc.anchors().items():
                a[f"act{i + 1}.{k}"] = v
            a[f"act{i + 1}"] = sc.a
        return a

    def resolve_time(self, at):
        if isinstance(at, (int, float)):
            return float(at)
        s = str(at).replace(" ", "")
        off = 0.0
        for sign in ("+", "-"):
            if sign in s[1:]:
                i = s.rfind(sign)
                base, off = s[:i], float(s[i:])
                s = base
                break
        return self.anchors()[s] + off

    # ------------------------------------------------------------------
    # logo transform over the film
    # ------------------------------------------------------------------
    def _lockup_layout(self):
        """Where the icon and wordmark sit in the lockup (both lockups)."""
        icon_h = 230.0 if not self.portrait else 210.0
        size = 132.0 if not self.portrait else 112.0
        tw = self.type.width(self.wordmark, size, self.wordmark_weight)
        iw = icon_h * self.logo.aspect
        gap = icon_h * 0.2
        total = iw + gap + tw
        if total > self.DW * 0.86:
            # stacked lockup for narrow frames
            self.stacked = True
            size = min(size, self.DW * 0.8 / max(1.0, tw / size))
            tw = self.type.width(self.wordmark, size, self.wordmark_weight)
            self.lock_icon = (self.CX, self.CY - size * 0.75, icon_h)
            self.lock_text = (self.CX - tw / 2, self.CY + icon_h * 0.5 + size * 0.55, size, tw)
        else:
            self.stacked = False
            x0 = self.CX - total / 2
            self.lock_icon = (x0 + iw / 2, self.CY, icon_h)
            self.lock_text = (x0 + iw + gap, self.CY + size * 0.36, size, tw)

    def logo_xf(self, t):
        """(cx, cy, height, alpha, blur, punch) of the icon at time t, or None."""
        base_h = self.logo.design_h
        if t < self.slam:
            return None
        if t < self.D(1) + 0.25:
            if self.cold:
                return None
            return self._slam_xf(t, self.slam, base_h, 38.0, 0.0)
        if t < self.hit:
            return None
        if t < self.end_t + 0.05:
            if self.ending == "race":
                return self._slam_xf(t, self.hit, base_h * 0.86, 30.0, 1.0)
            if self.ending == "cut":
                return self._slam_xf(t, self.hit, base_h * 0.92, 14.0, 1.0)
            return self._reveal_xf(t, base_h * 0.86)
        return None

    def _reveal_xf(self, t, h0):
        """Understated endings: the icon focuses in (reveal) or arrives after the name (write)."""
        ix, iy, ih = self.lock_icon
        if self.ending == "write":
            t0 = self.hit + 0.75
            a = smooth(ramp(t, t0, t0 + 0.45))
            k = ease_out(ramp(t, t0, t0 + 0.6), 2.6)
            h = ih * (0.88 + 0.12 * k)
            return dict(cx=ix, cy=iy, h=h, alpha=a * (1 - smooth(ramp(t, self.fade_t, self.end_t))),
                        blur=6 * (1 - k), punch=0.0, t0=self.hit, soft=True)
        e = ease_out(ramp(t, self.hit, self.hit + 0.9), 2.4)
        h = h0 * (0.93 + 0.07 * e)
        cx, cy = self.CX, self.CY
        k = smoother(ramp(t, self.hit + 1.0, self.hit + 1.7))
        cx, cy, h = lerp(cx, ix, k), lerp(cy, iy, k), lerp(h, ih, k)
        alpha = smooth(ramp(t, self.hit, self.hit + 0.6)) * (1 - smooth(ramp(t, self.fade_t, self.end_t)))
        return dict(cx=cx, cy=cy, h=h, alpha=alpha, blur=9 * (1 - e), punch=0.0, t0=self.hit, soft=True)

    def _slam_xf(self, t, t0, h0, punch, second):
        ix, iy, ih = self.lock_icon
        lt = t - t0
        sc = 1.0 + 0.10 * max(0.0, damped(lt, 3.2, 0.55)) if lt >= 0 else 1.25
        cx, cy, h = self.CX, self.CY, h0 * sc
        if lt < 0.52 + 0.08 * second:
            pass
        else:
            k = smoother(ramp(t, t0 + 0.52 + 0.08 * second, t0 + 1.52 - 0.42 * second))
            cx, cy, h = lerp(self.CX, ix, k), lerp(self.CY, iy, k), lerp(h0, ih, k)
        alpha, blur = 1.0, 0.0
        if not second:
            # recede before act 1: -16 % scale, up 120 px, depth blur, fade by D1+0.2
            k = smooth(ramp(t, self.D(1) - 0.38, self.D(1) + 0.2))
            h *= 1 - 0.16 * k
            cy -= 120 * k
            blur = 7 * k
            alpha = 1 - smooth(ramp(t, self.D(1) - 0.1, self.D(1) + 0.2))
        else:
            alpha = 1 - smooth(ramp(t, self.fade_t, self.end_t))
        return dict(cx=cx, cy=cy, h=h, alpha=alpha, blur=blur, punch=punch * damped(lt, 6.0, 0.35), t0=t0)

    # ------------------------------------------------------------------
    # drawing the logo, its rim light, wordmark, shards
    # ------------------------------------------------------------------
    def draw_logo(self, cv: Canvas, t):
        xf = self.logo_xf(t)
        if xf is None or xf["alpha"] <= 0.003:
            return
        lt = t - xf["t0"]
        k = xf["h"] / self.logo.design_h
        shutter = 1.0 / self.fps
        if xf.get("soft"):
            cv.draw(self.logo.sprite, xf["cx"], xf["cy"], scale=k, alpha=xf["alpha"], blur=xf["blur"])
            t_rim = self.hit if self.ending == "reveal" else self.hit + 0.75
            if t >= t_rim:
                u = ease_out(ramp(t, t_rim, t_rim + 1.3), 2.0) + 0.3 * max(0.0, t - t_rim - 1.3)
                fade = xf["alpha"] if self.ending == "write" else 1 - smooth(ramp(t, self.fade_t, self.fade_t + 0.25))
                self._contour_light(cv, self._contour_world(xf, k), u, 0.45 if u < 1 else 0.22, 1.1 * fade,
                                    3.6 * max(0.7, k))
            return

        def facet_pos(f, tt):
            x2 = self._slam_xf(tt, xf["t0"], self.logo.design_h * (0.86 if xf["t0"] == self.hit else 1.0),
                               38.0 if xf["t0"] == self.slam else 30.0, xf["t0"] == self.hit)
            kk = x2["h"] / self.logo.design_h
            c = f["center"]
            n = c / (np.linalg.norm(c) + 1e-6)
            off = -n * x2["punch"] if len(self.logo.facets) > 1 else 0
            # arrival: facets come in from slightly outside during the first frame
            arr = 1.0 - smooth(ramp(tt, xf["t0"] - shutter, xf["t0"] + 0.5 * shutter))
            off = off + n * 90 * arr
            sc = kk * (1 + 0.15 * arr)
            return dict(x=x2["cx"] + (c[0] + (off[0] if np.ndim(off) else 0)) * kk,
                        y=x2["cy"] + (c[1] + (off[1] if np.ndim(off) else 0)) * kk,
                        scale=sc, alpha=x2["alpha"], blur=x2["blur"])

        moving = lt < 0.45 or (0.52 <= lt <= 1.6)
        for f in self.logo.facets:
            sp = f["sprite"]
            ax = 0.5 + f["center"][0] / sp.w
            ay = 0.5 + f["center"][1] / sp.h
            if moving:
                cv.draw_motion(sp, lambda tt, f=f: facet_pos(f, tt), t, shutter * 0.75,
                               n=7 if lt < 0.08 else 4, anchor=(ax, ay))
            else:
                p = facet_pos(f, t)
                cv.draw(sp, p["x"], p["y"], scale=p["scale"], alpha=p["alpha"], blur=p["blur"], anchor=(ax, ay))
        self._draw_rim(cv, t, xf, k)
        self._draw_shards(cv, t, xf, k)

    def _contour_world(self, xf, k):
        if "cx" in xf and "h" not in xf:
            return np.stack([xf["cx"] + self.logo.contour[:, 0] * k, xf["cy"] + self.logo.contour[:, 1] * k], 1)
        return np.stack([xf["cx"] + self.logo.contour[:, 0] * k, xf["cy"] + self.logo.contour[:, 1] * k], 1)

    def rim_head(self, t):
        """Arc fraction of the rim light head (first lap ease-out, then 0.32 laps/s)."""
        t0 = self.slam + 0.02
        lt = t - t0
        if lt < 0:
            return None
        if lt < 1.05:
            return ease_out(lt / 1.05, 2.2)
        return 1.0 + (lt - 1.05) * 0.32

    def _draw_rim(self, cv, t, xf, k):
        u = self.rim_head(t)
        if u is None:
            return
        fade = xf["alpha"]
        if xf["t0"] == self.hit:
            fade *= 1.0  # head detaches at fade_t (see end card)
            if t > self.fade_t:
                fade *= 1 - smooth(ramp(t, self.fade_t, self.fade_t + 0.25))
        pts = self._contour_world(xf, k)
        self._contour_light(cv, pts, u, 0.22, 1.05 * fade, 3.6 * max(0.7, k))

    def _contour_light(self, cv, pts, u, tail, energy, width, lap_tail=False):
        """A light running a closed contour, coloured by the logo underneath."""
        if energy <= 0.003:
            return
        path = Path(pts, closed=True)
        n = 160
        us = np.linspace(u - tail, u, n)
        w = (us - (u - tail)) / tail
        keep = us >= 0  # never show more tail than has been travelled
        if lap_tail and u > 1:
            keep &= us <= 1.0
        us, w = us[keep], w[keep]
        if len(us) < 2:
            return
        P = path.at(us)
        cc = self.logo.contour_colors
        idx = (np.mod(us, 1.0) * len(cc)).astype(int) % len(cc)
        cols = np.array([lin_from_srgb(c) for c in cc[idx]])
        cv.light_poly(P, cols, energy * w ** 2, width=width)
        if u <= 1.0 or not lap_tail:
            cv.light_dot(path.at(u), cols[-1], energy=1.25 * energy, radius=width * 0.9)

    def _shards(self, t0, scale_n, k):
        key = ("shards", t0)
        if key in self._sprites:
            return self._sprites[key]
        r = rng(int(t0 * 1000) + 5)
        n = int(140 * scale_n)
        c = self.logo.contour
        idx = r.integers(0, len(c), n)
        p0 = c[idx]
        nrm = p0 / (np.linalg.norm(p0, axis=1, keepdims=True) + 1e-6)
        ang = r.normal(0, 0.45, n)
        ca, sa = np.cos(ang), np.sin(ang)
        d = np.stack([nrm[:, 0] * ca - nrm[:, 1] * sa, nrm[:, 0] * sa + nrm[:, 1] * ca], 1)
        v = r.uniform(250, 1400, n)
        life = r.uniform(0.35, 0.9, n)
        cols = self.logo.contour_colors[idx]
        out = dict(p0=p0, d=d, v=v, life=life, cols=cols, w=r.uniform(2.0, 4.2, n))
        self._sprites[key] = out
        return out

    def _draw_shards(self, cv, t, xf, k):
        t0 = xf["t0"]
        lt = t - t0
        if lt < 0 or lt > 0.95:
            return
        sh = self._shards(t0, 1.0 if t0 == self.slam else 0.7 if self.ending == "race" else 0.45, k)
        base = self.logo.design_h * (1.0 if t0 == self.slam else 0.86) / self.logo.design_h
        drag = 6.0
        for i in range(len(sh["v"])):
            if lt > sh["life"][i]:
                continue
            fade = (1 - lt / sh["life"][i]) ** 1.5
            dist = sh["v"][i] * (1 - math.exp(-drag * lt)) / drag
            vel = sh["v"][i] * math.exp(-drag * lt)
            p = np.array([self.CX, self.CY]) + sh["p0"][i] * base + sh["d"][i] * dist
            q = p - sh["d"][i] * max(2.0, vel / self.fps * 1.2)
            col = lin_from_srgb(sh["cols"][i])
            cv.light_poly([q, p], col, [0.2 * fade, 1.1 * fade], width=sh["w"][i])

    def draw_wordmark(self, cv: Canvas, t):
        t2 = self.hit + (0.1 if self.ending == "write" else 1.15 if self.ending == "reveal" else 0.92)
        for t_start, second in ((self.slam + 0.98, False), (t2, True)):
            t_end = t_start + 0.56
            if t < t_start - 0.05:
                continue
            if not second and (t > self.D(1) + 0.25 or self.cold):
                continue
            if second and t > self.end_t:
                continue
            x0, base_y, size, tw = self.lock_text
            sp = self.type.text(self.wordmark, size, self.wordmark_weight,
                                self.brief.get("brand", {}).get("wordmark_color", self.text_color))
            prog = ease_in_out(ramp(t, t_start, t_end), 2.2)
            edge = x0 - 40 + (tw + 80) * prog
            rise = 14 * (1 - ease_out(ramp(t, t_start, t_end + 0.2)))
            alpha, blur, dy, sc = 1.0, 0.0, 0.0, 1.0
            if not second:
                kk = smooth(ramp(t, self.D(1) - 0.38, self.D(1) + 0.2))
                sc, dy, blur = 1 - 0.16 * kk, -120 * kk, 7 * kk
                alpha = 1 - smooth(ramp(t, self.D(1) - 0.1, self.D(1) + 0.2))
            else:
                alpha = 1 - smooth(ramp(t, self.fade_t, self.end_t))
            cx = self.CX
            xx = cx + (x0 - cx) * sc
            yy = self.CY + (base_y - self.CY) * sc + dy + rise
            cv.draw(sp, xx, yy, scale=sc, alpha=alpha, blur=blur, anchor=(0, sp.baseline / sp.h),
                    wipe=(cx + (edge - cx) * sc, 40 * sc))
            # the writing light runs along the baseline under the reveal edge
            ly = base_y + size * 0.16
            path = Path([(x0 - 30, ly), (x0 + tw + 30, ly)])
            head = ramp(t, t_start, t_end) * 1.0
            head = ease_in_out(head, 2.2) + 0.55 * ramp(t, t_end, t_end + 0.45)
            col = self.light_color(self.brief.get("write_light", 2))
            if alpha > 0:
                pts_t = lambda u: path  # noqa: E731
                cv.light_path(path if sc == 1 else Path([(cx + (x0 - 30 - cx) * sc, self.CY + (ly - self.CY) * sc + dy),
                                                         (cx + (x0 + tw + 30 - cx) * sc, self.CY + (ly - self.CY) * sc + dy)]),
                              head, tail=0.55, color=col, width=3.4, energy=0.95, fade=alpha)

    # ------------------------------------------------------------------
    # hook, peel, hand-overs, race
    # ------------------------------------------------------------------
    def facet_target(self, i, at_hit=False):
        f = self.logo.facets[i % len(self.logo.facets)]
        k = 0.86 if at_hit else 1.0
        return np.array([self.CX, self.CY]) + f["center"] * k

    def hook_paths(self):
        if "hook_paths" in self._sprites:
            return self._sprites["hook_paths"]
        W, H = self.DW, self.DH
        paths = []
        n = len(self.logo.facets)
        if n == 3 and not self.portrait:
            starts = [((-0.10 * W, 0.04 * H), (0.22 * W, 0.0), (0.42 * W, 0.16 * H)),
                      ((1.10 * W, 0.30 * H), (0.86 * W, 0.22 * H), (0.66 * W, 0.40 * H)),
                      ((0.04 * W, 1.10 * H), (0.14 * W, 0.78 * H), (0.33 * W, 0.64 * H))]
            # match each preset to the facet closest in direction
            order = sorted(range(3), key=lambda i: math.atan2(*self.logo.facets[i]["center"][::-1]))
            presets = sorted(range(3), key=lambda j: math.atan2(starts[j][2][1] - H / 2, starts[j][2][0] - W / 2))
            m = dict(zip(order, presets))
            for i in range(3):
                p0, p1, p2 = starts[m[i]]
                paths.append(Path.beziers([(p0, p1, p2, self.facet_target(i))]))
        else:
            for i in range(n):
                c = self.logo.facets[i]["center"]
                ang = math.atan2(c[1], c[0]) + 0.95
                R = math.hypot(W, H) * 0.62
                p0 = np.array([self.CX + math.cos(ang) * R, self.CY + math.sin(ang) * R])
                p3 = self.facet_target(i)
                d = p3 - p0
                L = np.linalg.norm(d)
                d /= L
                rot = lambda v, a: np.array([v[0] * math.cos(a) - v[1] * math.sin(a), v[0] * math.sin(a) + v[1] * math.cos(a)])  # noqa: E731
                p1 = p0 + rot(d, 0.55) * L * 0.35
                p2 = p3 - rot(d, -0.35) * L * 0.32
                paths.append(Path.beziers([(p0, p1, p2, p3)]))
        self._sprites["hook_paths"] = paths
        return paths

    def draw_hook(self, cv, t):
        if t > self.slam + 0.06:
            return
        if self.cold and self.scenes:
            x = clamp(t / self.slam)
            u = 0.18 + 0.82 * (0.35 * x + 0.65 * x * x)
            if t >= self.slam:
                u = 1.0 + (t - self.slam) * 8
            cols = [self.scenes[0].color] + [c for c in self.palette if c != self.scenes[0].color]
            for i, p in enumerate(self.cold_paths()):
                cv.light_path(p, u, tail=0.42, color=cols[i % len(cols)], width=4.5, energy=1.1)
            return
        if self.brief.get("hook", "converge") == "trace":
            # one light draws the logo's silhouette, then the fill slams in
            x = clamp(t / self.slam)
            u = ease_in_out(x, 1.7) if t < self.slam else 1.0 + (t - self.slam) * 3
            pts = self._contour_world(dict(cx=self.CX, cy=self.CY), 1.0)
            self._contour_light(cv, pts, u, 0.6, 1.15, 4.2, lap_tail=True)
            return
        x = clamp(t / self.slam)
        u = 0.18 + 0.82 * (0.35 * x + 0.65 * x * x)
        if t >= self.slam:
            u = 1.0 + (t - self.slam) * 8  # drain into the facet
        for i, p in enumerate(self.hook_paths()):
            cv.light_path(p, u, tail=0.42, color=self.logo.facets[i]["color"], width=4.5, energy=1.1)

    def cold_paths(self):
        if "cold" in self._sprites:
            return self._sprites["cold"]
        W, H = self.DW, self.DH
        tgt = np.array(self.scenes[0].entry(), float)
        if self.signature != "none":
            from .motifs import motif_points
            paths = [Path(motif_points(self.signature, (-0.04 * W, tgt[1]), tgt, seed=5))]
            self._sprites["cold"] = paths
            return paths
        starts = [((-0.10 * W, 0.04 * H), (0.22 * W, 0.0)), ((1.10 * W, 0.30 * H), (0.86 * W, 0.22 * H)),
                  ((0.04 * W, 1.10 * H), (0.14 * W, 0.78 * H))]
        paths = []
        for (p0, p1) in starts:
            p0, p1 = np.array(p0), np.array(p1)
            p2 = tgt + (p1 - tgt) * 0.35
            paths.append(Path.beziers([(p0, p1, p2, tgt)]))
        self._sprites["cold"] = paths
        return paths

    def peel_path(self):
        if "peel" in self._sprites:
            return self._sprites["peel"]
        sc = self.scenes[0] if self.scenes else None
        t_end = self.D(1)
        target = np.array(sc.entry()) if sc else np.array([self.CX, self.CY + 200])
        fi = self.scenes[0].facet if sc else 0
        xf = self._slam_xf(t_end - 0.38, self.slam, self.logo.design_h, 38, 0)
        k = xf["h"] / self.logo.design_h
        start = np.array([xf["cx"], xf["cy"]]) + self.logo.facets[fi % len(self.logo.facets)]["center"] * k
        mid = np.array([max(start[0], target[0]) + 260, (start[1] + target[1]) / 2 + 40])
        if self.signature != "none":
            from .motifs import motif_points
            out = start + np.array([0.0, 40.0])
            arc = Path.beziers([(start, start + (60, 30), out + (-40, 10), out)]).span(0, 1, 40)
            p = Path(np.vstack([arc, motif_points(self.signature, out, target, seed=99)[1:]]))
            self._sprites["peel"] = p
            return p
        p = Path.beziers([(start, start + (180, -60), mid + (40, -140), mid),
                          (mid, mid + (-40, 140), target + (220, 10), target)])
        self._sprites["peel"] = p
        return p

    def draw_peel(self, cv, t):
        t0, t1 = self.D(1) - 0.38, self.D(1)
        if self.cold or not (t0 <= t <= t1 + 0.12) or not self.scenes:
            return
        u = ease_in_out(ramp(t, t0, t1), 2.0) + max(0, t - t1) * 6
        cv.light_path(self.peel_path(), u, tail=0.36, color=self.scenes[0].color, width=4.5, energy=1.1)

    def draw_handovers(self, cv, t):
        for i in range(len(self.scenes) - 1):
            a, b = self.scenes[i], self.scenes[i + 1]
            if b.transition in ("whip", "cut", "wipe", "push"):
                continue
            t0 = b.a - 0.05
            t1 = b.land
            if not (t0 <= t <= t1 + 0.15):
                continue
            key = ("ho", i)
            if key not in self._sprites:
                p0, p1 = np.array(a.exit(), float), np.array(b.entry(), float)
                if self.signature != "none" and np.linalg.norm(p1 - p0) >= 300:
                    from .motifs import motif_points
                    self._sprites[key] = Path(motif_points(self.signature, p0, p1, seed=i))
                elif np.linalg.norm(p1 - p0) < 520:
                    # too short to read as a move: swoop out in an arc and come back in
                    lift = np.array([0.0, -330.0]) if p0[1] > self.DH * 0.3 else np.array([0.0, 330.0])
                    sx = 260.0 if p1[0] >= p0[0] else -260.0
                    self._sprites[key] = Path.beziers([(p0, p0 + lift + (-sx, 0), p1 + lift + (sx, 0), p1)])
                else:
                    side = 1.0 if p1[0] >= p0[0] else -1.0
                    self._sprites[key] = Path.through(p0, p1, bend=0.22, side=-side)
            u = ease_in_out(ramp(t, t0, t1), 2.0) + max(0.0, t - t1) * 5
            cv.light_path(self._sprites[key], u, tail=0.38, color=b.color, color_tail=a.color, width=4.5,
                          energy=1.15)

    def race_paths(self):
        if "race" in self._sprites:
            return self._sprites["race"]
        last = self.scenes[-1] if self.scenes else None
        n = len(self.logo.facets)
        starts = last.race_starts(n, self.race_t) if last else [
            (np.array([self.CX + 600 * math.cos(i * 2 * math.pi / n), self.CY + 300 * math.sin(i * 2 * math.pi / n)]),
             np.array([0.0, -1.0])) for i in range(n)]
        paths = []
        for i in range(n):
            p0, tan = starts[i]
            p3 = self.facet_target(i, at_hit=True)
            L = np.linalg.norm(p3 - p0)
            p1 = p0 + tan * L * 0.45
            p2 = p3 + (p0 - p3) * 0.25 + np.array([0, -L * 0.18])
            paths.append(Path.beziers([(p0, p1, p2, p3)]))
        self._sprites["race"] = paths
        return paths

    def race_colors(self):
        return [f["color"] for f in self.logo.facets]

    def draw_race(self, cv, t):
        if self.ending != "race" or not (self.race_t <= t <= self.hit + 0.06) or not self.scenes:
            return
        x = ramp(t, self.race_t, self.hit)
        u = ease_in(x, 2.0) * 0.97 + 0.03 * x
        if t >= self.hit:
            u = 1 + (t - self.hit) * 8
        for i, p in enumerate(self.race_paths()):
            cv.light_path(p, u, tail=0.4, color=self.race_colors()[i], width=4.5, energy=1.15)

    # ------------------------------------------------------------------
    # end card
    # ------------------------------------------------------------------
    def draw_end(self, cv, t):
        if t < self.fade_t:
            return
        ec = self.brief.get("end_card", {})
        y_line = self.CY + 44
        L = 300.0
        x0 = self.CX - L / 2
        # the rim head detaches and travels to the line start
        if t < self.end_t + 0.05:
            xf = self._slam_xf(self.fade_t, self.hit, self.logo.design_h * 0.86, 30, 1)
            k = xf["h"] / self.logo.design_h
            u = self.rim_head(self.fade_t)
            start = Path(self._contour_world(xf, k), closed=True).at(u)
            key = "dotpath"
            if key not in self._sprites:
                self._sprites[key] = Path.through(start, (x0, y_line), bend=0.18, side=-1)
            uu = ease_in_out(ramp(t, self.fade_t, self.end_t), 2.0)
            cv.light_path(self._sprites[key], uu, tail=0.16, color=self.light_color(2), width=4.0, energy=1.2)
        if t < self.end_t - 0.02:
            return
        fade = 1 - smooth(ramp(t, self.duration - 0.35, self.duration))
        lt = t - self.end_t
        # the line draws in thirds (one per light colour) with a moving glint
        prog = ease_out(ramp(lt, 0, 0.55), 2.5)
        cols = self.brief.get("end_line_colors") or self.palette[:3]
        n = len(cols)
        if self.signature not in ("none", "circuit", "rise"):
            from .motifs import motif_points
            key = "endmotif"
            if key not in self._sprites:
                self._sprites[key] = motif_points(self.signature, (x0, y_line), (x0 + L, y_line), amp=26, seed=3,
                                                  n=400, envelope=self.signature != "ecg")
            mp = self._sprites[key]
            m = max(2, int(len(mp) * prog))
            seg = len(mp) / n
            for i, c in enumerate(cols):
                a, b2 = int(i * seg), min(m, int((i + 1) * seg) + 1)
                if b2 - a >= 2:
                    cv.light_poly(mp[a:b2], lin(c), 0.85 * fade, width=4.0)
        else:
            for i, c in enumerate(cols):
                a0, a1 = i / n, (i + 1) / n
                if prog <= a0:
                    continue
                e = min(prog, a1)
                pts = [(x0 + L * a0 + (2 if i else 0), y_line), (x0 + L * e - (2 if i < n - 1 and e == a1 else 0), y_line)]
                cv.light_poly(pts, lin(c), 0.85 * fade, width=4.0)
        g = (lt * 0.55) % 1.4
        if g < 1.0 and prog > 0.99:
            gx = x0 + L * g
            cv.light_poly([(gx - 30, y_line), (gx, y_line)], lin("#FFFFFF") * 0 + lin(cols[min(n - 1, int(g * n))]),
                          [0.0, 0.9 * fade], width=4.0)
        rise = 14 * (1 - ease_out(ramp(lt, 0.15, 0.85)))
        a_txt = smooth(ramp(lt, 0.15, 0.6)) * fade
        style = ec.get("style", "credit")
        if style == "cta":
            top = ec.get("headline", self.brief.get("tagline", ""))
            big = ec.get("url", "")
        else:
            top = ec.get("top", "made by")
            big = ec.get("name", "your name")
        if top:
            sp = self.type.text(top, 30, 400, self.text2)
            cv.draw(sp, self.CX, y_line - 118 + rise, alpha=a_txt, anchor=(0.5, sp.baseline / sp.h))
        if big:
            sp = self.type.text(big, 74, 500, self.text_color)
            cv.draw(sp, self.CX, y_line - 30 + rise, alpha=a_txt, anchor=(0.5, sp.baseline / sp.h))
        small = ec.get("small")
        if small:
            sp = self.type.text(small, 20, 400, self.text2)
            cv.draw(sp, self.CX, self.DH - 70, alpha=smooth(ramp(lt, 0.4, 0.9)) * fade * 0.85,
                    anchor=(0.5, sp.baseline / sp.h))

    # ------------------------------------------------------------------
    # captions (optional, for muted autoplay)
    # ------------------------------------------------------------------
    def _load_voice(self):
        p = os.path.join(self.dir, "build", "voice.json")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        return None

    def draw_captions(self, cv, t):
        mode = self.brief.get("captions", "auto")
        if mode == "auto":
            mode = "on" if self.portrait else "off"
        if mode != "on" or not self.voice:
            return
        for ln in self.voice.get("lines", []):
            s, e = ln["start"], ln["end"]
            if not (s - 0.05 <= t <= e + 0.35):
                continue
            words = ln.get("words") or [{"w": ln["text"], "s": s, "e": e}]
            size = 44 if self.portrait else 36
            y = self.DH - (250 if self.portrait else 110)
            full = " ".join(w["w"] for w in words)
            tw = self.type.width(full, size, 500)
            x = self.CX - tw / 2
            out = 1 - smooth(ramp(t, e + 0.1, e + 0.35))
            for w in words:
                a = smooth(ramp(t, w["s"] - 0.06, w["s"] + 0.08)) * out
                sp = self.type.text(w["w"], size, 500, self.text_color)
                if a > 0:
                    cv.draw(sp, x, y + 8 * (1 - a), alpha=a, anchor=(0, sp.baseline / sp.h))
                x += self.type.width(w["w"] + " ", size, 500)

    # ------------------------------------------------------------------
    # stage tint, particles, shake
    # ------------------------------------------------------------------
    def tint(self, t):
        keys = [(0.0, self.primary), (self.D(1) - 0.4, mix_hex(self.primary, "#FFFFFF", 0.2))]
        for sc in self.scenes:
            keys.append((sc.a + 0.2, sc.tint_color()))
            keys.append((sc.b - 0.3, sc.tint_color()))
        keys.append((self.race_t + 0.4, self.primary))
        keys.sort()
        if t <= keys[0][0]:
            return keys[0][1]
        for (t0, c0), (t1, c1) in zip(keys, keys[1:]):
            if t <= t1:
                return mix_hex(c0, c1, smooth(ramp(t, t0, t1)))
        return keys[-1][1]

    def hits(self):
        hs = [self.slam, self.drop, self.hit]
        for sc in self.scenes:
            hs.append(sc.anchors().get("key", sc.a))
        return hs

    def _particle_speed(self, t):
        v = 0.12
        if t < self.slam:
            v = 2.6 + 3.0 * (t / self.slam)
        else:
            v += 2.2 * math.exp(-(t - self.slam) / 0.25)
        for sc in getattr(self, "scenes", []):
            if t >= sc.a:
                v += 0.45 * math.exp(-(t - sc.a) / 0.35)
        if hasattr(self, "hit") and t >= self.race_t:
            v += 0.9 * ramp(t, self.race_t, self.hit) * (1 if t < self.hit else math.exp(-(t - self.hit) / 0.3))
        return v

    def whip_offset(self, sc, t):
        """Whip pan: the outgoing scene flies left, the incoming one lands from the right."""
        off = 0.0
        nxt = sc.next
        if nxt is not None and nxt.transition == "whip" and t >= nxt.a - 0.16:
            off -= self.DW * 1.25 * (ease_in(ramp(t, nxt.a - 0.16, nxt.a), 2.2) if t < nxt.a else 1.0)
        if sc.transition == "whip" and sc.i > 0:
            off += self.DW * 1.25 * (1.0 - ease_out(ramp(t, sc.a, sc.a + 0.2), 2.6))
        return off

    WIPE = 0.45

    def wipe_x(self, sc, t):
        """x of the wiping line of light for an incoming 'wipe' scene (None outside the wipe)."""
        t0 = sc.a - 0.05
        if not (t0 <= t <= t0 + self.WIPE):
            return None
        return lerp(-40.0, self.DW + 40.0, ease_in_out(ramp(t, t0, t0 + self.WIPE), 2.2))

    def transition_state(self, sc, t):
        """(zoom, fade, region) for a scene at t, from wipe / push transitions; None = hidden."""
        zoom, fade, region = 1.0, 1.0, None
        nxt = sc.next
        if nxt is not None and nxt.transition == "wipe":
            t0 = nxt.a - 0.05
            if t >= t0 + self.WIPE:
                return None
            x = self.wipe_x(nxt, t)
            if x is not None:
                region = (x, self.DW + 100.0)
        if sc.transition == "wipe" and sc.i > 0:
            t0 = sc.a - 0.05
            if t < t0:
                return None
            x = self.wipe_x(sc, t)
            if x is not None:
                region = (-100.0, x)
        if nxt is not None and nxt.transition == "push" and t >= nxt.a - 0.3:
            k = ramp(t, nxt.a - 0.3, nxt.a)
            if k >= 1:
                return None
            zoom *= 1 + 0.6 * ease_in(k, 2.0)
            fade *= 1 - smooth(k)
        if sc.transition == "push" and sc.i > 0:
            if t < sc.a - 0.02:
                return None
            k = ease_out(ramp(t, sc.a - 0.02, sc.a + 0.35), 2.6)
            zoom *= 0.86 + 0.14 * k
            fade *= smooth(ramp(t, sc.a - 0.02, sc.a + 0.25))
        return zoom, fade, region

    def draw_wipes(self, cv, t):
        for sc in self.scenes:
            if sc.transition != "wipe" or sc.i == 0:
                continue
            x = self.wipe_x(sc, t)
            if x is None:
                continue
            k = ramp(t, sc.a - 0.05, sc.a - 0.05 + self.WIPE)
            e = math.sin(math.pi * k)
            cv.light_poly([(x, -20), (x, self.DH + 20)], lin(sc.color), 1.2 * e + 0.2, width=5.0)
            cv.light_poly([(x - 18, -20), (x - 18, self.DH + 20)], lin(sc.color), 0.25 * e, width=12.0)

    def push_blur(self, t):
        return sum(3.0 * math.exp(-((t - sc.a) / 0.09) ** 2) for sc in self.scenes if sc.transition == "push" and sc.i)

    def whip_blur(self, t):
        return sum(130 * math.exp(-((t - sc.a) / 0.07) ** 2) for sc in self.scenes if sc.transition == "whip" and sc.i)

    def shake(self, t):
        dx = dy = 0.0
        hit_amp = {"race": 9.0 if self.cold else 7.0, "cut": 9.0}.get(self.ending, 0.0)
        for t0, amp in (((self.slam, 11.0),) if not self.cold else ()) + ((self.hit, hit_amp),):
            if amp <= 0:
                continue
            lt = t - t0
            if 0 <= lt < 0.2:
                e = math.exp(-lt / 0.025)
                dx += amp * e * math.sin(2 * math.pi * 23 * lt)
                dy += amp * e * math.sin(2 * math.pi * 29 * lt + 1.1)
        return dx, dy

    # ------------------------------------------------------------------
    # one frame
    # ------------------------------------------------------------------
    def frame(self, t, index=None):
        index = int(round(t * self.fps)) if index is None else index
        cv = Canvas(self.W, self.H, self.S, self.stage_mode, self.slate)
        Canvas._stage_lin = lin(self.stage_color)
        pl = max([pulse(t, h, 0.03, 0.4) for h in self.hits()] + [0.0])
        stage_background(cv, t, self.stage_color, self.tint(t), float(self.brief.get("stage", {}).get("strength", 0.16)),
                         pl, dark=self.stage_mode == "dark")
        pf = 1.0 - 0.35 * smooth(ramp(t, self.end_t, self.end_t + 0.5))
        self.particles.draw(cv, t, fade=pf)
        self.draw_hook(cv, t)
        for sc in self.scenes:
            if sc.a - 0.6 <= t <= sc.b + 0.9 or (sc is self.scenes[-1] and t <= self.hit + 0.1 and t >= sc.a - 0.6):
                if sc.next is not None and sc.next.transition == "cut" and t >= sc.next.a:
                    continue
                if self.ending == "cut" and t >= self.hit:
                    continue
                if sc.transition == "cut" and t < sc.a:
                    continue
                cv.off = (self.whip_offset(sc, t), 0.0)
                st = self.transition_state(sc, t)
                if st is None:
                    cv.off = (0.0, 0.0)
                    continue
                cv.zoom, cv.fade, cv.region = st
                sc.draw(cv, t)
                cv.off, cv.zoom, cv.fade, cv.region = (0.0, 0.0), 1.0, 1.0, None
        self.draw_wipes(cv, t)
        self.draw_peel(cv, t)
        self.draw_handovers(cv, t)
        self.draw_race(cv, t)
        self.draw_logo(cv, t)
        self.draw_wordmark(cv, t)
        self.draw_end(cv, t)
        self.draw_captions(cv, t)
        img = cv.composite()
        pb = self.push_blur(t)
        if pb > 0.3:
            img = cv2.GaussianBlur(img, (0, 0), pb * self.S)
        wb = self.whip_blur(t)
        if wb > 0.6:
            k = max(3, int(wb * self.S) | 1)
            img = cv2.blur(img, (k, 1))
        return post(cv, img, t, index, shake=self.shake(t), breath=0.008, vignette=0.07, grain=0.005)

    # ------------------------------------------------------------------
    # sound cues (read by the mixer)
    # ------------------------------------------------------------------
    def cues(self):
        if self.cold:
            c = [dict(t=0.0, kind="air", dur=self.slam - 0.015, gain=-12),
                 dict(t=self.slam - 0.015, kind="riser_end", dur=self.slam - 0.015, gain=-15),
                 dict(t=self.slam, kind="whoosh_hit", gain=-12), dict(t=self.slam, kind="transient", gain=-14),
                 dict(t=self.slam + 0.01, kind="pop_low", gain=-17)]
        else:
            c = [
                dict(t=0.0, kind="air", dur=self.slam - 0.015, gain=-10),
                dict(t=self.slam - 0.015, kind="riser_end", dur=self.slam - 0.015, gain=-12),
                dict(t=self.slam, kind="impact", gain=-3),
                dict(t=self.slam + 0.02, kind="ring", gain=-20, dur=2.5),
                dict(t=self.slam + 0.77, kind="whoosh_soft", gain=-17, pan=-0.3),
                dict(t=self.slam + 1.52, kind="confirm", gain=-19),
                dict(t=self.D(1) - 0.02, kind="whoosh_hit", gain=-13, pan=0.3),
            ]
        for i, sc in enumerate(self.scenes):
            c += sc.cues()
            if not i:
                continue
            if sc.transition == "whip":
                c += [dict(t=sc.a, kind="whoosh_fast", gain=-11), dict(t=sc.a + 0.05, kind="transient", gain=-16)]
            elif sc.transition == "wipe":
                c += [dict(t=sc.a - 0.05, kind="swoosh", gain=-14, dur=0.55), dict(t=sc.a + 0.15, kind="shimmer", gain=-22)]
            elif sc.transition == "push":
                c += [dict(t=sc.a - 0.05, kind="whoosh_hit", gain=-12), dict(t=sc.a, kind="pop_low", gain=-17)]
            elif sc.transition == "cut":
                c.append(dict(t=sc.a, kind="transient", gain=-14))
            else:
                c.append(dict(t=sc.land - 0.08, kind="whoosh_hit", gain=-15, pan=0.2 * (-1) ** i))
        c += [
        ]
        if self.ending == "race":
            c += [dict(t=self.race_t + 0.5, kind="breath", gain=-16, dur=self.hit - self.race_t - 0.5),
                  dict(t=self.race_t + 0.92, kind="whoosh_fast", gain=-13),
                  dict(t=self.hit, kind="impact", gain=-3),
                  dict(t=self.hit + 0.02, kind="ring", gain=-20, dur=3.0),
                  dict(t=self.hit + 1.2, kind="confirm", gain=-20)]
        elif self.ending == "cut":
            c += [dict(t=self.hit - 0.6, kind="breath", gain=-18, dur=0.55),
                  dict(t=self.hit, kind="impact", gain=-2), dict(t=self.hit, kind="transient", gain=-8),
                  dict(t=self.hit + 1.0, kind="confirm", gain=-20)]
        elif self.ending == "reveal":
            c += [dict(t=self.hit - 0.05, kind="pop_low", gain=-15), dict(t=self.hit, kind="shimmer", gain=-17),
                  dict(t=self.hit + 0.02, kind="ring", gain=-22, dur=3.0),
                  dict(t=self.hit + 1.15, kind="whoosh_soft", gain=-21)]
        else:  # write
            c += [dict(t=self.hit + 0.05, kind="whoosh_soft", gain=-18), dict(t=self.hit + 0.8, kind="pop_low", gain=-17),
                  dict(t=self.hit + 0.85, kind="shimmer", gain=-20)]
        c.append(dict(t=self.end_t + 0.05, kind="sparkle", gain=-22))
        c += self._motif_cues()
        return sorted(c, key=lambda d: d["t"])

    def _motif_cues(self):
        if self.signature in ("none", "rise"):
            return []
        from .motifs import beats
        kind = {"ecg": "beep", "circuit": "tick", "route": "pop_soft", "wave": "shimmer", "ripple": "pop_soft",
                "spark": "sparkle"}.get(self.signature, "tick")
        out = []
        for i in range(len(self.scenes) - 1):
            a, b = self.scenes[i], self.scenes[i + 1]
            if b.transition in ("whip", "cut", "wipe", "push"):
                continue
            p0, p1 = np.array(a.exit(), float), np.array(b.entry(), float)
            if np.linalg.norm(p1 - p0) < 300:
                continue
            t0, t1 = b.a - 0.05, b.land
            ts = np.linspace(t0, t1, 400)
            us = np.array([ease_in_out(ramp(t, t0, t1), 2.0) for t in ts])
            last = -9.0
            for f in beats(self.signature, p0, p1, seed=i):
                j = int(np.searchsorted(us, f))
                if j < len(ts) and ts[j] - last >= 0.3:   # never machine-gun the beeps
                    out.append(dict(t=float(ts[j]), kind=kind, gain=-21, pitch=1.0))
                    last = ts[j]
        return out

    def music_plan(self):
        return dict(bpm=self.bpm, d0=self.d0, slam=self.slam, drop=self.drop, breath=self.hit - 0.6,
                    hit=self.hit, end=self.duration, name=self.name,
                    soft_hit=self.ending in ("reveal", "write"))


def lin_from_srgb(c):
    from .core import srgb_to_linear
    return srgb_to_linear(np.asarray(c, np.float32))
