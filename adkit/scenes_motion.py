"""Motion-graphics scenes: the premium SaaS explainer vocabulary.

    type      kinetic typography: build (words land on beats around an inline icon), swap (a sentence
              rewrites itself), giant (one huge gradient word), cloud (keywords at depth around a
              focus word), tilt (3D receding type), teaser (blur-in, letters slide out)
    morph     one container that keeps becoming the next thing: pill -> glowing bar -> card
    orbit     items float in 3D around the product's icon, then converge into it
    tunnel    cards on a rotating 3D ring around a headline
    success   a big check, a burst of 3D confetti, one word
    montage   images cut on every beat with a push-in and a cinematic grade
    grid      integration tiles laid on a tilted 3D plane, popping in on the beat

Every timed event lands on the music's beat grid (film.grid), so these scenes cut with the song.
"""
from __future__ import annotations

import math
import os

import numpy as np
from PIL import Image

from . import ui
from .core import (Path, ease_in, ease_in_out, ease_out, lerp, lin, mix_hex, pulse, ramp, rng, smooth,
                   spring_in)
from .scenes import Scene, register, rrect_points


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------

def beats_in(sc, t0, t1, every=1):
    bs = sc.f.grid.beats_between(t0, t1)
    return bs[::max(1, every)]


def _norm(w):
    return "".join(c for c in w.lower() if c.isalnum())


def voice_times(sc, words, t0, t1):
    """When the voice-over says these words (in order) inside [t0, t1], their start times; else None.

    Kinetic type that matches the narration lands each word exactly as it is spoken."""
    v = sc.f.voice
    if not v or not words:
        return None
    spoken = [(_norm(w["w"]), w["s"]) for ln in v.get("lines", []) for w in ln.get("words", [])
              if t0 - 0.4 <= w["s"] <= t1]
    out, j = [], 0
    for w in words:
        n = _norm(w)
        while j < len(spoken) and spoken[j][0] != n:
            j += 1
        if j == len(spoken):
            return None
        out.append(spoken[j][1])
        j += 1
    return out


def star_points(cx, cy, r, rot=0.0, k=0.38, n=4):
    """A soft 4-point sparkle (the 'AI' star)."""
    pts = []
    for i in range(n * 2 * 6):
        a = rot + i * math.pi / (n * 6)
        lobe = abs(math.cos(n * (a - rot) / 2))
        rr = r * (k + (1 - k) * lobe ** 3)
        pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
    return pts


def gradient_text(sc, text, size, weight, c1, c2, typ=None):
    """A text sprite filled with a left-to-right gradient."""
    typ = typ or sc.f.type
    key = ("grad", text, size, weight, c1, c2, id(typ))
    if key in sc.f._sprites:
        return sc.f._sprites[key]
    base = typ.text(text, size, weight, "#FFFFFF")
    img = base.img.copy()
    h, w = img.shape[:2]
    g = np.linspace(0, 1, w, dtype=np.float32)[None, :, None]
    col = lin(c1)[None, None, :] * (1 - g) + lin(c2)[None, None, :] * g
    img[..., :3] = col * img[..., 3:4]
    from .gfx import Sprite
    sp = Sprite(img, base.ppd, base.baseline, dict(base.meta))
    sc.f._sprites[key] = sp
    return sp


def icon_sprite(sc, kind, size, color):
    """An inline icon for kinetic type: the brand logo, a sparkle, or a dot."""
    if kind == "logo":
        return None
    key = ("icon", kind, size, color)
    if key in sc.f._sprites:
        return sc.f._sprites[key]
    p = ui.Painter(size, size, sc.ppd * 1.5, sc.fonts)
    if kind == "star":
        p.poly([(x, y) for x, y in star_points(size / 2, size / 2, size * 0.48, -math.pi / 2)], color)
    else:
        p.circle(size / 2, size / 2, size * 0.32, color)
    sp = p.sprite()
    sc.f._sprites[key] = sp
    return sp


def confetti(cv, origin, t_since, seed, colors, n=46, power=1.0, area=None):
    """3D confetti: thin cards bursting out, tumbling (foreshortened) and falling."""
    if t_since < 0 or t_since > 2.2:
        return
    r = rng(seed)
    for i in range(n):
        ang = r.uniform(-math.pi, 0) if r.random() < 0.8 else r.uniform(0, math.pi)
        sp = r.uniform(500, 1500) * power
        vx, vy = math.cos(ang) * sp, math.sin(ang) * sp - r.uniform(200, 600)
        drag = 2.4
        t = t_since
        x = origin[0] + vx * (1 - math.exp(-drag * t)) / drag
        y = origin[1] + vy * (1 - math.exp(-drag * t)) / drag + 420 * t * t
        w, h = r.uniform(14, 30), r.uniform(7, 14)
        rot = r.uniform(0, 6.3) + t * r.uniform(-8, 8)
        tumble = abs(math.cos(r.uniform(0, 6.3) + t * r.uniform(4, 12)))
        a = 1 - smooth(ramp(t, 1.3, 2.2))
        cs, sn = math.cos(rot), math.sin(rot)
        hw, hh = w / 2, h / 2 * (0.15 + 0.85 * tumble)
        pts = [(x + cs * dx - sn * dy, y + sn * dx + cs * dy) for dx, dy in ((-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh))]
        cv.fill_poly(pts, colors[i % len(colors)], alpha=a)


def _bg_ok(sc):
    return sc.f.stage_mode == "dark"


# ===========================================================================
# type: kinetic typography
# ===========================================================================

@register("type")
class Type(Scene):
    """Kinetic typography in six styles. Words land on the beat.

    data: {"style": "build", "text": "The future of your business is here", "icon": "star"|"logo"|null,
           "icon_at": 5, "highlight": ["future"], "size": 96}
          {"style": "swap", "phrases": ["Too much", "Too many questions", "Too little time"]}
          {"style": "giant", "word": "new", "sub": "features and updates"}
          {"style": "cloud", "focus": ["Feedback", "Updates", "Roadmap"], "words": ["Voting board", ...]}
          {"style": "tilt", "text": "with ease"}
          {"style": "teaser", "text": "COMING SOON"}
    Any style: "font": "display" uses the brand's display face instead of the body face.
    """
    weight = 1.2

    def setup(self):
        d, f = self.data, self.f
        self.style = d.get("style", "build")
        self.typ = f.display_type() if d.get("font") == "display" else f.type
        self.size = float(d.get("size", {"giant": 300, "teaser": 120, "tilt": 150, "cloud": 92}.get(self.style, 92)))
        if f.portrait:
            self.size *= 0.78
        self.hl = [h.lower().strip(".,!?") for h in d.get("highlight", [])]
        self.t0 = self.t_in + 0.08
        bs = beats_in(self, self.t0, self.b - 0.3)
        self.beats = bs or list(np.linspace(self.t0, self.b - 0.4, 6))
        if self.style == "build":
            words = (d.get("text") or f.brief.get("tagline", f.name)).split(" ")
            toks = [dict(w=w) for w in words]
            if d.get("icon"):
                at = int(d.get("icon_at", len(words) // 2))
                toks.insert(min(at, len(toks)), dict(icon=d["icon"]))
            # the icon (if any) lands first, then the words around it, one per beat (or half-beat)
            order = sorted(range(len(toks)), key=lambda i: (0 if "icon" in toks[i] else 1, i))
            per = max(1, len(self.beats) // max(1, len(toks)))
            half = len(self.beats) < len(toks)
            grid = self.beats if not half else sorted(self.beats + [(a + b) / 2 for a, b in zip(self.beats, self.beats[1:])])
            for k, i in enumerate(order):
                toks[i]["t"] = grid[min(len(grid) - 1, k * (1 if half else per))]
            vt = voice_times(self, words, self.a, self.b)
            if vt:  # the narration says this line: each word lands as it is spoken, the icon first
                for tk, tv in zip([tk for tk in toks if "w" in tk], vt):
                    tk["t"] = max(self.t0, tv)
                for tk in toks:
                    if "icon" in tk:
                        tk["t"] = self.t0
            self.toks = toks
            self.key = max(tk["t"] for tk in toks)
        elif self.style == "swap":
            self.phrases = d.get("phrases") or ["Too much", "Too many questions", "Too little time"]
            step = max(1, len(self.beats) // len(self.phrases))
            self.t_ph = [self.t0] + [self.beats[min(len(self.beats) - 1, i * step)] for i in range(1, len(self.phrases))]
            # with matching narration, each phrase changes on its first new spoken word
            firsts = []
            for i, ph in enumerate(self.phrases):
                prev = self.phrases[i - 1].split(" ") if i else []
                cur = ph.split(" ")
                new = [w for j, w in enumerate(cur) if j >= len(prev) or prev[j] != w]
                firsts.append(new[0] if new else cur[0])
            vt = voice_times(self, firsts, self.a, self.b)
            if vt:
                self.t_ph = [max(self.t0, x) for x in vt]
            self.key = self.t_ph[-1]
        elif self.style == "cloud":
            self.focus = d.get("focus") or ["Feedback", "Updates", "Roadmap"]
            self.words = d.get("words") or ["Voting board", "All-in-one widget", "Roadmap", "Multiple sources",
                                            "Pop-up", "Increase loyalty", "Changelog", "Integrations"]
            step = max(1, len(self.beats) // len(self.focus))
            self.t_f = [self.beats[min(len(self.beats) - 1, i * step)] for i in range(len(self.focus))]
            r = rng(self.i + 5)
            self.place = [(r.uniform(0.06, 0.94), r.uniform(0.1, 0.9), r.uniform(0.2, 1.0), r.uniform(0, 6.3))
                          for _ in self.words]
            self.key = self.t_f[-1]
        else:
            self.key = self.beats[min(len(self.beats) - 1, 2 if self.L > 2.5 else 1)]
        self.disp = f.display_type() if self.style in ("teaser", "giant") else f.type
        # "font": "display" sets every style in the brand's display face (a serif, a condensed...)
        self.typ = f.display_type() if d.get("font") == "display" else f.type
        if d.get("font") == "display":
            self.disp = self.typ

    def entry(self):
        return (self.f.CX - 400, self.f.CY + self.size * 0.5)

    def exit(self):
        return (self.f.CX + 400, self.f.CY + self.size * 0.5)

    def anchors(self):
        return {"key": self.key}

    def cues(self):
        c = []
        if self.style == "build":
            c = [dict(t=tk["t"], kind="tick" if "icon" not in tk else "pop", gain=-22, pitch=0.9 + 0.04 * i)
                 for i, tk in enumerate(self.toks)]
        elif self.style == "swap":
            c = [dict(t=t, kind="whoosh_short", gain=-21) for t in self.t_ph]
        elif self.style == "cloud":
            c = [dict(t=t, kind="pop_soft", gain=-20) for t in self.t_f]
        elif self.style == "giant":
            c = [dict(t=self.t0, kind="whoosh_hit", gain=-14), dict(t=self.t0 + 0.05, kind="pop_low", gain=-15)]
        elif self.style == "teaser":
            c = [dict(t=self.t0, kind="riser_short", gain=-18, dur=0.9), dict(t=self.t0 + 0.9, kind="impact", gain=-9),
                 dict(t=self.b - 0.5, kind="whoosh_fast", gain=-15)]
        else:
            c = [dict(t=self.t0, kind="swoosh", gain=-18, dur=0.8)]
        return c

    def _col(self, w):
        return self.color if w.lower().strip(".,!?") in self.hl else self.f.text_color

    def draw(self, cv, t):
        out = 1 - smooth(ramp(t, self.b - 0.15, self.b + 0.2))
        if out <= 0 or t < self.t_in - 0.05:
            return
        getattr(self, "_" + self.style, self._build)(cv, t, out)

    # -- build ---------------------------------------------------------------
    def _tok_w(self, tk):
        if "icon" in tk:
            return self.size * 0.95
        return self.typ.width(tk["w"], self.size, 600)

    def _layout(self, vis):
        sp = self.typ.width(" ", self.size, 600)
        ws = [self._tok_w(self.toks[i]) for i in vis]
        tot = sum(ws) + sp * (len(ws) - 1)
        x = self.f.CX - tot / 2
        pos = {}
        for i, w in zip(vis, ws):
            pos[i] = x
            x += w + sp
        return pos

    def _build(self, cv, t, out):
        f = self.f
        order = sorted(range(len(self.toks)), key=lambda i: self.toks[i]["t"])
        vis = [i for i in order if self.toks[i]["t"] <= t]
        if not vis:
            return
        last_t = max(self.toks[i]["t"] for i in vis)
        prev = [i for i in vis if self.toks[i]["t"] < last_t]
        k = ease_out(ramp(t, last_t, last_t + 0.3), 3.0)
        p1 = self._layout(sorted(vis))
        p0 = self._layout(sorted(prev)) if prev else p1
        y = f.CY + self.size * 0.34
        for i in sorted(vis):
            tk = self.toks[i]
            x = lerp(p0.get(i, p1[i]), p1[i], k)
            a = ease_out(ramp(t, tk["t"], tk["t"] + 0.28), 2.6) * out
            rise = (1 - a) * 22
            if "icon" in tk:
                s = self.size * 0.95
                pop = spring_in(t - tk["t"], 4.0, 0.45)
                if tk["icon"] == "logo":
                    lg = f.logo.sprite
                    cv.draw(lg, x + s / 2, f.CY, scale=max(0.01, pop) * s * 0.9 / lg.h, rot=8 * math.sin(t * 1.3), alpha=out)
                else:
                    sp = icon_sprite(self, tk["icon"], 96, self.color)
                    cv.draw(sp, x + s / 2, f.CY, scale=max(0.01, pop) * s / sp.w * 0.9,
                            rot=(t - tk["t"]) * 40, alpha=out)
                continue
            sp = self.typ.text(tk["w"], self.size, 600, self._col(tk["w"]))
            cv.draw(sp, x, y + rise, alpha=a, blur=5 * (1 - a), anchor=(0, sp.baseline / sp.h))

    # -- swap ----------------------------------------------------------------
    def _swap(self, cv, t, out):
        f = self.f
        idx = max([i for i, tp in enumerate(self.t_ph) if t >= tp] or [0])
        k = ease_out(ramp(t, self.t_ph[idx], self.t_ph[idx] + 0.35), 2.8)
        cur = self.phrases[idx].split(" ")
        prv = self.phrases[idx - 1].split(" ") if idx > 0 else []

        def lay(words):
            sp = self.typ.width(" ", self.size, 500)
            ws = [self.typ.width(w, self.size, 500) for w in words]
            x = f.CX - (sum(ws) + sp * (len(ws) - 1)) / 2
            xs = []
            for w in ws:
                xs.append(x)
                x += w + sp
            return xs

        y = f.CY + self.size * 0.34
        x1 = lay(cur)
        x0 = lay(prv) if prv else x1
        shift = (1 - k) * (x0[0] - x1[0]) if prv else 0.0   # new words ride along as the line re-centres
        for j, w in enumerate(cur):
            same = j < len(prv) and prv[j] == w
            xs = lerp(x0[j], x1[j], k) if same else x1[j] + shift
            a = out if same else ramp(k, 0.35, 1.0) * out   # new words wait for the old ones to clear
            dy = 0 if same else (1 - k) * 40
            sp = self.typ.text(w, self.size, 500, self._col(w))
            cv.draw(sp, xs, y + dy, alpha=a, blur=0 if same else 6 * (1 - k), anchor=(0, sp.baseline / sp.h))
        for j, w in enumerate(prv):
            if j < len(cur) and cur[j] == w:
                continue
            sp = self.typ.text(w, self.size, 500, self.f.text2)
            g = ramp(k, 0.0, 0.45)
            cv.draw(sp, x0[j], y - g * 40, alpha=(1 - g) * out, blur=6 * g, anchor=(0, sp.baseline / sp.h))

    # -- giant ---------------------------------------------------------------
    def _giant(self, cv, t, out):
        f = self.f
        word = self.data.get("word", "new")
        c1 = self.data.get("from", mix_hex(self.color, "#FFFFFF", 0.15))
        c2 = self.data.get("to", f.palette[1 % len(f.palette)])
        sp = gradient_text(self, word, self.size, 700, c1, c2, self.disp)
        pop = spring_in(t - self.t0, 3.2, 0.55)
        shrink = ease_in_out(ramp(t, self.key, self.key + 0.5), 2.0)
        sc = max(0.01, pop) * (1 - 0.45 * shrink)
        cy = f.CY - 60 * shrink
        cv.draw(sp, f.CX, cy + self.size * 0.3 * sc, scale=sc, alpha=out, anchor=(0.5, sp.baseline / sp.h))
        sub = self.data.get("sub")
        if sub:
            a = smooth(ramp(t, self.key + 0.2, self.key + 0.6)) * out
            ss = self.typ.text(sub, 54, 500, f.text_color)
            cv.draw(ss, f.CX, cy + self.size * 0.42, alpha=a, anchor=(0.5, ss.baseline / ss.h))

    # -- cloud ---------------------------------------------------------------
    def _cloud(self, cv, t, out):
        f = self.f
        for (u, v, z, ph), w in zip(self.place, self.words):
            x = u * f.DW + 40 * math.sin(t * 0.4 * (1 + z) + ph)
            y = v * f.DH + 18 * math.cos(t * 0.5 + ph)
            size = 30 + 34 * z
            a = smooth(ramp(t, self.t0 + 0.1 * z, self.t0 + 0.6)) * out * (0.35 + 0.55 * z)
            sp = self.typ.text(w, size, 600, self.color)
            cv.draw(sp, x, y, alpha=a, blur=(1 - z) * 6, anchor=(0.5, 0.5))
        idx = max([i for i, tf in enumerate(self.t_f) if t >= tf] or [0])
        k = ease_out(ramp(t, self.t_f[idx], self.t_f[idx] + 0.3), 2.8)
        if idx > 0:
            sp0 = self.typ.text(self.focus[idx - 1], self.size, 700, f.text_color)
            cv.draw(sp0, f.CX, f.CY - 30 * k, alpha=(1 - k) * out, blur=6 * k, anchor=(0.5, 0.5))
        sp = self.typ.text(self.focus[idx], self.size, 700, f.text_color)
        cv.draw(sp, f.CX, f.CY + 30 * (1 - k), alpha=k * out, blur=6 * (1 - k), anchor=(0.5, 0.5))

    # -- tilt (3D receding type) ---------------------------------------------
    def _tilt(self, cv, t, out):
        f = self.f
        text = self.data.get("text", "with ease")
        sp = self.typ.text(text, self.size, 700, self._col(text) if not self.hl else self.color)
        k = ease_out(ramp(t, self.t0, self.t0 + 0.9), 2.4)
        w, h = sp.w, sp.h
        tilt = math.radians(lerp(58, 24, k))
        foc = 900.0
        cx, cy = f.CX + lerp(260, 0, k), f.CY
        pts = []
        for x, y in ((-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)):
            Y = y * math.cos(tilt)
            Z = y * math.sin(tilt) + 120
            q = foc / (foc + Z)
            pts.append((cx + x * q, cy + Y * q))
        cv.draw_quad(sp, pts, alpha=out * smooth(ramp(t, self.t0 - 0.05, self.t0 + 0.3)))

    # -- teaser (blur-in with tracking, letters slide out) ---------------------
    def _teaser(self, cv, t, out):
        f = self.f
        text = self.data.get("text", "COMING SOON")
        typ = self.disp
        k = ease_out(ramp(t, self.t0, self.t0 + 0.9), 2.6)
        trk = lerp(0.6, 0.06, k)
        widths = [typ.width(ch, self.size, 700) for ch in text]
        gap = trk * self.size
        tot = sum(widths) + gap * (len(text) - 1)
        x = f.CX - tot / 2
        t_out = self.b - 0.55
        for i, (ch, w) in enumerate(zip(text, widths)):
            if ch == " ":
                x += w + gap
                continue
            sp = typ.text(ch, self.size, 700, f.text_color)
            # exit: letters slide sideways from the edges in, smearing
            e = ease_in(ramp(t, t_out + 0.025 * abs(i - len(text) / 2), t_out + 0.35), 2.2)
            dx = (1 if i >= len(text) / 2 else -1) * 900 * e
            cv.draw(sp, x + dx, f.CY + self.size * 0.35, alpha=k * (1 - e) * out, blur=12 * (1 - k) + 10 * e,
                    anchor=(0, sp.baseline / sp.h))
            x += w + gap


# ===========================================================================
# morph: one container that keeps becoming the next thing
# ===========================================================================

@register("morph")
class Morph(Scene):
    """data: {"steps": [{"w": 300, "h": 84, "r": 42, "fill": "#0F1A16", "label": "Ask AI", "icon": "star"},
                        {"w": 1000, "h": 120, "r": 30, "glow": true},
                        {"w": 1040, "h": 520, "r": 34, "label": "How do I add Doks to Next.js?",
                         "lines": ["Install the SDK", "Wrap your app", "Ship it"]}],
              "cursor": true}"""
    weight = 1.6

    def setup(self):
        d, f = self.data, self.f
        T = f.theme
        self.steps = d.get("steps") or [
            dict(w=300, h=84, r=42, fill=T.card, label="Ask AI", icon="star"),
            dict(w=1000, h=120, r=30, glow=True),
            dict(w=1040, h=520, r=34, label="How do I get started?", lines=["Connect your data", "Ask anything",
                                                                             "Ship the answer"])]
        bs = beats_in(self, self.land + 0.6, self.b - 0.4)
        n = len(self.steps)
        self.cursor = bool(d.get("cursor", True))
        self.t_click = self.land + 0.45
        step = max(1, len(bs) // n) if bs else 1
        self.t_step = [self.t_click + 0.15] + ([bs[min(len(bs) - 1, i * step)] for i in range(1, n)] if bs else
                                              list(np.linspace(self.t_click + 0.9, self.b - 0.6, n - 1)))
        self.key = self.t_step[-1]

    def shape(self, t):
        """Interpolated w, h, r, fill and step index (springy)."""
        S = self.steps
        i = max([j for j, ts in enumerate(self.t_step) if t >= ts] or [0])
        if i == 0 or t < self.t_step[0]:
            s = S[0]
            return s["w"], s["h"], s["r"], s.get("fill", self.f.theme.card), 0, 1.0
        k = min(1.2, spring_in(t - self.t_step[i], 3.0, 0.6))
        a, b = S[i - 1], S[i]
        w = lerp(a["w"], b["w"], k)
        h = lerp(a["h"], b["h"], k)
        r = lerp(a["r"], b["r"], min(1, k))
        return w, h, r, b.get("fill", a.get("fill", self.f.theme.card)), i, ramp(t, self.t_step[i] + 0.15, self.t_step[i] + 0.45)

    def entry(self):
        return (self.f.CX - 150, self.f.CY)

    def exit(self):
        return (self.f.CX + 500, self.f.CY + 200)

    def anchors(self):
        a = {"key": self.key, "click": self.t_click}
        for i, ts in enumerate(self.t_step):
            a[f"s{i + 1}"] = ts
        return a

    def cues(self):
        c = [dict(t=self.t_click, kind="click", gain=-15)]
        c += [dict(t=ts, kind="whoosh_hit" if i else "pop", gain=-17) for i, ts in enumerate(self.t_step)]
        c.append(dict(t=self.key + 0.5, kind="shimmer", gain=-21))
        return c

    def draw(self, cv, t):
        f, T = self.f, self.f.theme
        out = 1 - smooth(ramp(t, self.b - 0.15, self.b + 0.2))
        a_in = ease_out(ramp(t, self.t_in, self.t_in + 0.4), 2.6)
        al = a_in * out
        if al <= 0:
            return
        w, h, r, fill, i, lab_k = self.shape(t)
        cx, cy = f.CX, f.CY
        x0, y0 = cx - w / 2, cy - h / 2
        st = self.steps[i]
        pts = rrect_points(x0, y0, w, h, min(r, h / 2), 220)
        # glow: a soft halo + a lit outline (light) in the brand colour
        if st.get("glow") or i == 0:
            halo = rrect_points(x0 - 30, y0 - 30, w + 60, h + 60, min(r + 30, (h + 60) / 2), 120)
            cv.fill_poly(halo, self.color, alpha=0.18 * al, blur=26)
        cv.fill_poly(pts, fill, alpha=al)
        if st.get("glow"):
            cv.fill_poly(pts, self.color, alpha=0.75 * al * (0.75 + 0.25 * math.sin(t * 6)))
        u = ease_out(ramp(t, self.t_step[min(i, len(self.t_step) - 1)], self.t_step[min(i, len(self.t_step) - 1)] + 0.6), 2.0)
        cv.light_path(Path(pts), u, tail=0.6, color=self.color, width=3.0, energy=0.9 * out)
        # content of the current step
        if st.get("label"):
            a = lab_k * al if i else al
            if st.get("lines"):
                sp = f.type.text(st["label"], 34, 600, T.text)
                cv.draw(sp, x0 + 48, y0 + 72, alpha=a, anchor=(0, sp.baseline / sp.h))
                for j, ln in enumerate(st["lines"]):
                    tj = self.t_step[i] + 0.4 + 0.25 * j
                    aj = ease_out(ramp(t, tj, tj + 0.3)) * al
                    ls = f.type.text(ln, 30, 400, T.text2)
                    cv.draw(ls, x0 + 48, y0 + 140 + j * 54, alpha=aj, anchor=(0, ls.baseline / ls.h))
            else:
                sp = f.type.text(st["label"], 30, 600, T.text)
                ix = 0
                if st.get("icon"):
                    ic = icon_sprite(self, st["icon"], 64, self.color)
                    cv.draw(ic, cx - sp.w / 2 - 6, cy, scale=0.5, rot=t * 30, alpha=a)
                    ix = 22
                cv.draw(sp, cx - sp.w / 2 + ix, cy + 11, alpha=a, anchor=(0, sp.baseline / sp.h))
        # cursor arrives and clicks the pill
        if self.cursor and t < self.t_step[0] + 0.6:
            cur = self.sp("cursor", lambda: ui.cursor("#FFFFFF" if _bg_ok(self) else "#1F1F1F", self.ppd, self.fonts))
            k = ease_out(ramp(t, self.t_in, self.t_click), 2.6)
            px, py = lerp(cx + 420, cx + 40, k), lerp(cy + 260, cy + 12, k)
            press = 1 - 0.15 * pulse(t, self.t_click, 0.02, 0.12)
            cv.draw(cur, px, py, scale=press, anchor=(0.07, 0.05), alpha=out * (1 - smooth(ramp(t, self.t_step[0] + 0.2, self.t_step[0] + 0.6))))
            rr = ease_out(ramp(t, self.t_click, self.t_click + 0.45))
            if 0 < rr < 1:
                ring = [(cx + 40 + 60 * rr * math.cos(q), cy + 12 + 60 * rr * math.sin(q)) for q in np.linspace(0, 6.3, 48)]
                cv.light_poly(ring, lin(self.color), (1 - rr) * out, width=3)


# ===========================================================================
# orbit: things float around the product, then converge into it
# ===========================================================================

@register("orbit")
class Orbit(Scene):
    """data: {"items": [{"label": "Slack", "color": "#E01E5A"}, ...] (4-8), "center": "logo" | "text",
              "text": "into one place", "converge": true}"""
    weight = 1.5

    def setup(self):
        d, f = self.data, self.f
        self.items = d.get("items") or [dict(label=n, color=c) for n, c in (("Email", "#EA4335"), ("Chat", "#5865F2"),
                                                                         ("Docs", "#4285F4"), ("Calls", "#0EBC5F"),
                                                                         ("Tickets", "#F4B400"), ("CRM", "#A259FF"))]
        n = len(self.items)
        r = rng(self.i + 9)
        self.ph = [2 * math.pi * k / n + r.uniform(-0.2, 0.2) for k in range(n)]
        self.rad = [r.uniform(0.85, 1.1) for _ in range(n)]
        self.converge = bool(d.get("converge", True))
        self.t_conv = f.snap(self.a + 0.62 * self.L) if self.converge else self.b + 9
        self.key = self.t_conv + 0.55

    def pos(self, k, t):
        f = self.f
        ang = self.ph[k] + (t - self.a) * 0.45
        R = 560 * self.rad[k] * (1 if not f.portrait else 0.72)
        c = ease_in(ramp(t, self.t_conv + 0.02 * k, self.t_conv + 0.5 + 0.02 * k), 2.2)
        R *= (1 - c)
        x = f.CX + R * math.cos(ang)
        z = math.sin(ang)            # -1 back .. 1 front
        y = f.CY + R * 0.34 * z + 40 * math.sin(t * 0.8 + k)
        s = (0.78 + 0.28 * (z + 1) / 2) * (1 - 0.6 * c)
        return x, y, z, s, c

    def entry(self):
        return (self.f.CX - 560, self.f.CY)

    def exit(self):
        return (self.f.CX, self.f.CY)

    def anchors(self):
        return {"key": self.key, "converge": self.t_conv}

    def cues(self):
        c = [dict(t=self.land + 0.1 * k, kind="pop_soft", gain=-24, pitch=0.9 + 0.06 * k) for k in range(len(self.items))]
        if self.converge:
            c += [dict(t=self.t_conv, kind="swirl", gain=-17, dur=0.6), dict(t=self.key, kind="pop_low", gain=-13),
                  dict(t=self.key + 0.02, kind="success", gain=-17)]
        return c

    def _pill(self, it):
        T = self.f.theme
        key = ("pill", it["label"])

        def mk():
            w = 104 + self.f.type.width(it["label"], 40, 600)
            p = ui.Painter(w, 96, self.ppd * 1.2, self.fonts)
            p.rrect(0, 0, w, 96, 48, fill=T.card, outline=T.border, width=1.5)
            p.circle(48, 48, 21, it.get("color", self.color))
            p.text(84, 62, it["label"], 40, 600, T.text)
            return p.sprite()
        return self.sp(key, mk)

    def draw(self, cv, t):
        f, T = self.f, self.f.theme
        out = 1 - smooth(ramp(t, self.b - 0.15, self.b + 0.2))
        a_in = ease_out(ramp(t, self.t_in, self.t_in + 0.5), 2.6)
        if out * a_in <= 0:
            return
        st = [(self.pos(k, t), k) for k in range(len(self.items))]
        back = [s for s in st if s[0][2] < 0]
        front = [s for s in st if s[0][2] >= 0]
        for (x, y, z, s, c), k in sorted(back, key=lambda q: q[0][2]):
            sp = self._pill(self.items[k])
            pop = spring_in(t - (self.land + 0.1 * k), 4.0, 0.5)
            cv.draw(sp, x, y, scale=s * max(0.01, pop), alpha=out * (1 - c) * 0.9, blur=(1 - (z + 1) / 2) * 3)
        # the product at the centre
        tile = self.sp("tile", lambda: _app_tile(self))
        hit = pulse(t, self.key, 0.02, 0.25)
        cv.draw(tile, f.CX, f.CY, scale=(0.9 + 0.1 * a_in) * (1 + 0.12 * hit), alpha=a_in * out)
        if t >= self.key:
            rr = ease_out(ramp(t, self.key, self.key + 0.7), 2.0)
            ring = [(f.CX + 200 * rr * math.cos(q), f.CY + 200 * rr * math.sin(q)) for q in np.linspace(0, 6.3, 64)]
            cv.light_poly(ring, lin(self.color), 0.9 * (1 - rr) * out, width=4)
        for (x, y, z, s, c), k in sorted(front, key=lambda q: q[0][2]):
            sp = self._pill(self.items[k])
            pop = spring_in(t - (self.land + 0.1 * k), 4.0, 0.5)
            cv.draw(sp, x, y, scale=s * max(0.01, pop), alpha=out * (1 - c))
        txt = self.data.get("text")
        if txt:
            a = smooth(ramp(t, self.key + 0.1, self.key + 0.45)) * out
            sp = f.type.text(txt, 60, 600, f.text_color)
            cv.draw(sp, f.CX, f.CY + 230, alpha=a, anchor=(0.5, sp.baseline / sp.h))


def _app_tile(sc):
    f = sc.f
    s = 220
    p = ui.Painter(s, s, sc.ppd * 1.3, sc.fonts)
    # a dark tile on a dark stage (a light one on a light stage) so the logo, often in the brand colour, reads
    p.rrect(0, 0, s, s, 54, fill=mix_hex(sc.color, f.stage_color, 0.86) if f.stage_mode == "dark" else "#FFFFFF",
            outline=mix_hex(sc.color, f.stage_color, 0.35), width=2.5)
    im = Image.fromarray(f.logo.rgba8)
    k = (s * 0.58 * p.k) / max(im.width, im.height)
    im = im.resize((max(1, int(im.width * k)), max(1, int(im.height * k))), Image.LANCZOS)
    p.im.alpha_composite(im, ((p.im.width - im.width) // 2, (p.im.height - im.height) // 2))
    return p.sprite()


# ===========================================================================
# tunnel: cards on a rotating ring around a headline
# ===========================================================================

@register("tunnel")
class Tunnel(Scene):
    """data: {"images": ["assets/a.png", ...] (optional), "words": ["Books.", "Audio.", "Video."],
              "text": "All in one platform"}"""
    weight = 1.6

    def setup(self):
        d, f = self.data, self.f
        self.words = d.get("words") or ["Fast.", "Simple.", "Everywhere."]
        self.final = d.get("text")
        bs = beats_in(self, self.land + 0.2, self.b - 0.5)
        step = max(1, len(bs) // (len(self.words) + (1 if self.final else 0))) if bs else 1
        self.t_w = [bs[min(len(bs) - 1, i * step)] if bs else self.land + 0.6 * i
                    for i in range(len(self.words) + (1 if self.final else 0))]
        self.key = self.t_w[-1]
        self.cards = []
        imgs = d.get("images") or []
        for i in range(10):
            self.cards.append(imgs[i % len(imgs)] if imgs else None)

    def card_sprite(self, i):
        key = ("tcard", i)

        def mk():
            p = self.cards[i]
            if p:
                path = p if os.path.isabs(p) else os.path.join(self.f.dir, p)
                im = Image.open(path).convert("RGBA")
                im = im.resize((360, int(360 * im.height / im.width)), Image.LANCZOS)
                from .gfx import Sprite
                return Sprite.from_pil(im, self.ppd)
            T = self.f.theme
            p = ui.Painter(300, 420, self.ppd, self.fonts)
            p.rrect(0, 0, 300, 420, 26, fill=T.card, outline=T.border, width=1.2)
            r = rng(i)
            p.rrect(18, 18, 264, 150, 14, fill=mix_hex(self.f.palette[i % len(self.f.palette)], T.card, 0.55))
            for j in range(6):
                p.rrect(18, 190 + j * 34, 264 * r.uniform(0.5, 0.95), 12, 6, fill=T.track)
            return p.sprite()
        return self.sp(key, mk)

    def entry(self):
        return (self.f.CX, self.f.CY)

    def exit(self):
        return (self.f.CX, self.f.CY)

    def anchors(self):
        return {"key": self.key}

    def cues(self):
        return [dict(t=self.a, kind="swirl", gain=-16, dur=1.4)] + [dict(t=tw, kind="tick", gain=-20, pitch=1 + 0.1 * i)
                                                                       for i, tw in enumerate(self.t_w)]

    def draw(self, cv, t):
        f = self.f
        out = 1 - smooth(ramp(t, self.b - 0.15, self.b + 0.2))
        a_in = smooth(ramp(t, self.t_in, self.t_in + 0.5))
        if out * a_in <= 0:
            return
        foc = 1100.0
        R = 820.0 if not f.portrait else 620.0
        spin = (t - self.a) * 0.35
        quads = []
        n = len(self.cards)
        for i in range(n):
            ang = 2 * math.pi * i / n + spin
            zc = 900 + 650 * ((i * 0.37 + (t - self.a) * 0.25) % 1.0)
            cx3, cy3 = R * math.cos(ang), R * math.sin(ang) * 0.62
            tang = np.array([-math.sin(ang), math.cos(ang) * 0.62])
            tang /= np.linalg.norm(tang)
            hw, hh = 150, 210
            corners3 = [(cx3 - tang[0] * hw, cy3 - tang[1] * hw, zc - hh), (cx3 + tang[0] * hw, cy3 + tang[1] * hw, zc - hh),
                        (cx3 + tang[0] * hw, cy3 + tang[1] * hw, zc + hh), (cx3 - tang[0] * hw, cy3 - tang[1] * hw, zc + hh)]
            pts = [(f.CX + X * foc / Z, f.CY + Y * foc / Z) for X, Y, Z in corners3]
            quads.append((zc, i, pts))
        for zc, i, pts in sorted(quads, key=lambda q: -q[0]):
            fog = 1 - ramp(zc, 1200, 1550) * 0.7
            cv.draw_quad(self.card_sprite(i), pts, alpha=a_in * out * fog)
            # a hairline of light around each card so dark UI cards still read on a dark stage
            cv.light_poly(pts + [pts[0]], lin(self.color), 0.35 * a_in * out * fog, width=1.6)
        idx = max([j for j, tw in enumerate(self.t_w) if t >= tw] or [-1])
        if idx >= 0:
            words = self.words[:idx + 1] if idx < len(self.words) else [self.final]
            text = " ".join(words) if idx < len(self.words) else self.final
            k = ease_out(ramp(t, self.t_w[idx], self.t_w[idx] + 0.3), 2.6)
            sp = f.type.text(text, 72, 600, self.color if idx >= len(self.words) else f.text_color)
            cv.draw(sp, f.CX, f.CY + 25 + 12 * (1 - k), alpha=out * (0.4 + 0.6 * k), anchor=(0.5, sp.baseline / sp.h))


# ===========================================================================
# success: a big check, confetti, one word
# ===========================================================================

@register("success")
class Success(Scene):
    """data: {"text": "Done", "sub": "97/100"}"""
    weight = 0.9

    def setup(self):
        # short scenes hit on the first beat after the cut; longer ones let the circle breathe first
        self.t_hit = self.f.snap(max(self.land + 0.3, self.a + min(0.4 * self.L, 0.45 if self.L < 2.5 else 9)))
        self.key = self.t_hit

    def entry(self):
        return (self.f.CX, self.f.CY)

    def exit(self):
        return (self.f.CX, self.f.CY + 120)

    def anchors(self):
        return {"key": self.key}

    def cues(self):
        return [dict(t=self.t_hit - 0.4, kind="riser_short", gain=-22, dur=0.4), dict(t=self.t_hit, kind="pop_low", gain=-12),
                dict(t=self.t_hit + 0.02, kind="success", gain=-14), dict(t=self.t_hit + 0.05, kind="sparkle", gain=-18)]

    def draw(self, cv, t):
        f = self.f
        out = 1 - smooth(ramp(t, self.b - 0.15, self.b + 0.2))
        a_in = smooth(ramp(t, self.t_in, self.t_in + 0.35))
        if out * a_in <= 0:
            return
        cx, cy = f.CX, f.CY + 40
        R = 120
        g = spring_in(t - (self.t_hit - 0.35), 3.6, 0.55)
        circ = [(cx + R * g * math.cos(q), cy + R * g * math.sin(q)) for q in np.linspace(0, 2 * math.pi, 90)]
        cv.fill_poly(circ, mix_hex(self.color, "#FFFFFF", 0.82) if f.stage_mode == "light" else mix_hex(self.color, "#000000", 0.6),
                     alpha=a_in * out)
        cv.fill_poly([(cx + (R + 40) * g * math.cos(q), cy + (R + 40) * g * math.sin(q)) for q in np.linspace(0, 6.3, 60)],
                     self.color, alpha=0.12 * a_in * out, blur=20)
        if t >= self.t_hit - 0.05:
            ck = Path([(cx - 52, cy + 2), (cx - 14, cy + 40), (cx + 56, cy - 36)])
            u = ease_out(ramp(t, self.t_hit - 0.05, self.t_hit + 0.3), 2.2)
            cv.light_poly(ck.span(0, u, 40), lin(self.color), 1.0 * out, width=12)
            cv.light_path(ck, u, tail=0.5, color=self.color, width=12, energy=1.2 * out)
        confetti(cv, (cx, cy), t - self.t_hit, self.i * 7 + 1, f.palette + [mix_hex(self.color, "#FFFFFF", 0.4)])
        txt = self.data.get("text", "Done")
        a = smooth(ramp(t, self.t_hit - 0.1, self.t_hit + 0.25)) * out
        sp = f.type.text(txt, 84, 500, self.color)
        cv.draw(sp, cx, cy - 190 + 16 * (1 - a), alpha=a, anchor=(0.5, sp.baseline / sp.h))
        if self.data.get("sub"):
            ss = f.type.text(self.data["sub"], 40, 500, f.text2)
            cv.draw(ss, cx, cy + 210, alpha=a, anchor=(0.5, ss.baseline / ss.h))


# ===========================================================================
# montage: images cut on every beat
# ===========================================================================

@register("montage")
class Montage(Scene):
    """data: {"images": ["assets/shot1.jpg", ...], "every": 1 (beats per shot), "text": "IS COMING",
              "grade": "teal-orange" | "none"}"""
    weight = 1.4

    def setup(self):
        d = self.data
        self.images = d.get("images") or []
        if not self.images:
            raise ValueError("montage needs data.images (product shots, screenshots, frames)")
        cuts = beats_in(self, self.t_in, self.b, int(d.get("every", 1)))
        self.cuts = cuts if len(cuts) >= 2 else list(np.linspace(self.t_in, self.b, len(self.images) + 1))
        self.key = self.cuts[min(len(self.cuts) - 1, 2)]

    def shot(self, i):
        key = ("shot", i)

        def mk():
            p = self.images[i % len(self.images)]
            path = p if os.path.isabs(p) else os.path.join(self.f.dir, p)
            im = Image.open(path).convert("RGB")
            W, H = self.f.DW, self.f.DH
            k = max(W * 1.15 / im.width, H * 1.15 / im.height)
            im = im.resize((int(im.width * k), int(im.height * k)), Image.LANCZOS)
            a = np.asarray(im).astype(np.float32) / 255
            if self.data.get("grade", "teal-orange") == "teal-orange":
                lum = a.mean(-1, keepdims=True)
                teal, orange = np.array([0.0, 0.32, 0.38]), np.array([1.0, 0.62, 0.32])
                a = a * 0.78 + (teal * (1 - lum) + orange * lum) * 0.22
                a = np.clip((a - 0.04) * 1.12, 0, 1)
            im = Image.fromarray((a * 255).astype(np.uint8)).convert("RGBA")
            from .gfx import Sprite
            return Sprite.from_pil(im, im.width / (W * 1.15))   # design width = 1.15 x frame
        return self.sp(key, mk)

    def entry(self):
        return (self.f.CX, self.f.CY)

    def exit(self):
        return (self.f.CX, self.f.CY)

    def anchors(self):
        return {"key": self.key}

    def cues(self):
        return [dict(t=c, kind="transient", gain=-18) for c in self.cuts[1:-1]]

    def draw(self, cv, t):
        f = self.f
        out = 1 - smooth(ramp(t, self.b - 0.1, self.b + 0.05))
        if t < self.t_in or out <= 0:
            return
        i = max([j for j, c in enumerate(self.cuts) if t >= c] or [0])
        c0 = self.cuts[i]
        c1 = self.cuts[i + 1] if i + 1 < len(self.cuts) else self.b
        k = ramp(t, c0, c1)
        sp = self.shot(i)
        z = 1.0 + 0.12 * ease_out(k, 1.6)
        dx = (40 if i % 2 else -40) * k
        cv.draw(sp, f.CX + dx, f.CY, scale=z, alpha=out)
        txt = self.data.get("text")
        if txt:
            disp = f.display_type()
            ts = disp.text(txt, 110, 700, "#FFFFFF")
            a = smooth(ramp(t, self.t_in, self.t_in + 0.4)) * out
            cv.draw(ts, f.CX, f.CY + 40, alpha=a, anchor=(0.5, ts.baseline / ts.h))


# ===========================================================================
# grid: integration tiles on a tilted 3D plane
# ===========================================================================

@register("grid")
class Grid3D(Scene):
    """data: {"items": [{"label": "Slack", "color": "#4A154B"}, ...] (6-12), "title": "Connect any source"}"""
    weight = 1.4

    def setup(self):
        d = self.data
        self.items = d.get("items") or [dict(label=s, color=c) for s, c in
                                        (("Notion", "#111111"), ("Slack", "#E01E5A"), ("Drive", "#0EBC5F"),
                                         ("GitHub", "#24292F"), ("Figma", "#A259FF"), ("Gmail", "#EA4335"),
                                         ("Zendesk", "#03363D"), ("Stripe", "#635BFF"))]
        bs = beats_in(self, self.land + 0.2, self.b - 0.4)
        n = len(self.items)
        self.t_pop = [bs[min(len(bs) - 1, k // 2)] if bs else self.land + 0.15 * k for k in range(n)]
        self.key = self.t_pop[-1] + 0.3

    def tile(self, it):
        T = self.f.theme
        key = ("gtile", it["label"])

        def mk():
            s = 150
            p = ui.Painter(s, s, self.ppd * 1.2, self.fonts)
            p.rrect(0, 0, s, s, 34, fill=T.card, outline=T.border, width=1.5)
            p.rrect(30, 30, s - 60, s - 60, 22, fill=it.get("color", self.color))
            p.text(s / 2, s / 2 + 18, it["label"][0].upper(), 52, 700, "#FFFFFF", anchor="ms")
            return p.sprite()
        return self.sp(key, mk)

    def entry(self):
        return (self.f.CX - 400, self.f.CY)

    def exit(self):
        return (self.f.CX + 400, self.f.CY)

    def anchors(self):
        return {"key": self.key}

    def cues(self):
        return [dict(t=tp, kind="pop", gain=-21, pitch=0.9 + 0.05 * k) for k, tp in enumerate(self.t_pop)]

    def draw(self, cv, t):
        f = self.f
        out = 1 - smooth(ramp(t, self.b - 0.15, self.b + 0.2))
        a_in = smooth(ramp(t, self.t_in, self.t_in + 0.4))
        if out * a_in <= 0:
            return
        n = len(self.items)
        cols = 4 if n > 6 else 3
        rows = int(math.ceil(n / cols))
        tilt = math.radians(lerp(52, 30, ease_in_out(ramp(t, self.a, self.b), 1.5)))
        yaw = math.radians(lerp(-14, 10, ease_in_out(ramp(t, self.a, self.b), 1.5)))
        foc = 1300.0
        gap = 190.0
        items = []
        for k, it in enumerate(self.items):
            c, r_ = k % cols, k // cols
            X = (c - (cols - 1) / 2) * gap
            Y = (r_ - (rows - 1) / 2) * gap
            corners = []
            for dx, dy in ((-75, -75), (75, -75), (75, 75), (-75, 75)):
                x, y = X + dx, Y + dy
                x1 = x * math.cos(yaw)
                z1 = x * math.sin(yaw)
                y2 = y * math.cos(tilt)
                z2 = y * math.sin(tilt) + z1 + 900
                corners.append((f.CX + x1 * foc / z2, f.CY + 70 + y2 * foc / z2))
            pop = spring_in(t - self.t_pop[k], 4.0, 0.5)
            ccx = sum(p[0] for p in corners) / 4
            ccy = sum(p[1] for p in corners) / 4
            pts = [(ccx + (px - ccx) * max(0.01, pop), ccy + (py - ccy) * max(0.01, pop)) for px, py in corners]
            items.append((Y, pts, it, pop))
        for Y, pts, it, pop in sorted(items, key=lambda q: q[0]):
            if pop > 0.01:
                cv.draw_quad(self.tile(it), pts, alpha=a_in * out)
        title = self.data.get("title")
        if title:
            a = smooth(ramp(t, self.t_in, self.t_in + 0.5)) * out
            sp = f.type.text(title, 66, 600, f.text_color)
            cv.draw(sp, f.CX, f.CY - 300 if not f.portrait else f.CY - 500, alpha=a, anchor=(0.5, sp.baseline / sp.h))
