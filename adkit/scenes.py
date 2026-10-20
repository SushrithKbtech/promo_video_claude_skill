"""Scene library. Each scene is one *job* done by one light.

    upload   files fly into a folder; the light IS the progress bar
    search   the light is the search field's focus line, then traces the result
    share    a 3D page; the light waits on Share, then splits into live cursors
    devices  the page flies into a laptop; lights chase a loop behind the devices
    flow     the light connects steps (workflows, automations, prototypes, pipelines)
    chart    the light draws a rising line; a number rolls to its value
    chat     a prompt is typed; the light streams the answer (AI products)
    screen   a real screenshot in a browser frame; the light traces what matters

Every scene exposes entry() / exit() points for hand-overs, anchors() for the
voice ("key" is where the key word lands), cues() for sound and draw(cv, t).
Add a custom scene by subclassing Scene in the project's custom_scenes.py and
registering it in SCENES (see SKILL.md).
"""
from __future__ import annotations

import math
import os

import numpy as np
from PIL import Image

from . import ui
from .core import (Path, clamp, chaikin, damped, ease_in, ease_in_out, ease_out, lerp, lin, mix_hex,
                   pulse, ramp, resample, rng, smooth, smoother, spring_in, window)

SCENES = {}


def register(name):
    def deco(cls):
        SCENES[name] = cls
        cls.kind = name
        return cls
    return deco


def rrect_points(x, y, w, h, r, n=320, start="left"):
    """Closed rounded-rectangle outline, clockwise from the left-middle."""
    r = min(r, w / 2, h / 2)
    pts = []
    def arc(cx, cy, a0, a1, k=16):
        for a in np.linspace(a0, a1, k):
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    pts.append((x, y + h / 2))
    arc(x + r, y + r, math.pi, 1.5 * math.pi)
    arc(x + w - r, y + r, 1.5 * math.pi, 2 * math.pi)
    arc(x + w - r, y + h - r, 0, 0.5 * math.pi)
    arc(x + r, y + h - r, 0.5 * math.pi, math.pi)
    pts.append((x, y + h / 2))
    return resample(np.array(pts), n, closed=False)


class Scene:
    weight = 1.0
    handover = 0.6

    def __init__(self, film, spec, idx, a, b):
        self.f = film
        self.spec = spec
        self.i = idx
        self.a, self.b = a, b
        self.L = b - a
        self.data = spec.get("data", {})
        c = spec.get("light", idx)
        self.color = film.light_color(c) if isinstance(c, int) else c
        self.facet = spec.get("facet", idx)
        self.transition = spec.get("transition", "light")
        # scenes entered by a camera move appear at once; after a light hand-over they wait for it
        self.t_in = a + (0.0 if self.transition in ("push", "wipe", "whip", "cut") else 0.18)
        self.land = a if idx == 0 else a + (0.22 if self.transition in ("whip", "cut") else
                                            0.3 if self.transition in ("wipe", "push") else self.handover)
        self.prev = None
        self.next = None
        self.fit = 1.0

    # --- helpers ---------------------------------------------------------
    def sp(self, key, fn):
        return self.f.sprite((self.i, key), fn)

    def text(self, s, size, weight=400, color=None):
        return self.f.type.text(s, size, weight, color or self.f.text_color)

    @property
    def T(self):
        return self.f.theme

    @property
    def fonts(self):
        return self.f.fonts

    @property
    def ppd(self):
        return self.f.S

    def fade(self, t, early=0.3, late=0.45):
        return window(t, self.a - early, self.b + late, early, late)

    # --- interface -------------------------------------------------------
    def setup(self):
        pass

    def entry(self):
        return (self.f.CX, self.f.CY)

    def exit(self):
        return (self.f.CX, self.f.CY)

    def anchors(self):
        return {"key": self.a + self.L * 0.6}

    def tint_color(self):
        return self.color

    def cues(self):
        return []

    def race_starts(self, n, t):
        p = np.array(self.exit(), float)
        out = []
        for i in range(n):
            ang = -math.pi / 2 + i * 2 * math.pi / n
            d = np.array([math.cos(ang), math.sin(ang)])
            out.append((p + d * 160, d))
        return out

    def draw(self, cv, t):
        pass


# ===========================================================================
# upload
# ===========================================================================

@register("upload")
class Upload(Scene):
    weight = 2.35

    def setup(self):
        f = self.f
        d = self.data
        self.fit = min(1.0, (f.DW - 80) / 1000)
        k = self.fit
        self.F = np.array([f.CX, f.CY - 150 * k])
        self.C = np.array([f.CX, f.CY + 120 * k])
        self.cw, self.ch = 960 * k, 150 * k
        self.xl, self.xr = self.C[0] - self.cw / 2 + 40 * k, self.C[0] + self.cw / 2 - 40 * k
        self.yt = self.C[1] + 40 * k
        self.count_label = d.get("count_label", "1,284 items")
        self.uploading = d.get("uploading", f"Uploading {self.count_label}")
        self.done_label = d.get("done", "Backed up")
        self.folder_label = d.get("folder", "My Drive")
        drop = f.drop if self.a + 1.5 < f.drop < self.b - 0.3 else self.a + 0.857 * self.L
        self.drop = drop
        self.done = min(drop + 0.4, self.b - 0.15)
        R = drop - self.a
        waves = d.get("waves") or [
            dict(kinds=["doc", "sheet", "pdf", "slides"], count=8),
            dict(kinds=["photo", "video"], count=8),
            dict(kinds=["slides", "doc", "design"], count=9),
        ]
        spans = [(0.115, 0.281), (0.365, 0.531), (0.635, 0.833)]
        files = []
        r = rng(41 + self.i)
        for wi, wv in enumerate(waves[:3]):
            s0, s1 = spans[wi]
            n = int(wv.get("count", 8))
            for j in range(n):
                tl = self.a + R * lerp(s0, s1, (j + r.uniform(0.1, 0.9)) / n)
                files.append(dict(tl=tl, kind=wv["kinds"][j % len(wv["kinds"])], wave=wi))
        allk = [k_ for wv in waves for k_ in wv["kinds"]]
        for j in range(3):
            files.append(dict(tl=self.a + R * lerp(0.844, 0.958, j / 2.5), kind=allk[(j * 3) % len(allk)], wave=3))
        for j in range(int(d.get("drop_count", 12))):
            files.append(dict(tl=drop + r.uniform(-0.02, 0.02), kind=allk[(j * 5 + 1) % len(allk)], wave=4))
        self.wave_starts = [min(fl["tl"] for fl in files if fl["wave"] == w) for w in range(3)]
        n = len(files)
        golden = math.pi * (3 - math.sqrt(5))
        for j, fl in enumerate(files):
            ang = (j * golden + r.uniform(-0.2, 0.2)) % (2 * math.pi)
            if 0.25 * math.pi < ang < 0.75 * math.pi:  # keep paths from crossing the card
                ang += math.pi
            rad = r.uniform(1050, 1400) * (0.75 if f.portrait else 1.0)
            S = self.F + np.array([math.cos(ang), math.sin(ang) * 0.75]) * rad
            E = self.F + np.array([0, 6 * k])
            fl["path"] = Path.through(S, E, bend=r.uniform(0.12, 0.28), side=1 if r.random() < 0.5 else -1)
            fl["dur"] = 0.6 if fl["wave"] == 4 else r.uniform(0.7, 0.9)
            fl["rot"] = r.uniform(-25, 25)
            fl["seed"] = j
            fl["pan"] = math.cos(ang) * 0.6
        self.files = files
        self.landings = sorted(fl["tl"] for fl in files)

    def entry(self):
        return (self.xl, self.yt)

    def exit(self):
        return (self.xr, self.yt)

    def anchors(self):
        return {"w1": self.wave_starts[0] - 0.05, "w2": self.wave_starts[1] - 0.07,
                "w3": self.wave_starts[2] - 0.05, "drop": self.drop, "key": self.done}

    def progress(self, t):
        if t >= self.done:
            return 1.0
        return 0.985 * ease_in_out(ramp(t, self.land, self.done - 0.08), 1.7)

    def cues(self):
        c = []
        for fl in self.files:
            c.append(dict(t=fl["tl"], kind="key", gain=-23 if fl["wave"] < 4 else -27,
                          pitch=0.8 + 0.45 * ((fl["seed"] * 0.618) % 1), pan=fl["pan"]))
            if fl["seed"] % 6 == 2 and fl["wave"] < 4:
                c.append(dict(t=fl["tl"] - fl["dur"] * 0.45, kind="whoosh_short", gain=-25, pan=fl["pan"]))
        c += [dict(t=self.drop, kind="whoosh_hit", gain=-12), dict(t=self.drop, kind="transient", gain=-10),
              dict(t=self.drop + 0.01, kind="pop_low", gain=-14),
              dict(t=self.done, kind="success", gain=-15), dict(t=self.done + 0.05, kind="sparkle", gain=-20)]
        return c

    def draw(self, cv, t):
        f, k, T = self.f, self.fit, self.T
        vis = self.fade(t, 0.3, 0.5)
        if vis <= 0:
            return
        app = ease_out(ramp(t, self.a - 0.3, self.a + 0.25))
        out = 1 - smooth(ramp(t, self.b + 0.05, self.b + 0.5))
        al = app * out
        sc_in = 0.96 + 0.04 * app
        # files (behind the folder, so they land *into* it)
        for fl in self.files:
            ts = fl["tl"] - fl["dur"]
            if not (ts <= t <= fl["tl"]):
                continue
            sp = self.sp(("file", fl["kind"], fl["seed"] % 6),
                         lambda fl=fl: ui.file_card(fl["kind"], fl["seed"] % 6, self.ppd * 1.25, self.fonts, T))
            def fn(tt, fl=fl, ts=ts):
                x = clamp((tt - ts) / fl["dur"])
                p = fl["path"].at(x ** 1.35)
                return dict(x=p[0], y=p[1], scale=lerp(1.25, 0.22, ease_in(x, 2.0)) * k,
                            rot=fl["rot"] * (1 - x), alpha=min(1.0, x * 5) * out)
            cv.draw_motion(sp, fn, t, 0.75 / self.f.fps, n=5)
        # folder with landing pulses
        pul = 0.0
        for tl in self.landings:
            if tl - 0.02 <= t <= tl + 0.3:
                pul += 0.035 * pulse(t, tl, 0.02, 0.1)
        pul += 0.10 * pulse(t, self.drop, 0.03, 0.22)
        fsp = self.sp("folder", lambda: ui.folder(250, 190, self.ppd * 1.15, self.fonts, T))
        cv.draw(fsp, self.F[0], self.F[1], scale=k * sc_in * (1 + min(pul, 0.16)), alpha=al)
        lab = self.text(self.folder_label, 26, 500, f.text2)
        cv.draw(lab, self.F[0], self.F[1] + 128 * k, scale=k, alpha=al, anchor=(0.5, lab.baseline / lab.h))
        # upload card
        card = self.sp("card", lambda: ui.card(960, 150, 22, self.ppd, self.fonts, T))
        cv.draw(card, self.C[0], self.C[1], scale=k * sc_in, alpha=al)
        trk = self.sp("track", lambda: ui.bar_track(880, self.ppd, self.fonts, T))
        cv.draw(trk, self.C[0], self.yt, scale=k * sc_in, alpha=al)
        x_txt, y_txt = self.C[0] - self.cw / 2 + 40 * k, self.C[1] - 6 * k
        dn = smooth(ramp(t, self.done - 0.02, self.done + 0.16))
        if dn < 1:
            s1 = self.text(self.uploading, 34, 400)
            cv.draw(s1, x_txt, y_txt, scale=k, alpha=al * (1 - dn), anchor=(0, s1.baseline / s1.h))
            pct = self.text(f"{int(round(self.progress(t) * 100))}%", 34, 500, self.color)
            cv.draw(pct, self.C[0] + self.cw / 2 - 40 * k, y_txt, scale=k, alpha=al * (1 - dn),
                    anchor=(1, pct.baseline / pct.h))
        if dn > 0:
            chk = self.sp("check", lambda: _check_sprite(self, self.color))
            pop = spring_in(t - self.done, 4.5, 0.5)
            cv.draw(chk, x_txt + 17 * k, y_txt - 12 * k, scale=k * max(0.01, pop), alpha=al * dn)
            s2 = self.text(self.done_label, 34, 500)
            cv.draw(s2, x_txt + 50 * k, y_txt, scale=k, alpha=al * dn, anchor=(0, s2.baseline / s2.h))
            s3 = self.text(self.count_label, 30, 400, f.text2)
            cv.draw(s3, self.C[0] + self.cw / 2 - 40 * k, y_txt, scale=k, alpha=al * dn,
                    anchor=(1, s3.baseline / s3.h))
        # the light is the bar fill (hot head), then drains into its head
        if t >= self.land - 0.01:
            p = self.progress(t)
            x_head = self.xl + (self.xr - self.xl) * p
            drain = ease_in(ramp(t, self.b - 0.05, self.b + 0.25), 2.0)
            x_tail = lerp(self.xl, x_head, drain)
            if x_head - x_tail > 1 and drain < 1:
                n = 40
                xs = np.linspace(x_tail, x_head, n)
                w = np.linspace(0, 1, n)
                inten = 0.78 * (0.85 + 0.15 * w ** 4)
                cv.light_poly(np.stack([xs, np.full(n, self.yt)], 1), lin(self.color), inten, width=5.2 * k)
            if t < self.b + 0.2:
                cv.light_dot((x_head, self.yt), lin(self.color), energy=1.5, radius=5.0 * k)
            # success glint along the full bar
            if self.done <= t <= self.done + 0.6:
                gp = Path([(self.xl, self.yt), (self.xr, self.yt)])
                cv.light_path(gp, ease_in_out(ramp(t, self.done, self.done + 0.45)) * 1.25, tail=0.25,
                              color="#FFFFFF" if f.stage_mode == "dark" else self.color, width=7 * k,
                              energy=0.9, head_glow=0.6)


def _check_sprite(sc, color):
    p = ui.Painter(34, 34, sc.ppd, sc.fonts)
    ui.draw_check(p, 17, 17, 15, color)
    return p.sprite()


# ===========================================================================
# search
# ===========================================================================

@register("search")
class Search(Scene):
    weight = 1.0

    def setup(self):
        f, d = self.f, self.data
        self.fit = k = min(1.0, (f.DW - 80) / 1000)
        self.pw, self.ph = 1000 * k, 92 * k
        self.P = np.array([f.CX, f.CY - 190 * k])
        self.placeholder = d.get("placeholder", f"Search in {f.name}")
        self.query = d.get("query", "q3 budget")
        self.results = d.get("results") or [
            dict(kind="doc", title="Q3 Budget Plan", meta="Edited 2 days ago"),
            dict(kind="sheet", title="Q3 budget - numbers", meta="Edited last week"),
            dict(kind="slides", title="Q3 budget review", meta="Edited Mar 4"),
        ]
        self.sel = int(d.get("select", 0))
        self.t_type = self.a + 0.88
        self.cps = 0.062
        self.t_typed = self.t_type + self.cps * len(self.query)
        self.key = max(self.a + 0.72 * self.L, self.t_typed + 0.5)
        self.rows_y0 = self.P[1] + self.ph / 2 + 34 * k
        self.line_y = self.P[1] + self.ph / 2 + 7 * k
        self.lx0, self.lx1 = self.P[0] - self.pw / 2 + 46 * k, self.P[0] + self.pw / 2 - 46 * k

    def row_rect(self, i, lift=0.0):
        k = self.fit
        y = self.rows_y0 + i * (78 + 10) * k - lift * k
        return (self.P[0] - 500 * k, y, 1000 * k, 78 * k)

    def entry(self):
        return (self.lx0, self.line_y)

    def exit(self):
        x, y, w, h = self.row_rect(self.sel, 8)
        return (x, y + h / 2)

    def exit_rect(self):
        return self.row_rect(self.sel, 8)

    def anchors(self):
        return {"type": self.t_type, "key": self.key}

    def cues(self):
        c = [dict(t=self.t_type + i * self.cps, kind="key", gain=-22, pitch=0.95 + 0.25 * ((i * 0.37) % 1),
                  pan=-0.2 + 0.4 * i / max(1, len(self.query))) for i, ch in enumerate(self.query) if ch != " "]
        for i in range(len(self.results)):
            c.append(dict(t=self.t_typed + 0.08 + i * 0.09, kind="tick", gain=-24, pitch=1 + 0.1 * i))
        c += [dict(t=self.key, kind="select", gain=-17), dict(t=self.key + 0.02, kind="whoosh_short", gain=-21)]
        return c

    def draw(self, cv, t):
        f, k, T = self.f, self.fit, self.T
        morph = isinstance(self.next, Share)
        app = ease_out(ramp(t, self.a + 0.45, self.a + 0.8), 2.5)
        out_others = 1 - smooth(ramp(t, self.b - 0.15, self.b + 0.2))
        out_sel = 1 - smooth(ramp(t, self.b + 0.05, self.b + 0.3)) if morph else out_others
        if app <= 0 or (out_others <= 0 and out_sel <= 0):
            return
        pill = self.sp("pill", lambda: ui.search_pill(self.placeholder, self.ppd, self.fonts, T))
        cv.draw(pill, self.P[0], self.P[1], scale=k * (0.92 + 0.08 * app), alpha=app * out_others)
        tx, ty = self.P[0] - self.pw / 2 + 96 * k, self.P[1] + 10 * k
        if t < self.t_type + 0.08:
            ph = self.text(self.placeholder, 28, 400, f.text2)
            cv.draw(ph, tx, ty, scale=k, alpha=app * out_others * (1 - ramp(t, self.t_type - 0.02, self.t_type + 0.08)),
                    anchor=(0, ph.baseline / ph.h))
        n = int(clamp((t - self.t_type) / self.cps, 0, len(self.query))) if t >= self.t_type else 0
        q = self.text(self.query, 30, 400)
        if n > 0:
            edge = tx + q.meta["advance"][n] * k
            cv.draw(q, tx, ty, scale=k, alpha=out_others, anchor=(0, q.baseline / q.h), wipe=(edge + 1, 1.0))
        # caret (as light, so it reads as the yellow focus)
        if self.t_type - 0.25 <= t <= self.key:
            blink = 1.0 if t < self.t_typed + 0.1 else (0.5 + 0.5 * math.cos((t - self.t_typed) * 2 * math.pi * 1.1))
            cx = tx + (q.meta["advance"][n] if n else 0) * k + 3 * k
            cv.light_poly([(cx, ty - 28 * k), (cx, ty + 6 * k)], lin(self.color), 0.9 * blink * out_others, width=3 * k)
        # focus line under the pill + running glint
        if t >= self.land - 0.02:
            g = ease_out(ramp(t, self.land, self.land + 0.4), 2.0)
            xe = lerp(self.lx0, self.lx1, g)
            n_ = 50
            xs = np.linspace(self.lx0, xe, n_)
            fade = out_others
            cv.light_poly(np.stack([xs, np.full(n_, self.line_y)], 1), lin(self.color), 0.62 * fade, width=4.2 * k)
            if g >= 1:
                gl = Path([(self.lx0, self.line_y), (self.lx1, self.line_y)])
                ph_ = ((t - self.land - 0.4) / 1.25) % 1.0
                cv.light_path(gl, ph_ * 1.3, tail=0.22, color=self.color, width=4.6 * k, energy=0.8 * fade, head_glow=0.4)
        # result rows
        for i, r in enumerate(self.results):
            t_in = self.t_typed + 0.05 + i * 0.09
            a_in = ease_out(ramp(t, t_in, t_in + 0.28), 2.5)
            if a_in <= 0:
                continue
            row = self.sp(("row", i), lambda r=r: ui.result_row(r["kind"], r["title"], r.get("meta", ""), self.ppd,
                                                                 self.fonts, T))
            selk = smooth(ramp(t, self.key - 0.05, self.key + 0.15))
            lift = 8 * selk if i == self.sel else 0
            x, y, w, h = self.row_rect(i, lift)
            o = out_sel if i == self.sel else out_others
            cv.draw(row, x + w / 2, y + h / 2 + 20 * k * (1 - a_in), scale=k, alpha=a_in * o,
                    dim=0.6 * selk if i != self.sel else 0)
        # trace the selected row's outline
        if t >= self.key:
            x, y, w, h = self.row_rect(self.sel, 8)
            pts = self.sp(("trace", round(k, 3)), lambda: rrect_points(x, y, w, h, 16 * k, 360))
            path = Path(pts)
            u = ease_out(ramp(t, self.key, self.key + 0.55), 2.2)
            cv.light_path(path, u, tail=0.45, color=self.color, width=4.0 * k, energy=1.05 * out_sel)
            hold = 0.42 * out_sel * (1 - 0.4 * ramp(t, self.key + 0.55, self.b)) * smooth(ramp(t, self.key + 0.25,
                                                                                               self.key + 0.7))
            if hold > 0:
                cv.light_poly(pts, lin(self.color), hold, width=3.4 * k)


# ===========================================================================
# share (3D page, cursors)
# ===========================================================================

@register("share")
class Share(Scene):
    weight = 1.25

    def setup(self):
        f, d = self.f, self.data
        self.pw, self.ph = 1240.0, 720.0
        self.fit = min(1.0, (f.DW - 120) / self.pw, (f.DH - 220) / self.ph)
        self.center = np.array([f.CX, f.CY + 10])
        self.title = d.get("title", "Q3 Budget Plan")
        self.content = d.get("page", "doc")
        ppl = d.get("people") or [dict(name="Ada"), dict(name="Tomi"), dict(name="Kemi")]
        pal = [c for c in f.palette if c.upper() != self.color.upper()] or f.palette
        for j, p in enumerate(ppl):
            p.setdefault("color", self.color if j == 0 else pal[(j - 1) % len(pal)])
        self.people = ppl[:3]
        self.share_label = d.get("button", "Share")
        self.push = 0.75
        self.land = self.a + self.push + 0.12
        self.tap = max(self.a + 0.61 * self.L, self.land + 0.55)
        self.focal = 2400.0

    def page(self):
        return self.sp("page", lambda: ui.doc_page(self.title, self.ppd, self.fonts, self.T,
                                                   share_label=self.share_label, avatars=(),
                                                   content=self.content, file_name=self.data.get("file", self.title),
                                                   header_kind=self.data.get("kind", "doc"), w=self.pw, h=self.ph))

    def layout(self):
        return self.page()[1]

    def pose(self, t):
        """center, scale, yaw, pitch of the page at time t."""
        k = self.fit
        g = ease_in_out(ramp(t, self.a, self.a + self.push), 2.4)
        if isinstance(self.prev, Search):
            x, y, w, h = self.prev.exit_rect()
            s0 = w / self.pw
            c0 = np.array([x + w / 2, y + h / 2])
        else:
            s0, c0 = 0.94 * k, self.center
        s = lerp(s0, k, g)
        c = lerp(c0, self.center, g)
        tilt = smooth(ramp(t, self.a + 0.2, self.a + self.push + 0.2))
        yaw = lerp(7.0, -5.0, smooth(ramp(t, self.a + self.push, self.b + 0.6))) * tilt
        pitch = 3.5 * tilt
        return c, s, math.radians(yaw), math.radians(pitch)

    def proj(self, t, lx, ly, pose=None):
        c, s, yaw, pitch = pose or self.pose(t)
        X, Y = (lx - self.pw / 2) * s, (ly - self.ph / 2) * s
        Z = 0.0
        X1 = X * math.cos(yaw) + Z * math.sin(yaw)
        Z1 = -X * math.sin(yaw) + Z * math.cos(yaw)
        Y2 = Y * math.cos(pitch) - Z1 * math.sin(pitch)
        Z2 = Y * math.sin(pitch) + Z1 * math.cos(pitch)
        q = self.focal / (self.focal + Z2)
        return np.array([c[0] + X1 * q, c[1] + Y2 * q]), s * q

    def quad(self, t):
        pose = self.pose(t)
        return [self.proj(t, x, y, pose)[0] for x, y in ((0, 0), (self.pw, 0), (self.pw, self.ph), (0, self.ph))]

    def share_center(self, t):
        sx, sy, sw, sh = self.layout()["share"]
        return self.proj(t, sx + sw / 2, sy + sh / 2)

    def entry(self):
        return tuple(self.share_center(self.land)[0])

    def exit(self):
        return tuple(self.share_center(self.b)[0])

    def anchors(self):
        return {"key": self.tap, "push": self.a}

    def targets(self):
        bars = self.layout()["bars"]
        pick = [1, 4, 7] if len(bars) >= 8 else list(range(min(3, len(bars))))
        return [bars[i] for i in pick][:len(self.people)]

    def cues(self):
        c = [dict(t=self.a + 0.05, kind="swoosh", gain=-18, dur=0.7),
             dict(t=self.tap, kind="click", gain=-14), dict(t=self.tap + 0.03, kind="send", gain=-18)]
        for j in range(len(self.people)):
            arr = self.tap + 0.04 * j + 0.32
            c.append(dict(t=arr, kind="pop", gain=-17, pitch=1.0 + 0.18 * j, pan=-0.4 + 0.4 * j))
            for s in range(0, 22, 3):
                c.append(dict(t=arr + 0.05 + s * 0.72 / 22, kind="key", gain=-30, pitch=1.1 + 0.05 * j,
                              pan=-0.4 + 0.4 * j))
            c.append(dict(t=arr + 0.1, kind="pop_soft", gain=-24, pitch=1.3 + 0.15 * j))
        return c

    def draw(self, cv, t):
        f, T = self.f, self.T
        if t < self.a - 0.3:
            return
        nxt = self.next
        handoff = isinstance(nxt, Devices)
        if handoff and t > nxt.a + 0.02:
            return  # the devices scene now owns the page
        out = 1.0 if handoff else 1 - smooth(ramp(t, self.b - 0.15, self.b + 0.2))
        a_in = smooth(ramp(t, self.a, self.a + 0.18)) if isinstance(self.prev, Search) else \
            smooth(ramp(t, self.a - 0.05, self.a + 0.4))
        al = a_in * out
        if al <= 0:
            return
        sp_page, lay = self.page()
        pose = self.pose(t)
        cv.draw_quad(sp_page, [self.proj(t, x, y, pose)[0] for x, y in
                               ((0, 0), (self.pw, 0), (self.pw, self.ph), (0, self.ph))], alpha=al)
        # share button (separate so it can press)
        sx, sy, sw, sh = lay["share"]
        pc, ps = self.proj(t, sx + sw / 2, sy + sh / 2, pose)
        btn = self.sp("btn", lambda: _share_btn(self, sw, sh))
        press = 1 - 0.07 * (smooth(ramp(t, self.tap - 0.02, self.tap + 0.04)) *
                            (1 - smooth(ramp(t, self.tap + 0.12, self.tap + 0.2))))
        cv.draw(btn, pc[0], pc[1], scale=ps * press, alpha=al)
        # the waiting light circles the button
        if self.land - 0.05 <= t <= self.tap + 0.1:
            pts = [self.proj(t, sx + x, sy + y, pose)[0] for x, y in rrect_points(0, 0, sw, sh, sh / 2, 160)]
            path = Path(np.array(pts), closed=True)
            u = (t - self.land) * 1.3
            fade = 1 - smooth(ramp(t, self.tap, self.tap + 0.1))
            cv.light_path(path, u, tail=0.42, color=self.color, width=3.6 * ps, energy=1.0, fade=fade)
        # strands -> cursors -> typing
        for j, (p, bar) in enumerate(zip(self.people, self.targets())):
            t0 = self.tap + 0.04 * j
            arr = t0 + 0.32
            if t < t0:
                continue
            bx, by, bw, bh = bar
            tgt_len = bw * (0.62 + 0.12 * j)
            steps = int(clamp((t - arr - 0.05) / 0.72, 0, 1) * 22)
            typed = tgt_len * steps / 22
            cur_local = (bx + typed, by + bh / 2)
            cpos, cs = self.proj(t, *cur_local, pose)
            if t <= arr + 0.15:
                start = self.share_center(t0)[0]
                key = ("strand", j)
                end = self.proj(arr, bx, by + bh / 2)[0]
                path = self.f.sprite((self.i, key), lambda s=start, e=end, j=j:
                                     Path.through(s, e, bend=0.18 + 0.1 * j, side=1 if j % 2 else -1))
                u = ease_out(ramp(t, t0, arr), 2.0) + max(0.0, t - arr) * 5
                cv.light_path(path, u, tail=0.45, color=p["color"], color_tail=self.color, width=4.0, energy=1.1)
            if t >= arr:
                pop = spring_in(t - arr, 4.5, 0.45)
                if typed > 1:
                    tb = self.sp(("typed", j), lambda c=p["color"]: ui.bar(400, 10, mix_hex(c, "#FFFFFF", 0.35),
                                                                           self.ppd, self.fonts))
                    lp, ls = self.proj(t, bx, by + bh / 2, pose)
                    cv.draw(tb, lp[0], lp[1], sx=ls * typed / 400, sy=ls, anchor=(0, 0.5), alpha=al)
                csp = self.sp(("cur", j), lambda c=p["color"]: ui.cursor(c, self.ppd, self.fonts))
                cv.draw(csp, cpos[0], cpos[1], scale=cs * max(0.01, pop), anchor=(0.07, 0.05), alpha=al)
                flag = self.sp(("flag", j), lambda p=p: ui.name_flag(p["name"], p["color"], self.ppd, self.fonts))
                cv.draw(flag, cpos[0] + 22 * cs, cpos[1] + 34 * cs, scale=cs * max(0.01, pop), anchor=(0, 0),
                        alpha=al)
                # avatar pops in the header
                av_x = sx - 30 - (len(self.people) - 1 - j) * 32
                ap, aps = self.proj(t, av_x, sy + sh / 2, pose)
                av = self.sp(("av", j), lambda p=p: ui.avatar(p["color"], p["name"][0], self.ppd, self.fonts))
                cv.draw(av, ap[0], ap[1], scale=aps * max(0.01, spring_in(t - arr - 0.1, 5.0, 0.4)), alpha=al)


def _share_btn(sc, w, h):
    p = ui.Painter(w, h, sc.ppd, sc.fonts)
    p.rrect(0, 0, w, h, h / 2, fill=sc.T.accent_bg)
    p.text(w / 2, h / 2 + 8, sc.share_label, 22, 500, sc.T.accent_fg, anchor="ms")
    return p.sprite()


# ===========================================================================
# devices (+ the loop the lights chase around)
# ===========================================================================

@register("devices")
class Devices(Scene):
    weight = 1.0

    def setup(self):
        f, d = self.f, self.data
        CX, CY = f.CX, f.CY
        if f.portrait:
            self.lap = dict(c=(CX, CY - 330), sw=800, sh=470)
            self.tab = dict(c=(CX - 250, CY + 420), sw=330, sh=440)
            self.pho = dict(c=(CX + 270, CY + 440), sw=200, sh=410)
            tri = [(CX, CY - 820), (CX + 500, CY + 760), (CX - 500, CY + 760)]
            self.dk = min(1.0, (f.DW - 120) / 832)
        else:
            self.lap = dict(c=(CX, CY - 10), sw=800, sh=470)
            self.tab = dict(c=(CX - 700, CY + 40), sw=330, sh=440)
            self.pho = dict(c=(CX + 690, CY + 50), sw=200, sh=410)
            tri = [(CX, CY - 380), (CX + 820, CY + 340), (CX - 820, CY + 340)]
            self.dk = 1.0
        self.loop = Path(resample(chaikin(np.array(tri, float), 4, closed=True), 1200, closed=True), closed=True)
        self.loop_t0 = self.a + 1.05
        self.key = max(self.a + 0.6 * self.L, self.a + 1.2)
        self.label = d.get("chip", "Synced")
        self.lights = [self.f.light_color(j) for j in range(3)]

    def screen_rect(self, dev):
        cx, cy = dev["c"]
        return (cx - dev["sw"] / 2, cy - dev["sh"] / 2, dev["sw"], dev["sh"])

    def entry(self):
        return self.lap["c"]

    def exit(self):
        return self.lap["c"]

    def anchors(self):
        return {"key": self.key}

    def tint_color(self):
        return mix_hex(self.lights[0], self.lights[-1], 0.5)

    def loop_u(self, j, t):
        return (t - self.loop_t0) * 0.62 + j / 3.0

    def race_starts(self, n, t):
        out = []
        for i in range(n):
            u = self.loop_u(i % 3, t) + (i // 3) * 0.17
            out.append((self.loop.at(u), self.loop.tangent(u)))
        return out

    def cues(self):
        return [dict(t=self.a + 0.15, kind="swirl", gain=-16, dur=1.3),
                dict(t=self.a + 0.62, kind="tick", gain=-22, pitch=1.2),
                dict(t=self.a + 0.72, kind="tick", gain=-22, pitch=1.35),
                dict(t=self.a + 0.82, kind="tick", gain=-22, pitch=1.5),
                dict(t=self.key, kind="success", gain=-16),
                dict(t=self.f.race_t + 0.12, kind="digital", gain=-17)]

    def _content(self, which, w, h):
        img = self.data.get(which)
        if img:
            path = img if os.path.isabs(img) else os.path.join(self.f.dir, img)
            im = Image.open(path).convert("RGBA")
            crop = self.data.get(which + "_crop")
            if crop:
                im = im.crop(tuple(crop))
            # web pages fit the screen width; a short page is extended with its own bottom colour,
            # a tall one is cropped from the top. Never distort, never cut the sides.
            tw, th = int(w * self.ppd * 2), int(h * self.ppd * 2)
            k = tw / im.width
            im = im.resize((tw, max(1, int(im.height * k + 0.5))), Image.LANCZOS)
            if im.height < th:
                bottom = np.asarray(im)[-3:].reshape(-1, 4).mean(0).astype(np.uint8)
                canvas = Image.new("RGBA", (tw, th), tuple(int(v) for v in bottom))
                canvas.paste(im, (0, 0))
                im = canvas
            else:
                im = im.crop((0, 0, tw, th))
            from .gfx import Sprite
            return Sprite.from_pil(im, self.ppd * 2)
        return _mobile_page(self, w, h, self.data.get("title") or getattr(self.prev, "title", self.f.name))

    def draw(self, cv, t):
        f, T = self.f, self.T
        if t < self.a - 0.05:
            return
        out = 1 - smooth(ramp(t, f.race_t, f.race_t + 0.6))
        if out <= 0:
            return
        dk = self.dk
        # laptop content: the page flies in from the share scene
        lx, ly, lw, lh = self.screen_rect(self.lap)
        tgt = [(lx, ly), (lx + lw, ly), (lx + lw, ly + lh), (lx, ly + lh)]
        tgt = [(f.CX + (x - f.CX) * dk, self.lap["c"][1] + (y - self.lap["c"][1]) * dk) for x, y in tgt]
        g = ease_in_out(ramp(t, self.a, self.a + 0.7), 2.4)
        morph = isinstance(self.prev, Share)
        if not morph and t < self.a + 0.15:
            return
        if morph:
            src = self.prev.quad(self.a)
            page = self.prev.page()[0]
        else:
            src = [(lerp(f.CX, x, 1.08), lerp(self.lap["c"][1], y, 1.08)) for x, y in tgt]
            page = self.sp("lapc", lambda: self._content("screen", 800, 470))
        quad = [lerp(np.array(s), np.array(d), g) for s, d in zip(src, tgt)]
        cv.draw_quad(page, quad, alpha=out * (1.0 if morph else smooth(ramp(t, self.a + 0.15, self.a + 0.5))))
        # devices
        lid_a = ease_out(ramp(t, self.a + 0.15, self.a + 0.6)) * out
        lid, base, _ = self.sp("lap", lambda: ui.laptop(800, 470, self.ppd, self.fonts, T))
        lc = self.lap["c"]
        cv.draw(lid, lc[0], lc[1], scale=dk * (1.04 - 0.04 * lid_a), alpha=lid_a)
        cv.draw(base, lc[0], lc[1] + (235 + 16 + 8) * dk, scale=dk, alpha=lid_a, anchor=(0.5, 0))
        for dev, side, name, phone in ((self.tab, -1, "tab", False), (self.pho, 1, "pho", True)):
            s_in = ease_out(ramp(t, self.a + 0.25, self.a + 0.85), 3.0)
            dx = side * 420 * (1 - s_in)
            cx, cy = dev["c"]
            cont = self.sp(name + "c", lambda dev=dev, phone=phone: self._content(
                "mobile" if phone else "tablet", dev["sw"], dev["sh"]))
            cv.draw(cont, cx + dx, cy, scale=dk, alpha=s_in * out)
            fr, _ = self.sp(name, lambda dev=dev, phone=phone: ui.handheld(dev["sw"], dev["sh"], self.ppd, self.fonts,
                                                                           T, phone=phone))
            cv.draw(fr, cx + dx, cy, scale=dk, alpha=s_in * out)
        # loop: faint track + three chasing lights, passing behind the devices
        if t >= self.a + 0.55:
            tr_a = smooth(ramp(t, self.a + 0.8, self.a + 1.3)) * out
            if tr_a > 0:
                pts = self.loop.span(0, 1, 600)
                cv.light_poly(pts, np.array([0.6, 0.6, 0.6]), 0.12 * tr_a, width=2.2, occlude=0.9)
            for j, col in enumerate(self.lights):
                if t >= f.race_t:
                    break
                if t < self.loop_t0:
                    key = ("spin", j)
                    tgt_p = self.loop.at(self.loop_u(j, self.loop_t0))
                    path = f.sprite((self.i, key), lambda tp=tgt_p, j=j:
                                    Path.through(np.array(lc), tp, bend=0.35, side=1 if j % 2 else -1))
                    u = ease_out(ramp(t, self.a + 0.6, self.loop_t0), 1.6)
                    cv.light_path(path, u, tail=0.5, color=col, width=4.2, energy=1.0, occlude=0.9)
                else:
                    u = self.loop_u(j, t)
                    lp = Path(self.loop.pts[:-1], closed=True)
                    cv.light_path(lp, u, tail=0.16, color=col, width=4.5, energy=1.1, occlude=0.9)
        # "Synced" chip
        if t >= self.key - 0.05:
            ch = self.sp("chip", lambda: ui.chip(self.label, self.ppd, self.fonts, T, color=self.lights[0]))
            pop = spring_in(t - self.key, 4.2, 0.5)
            cy = lc[1] + (235 + 70) * dk + 40
            cv.draw(ch, lc[0], cy, scale=max(0.01, pop), alpha=out)


def _mobile_page(sc, w, h, title):
    from .ui import Theme
    L = Theme()  # light product UI, whatever the stage
    p = ui.Painter(w, h, sc.ppd, sc.fonts)
    p.rrect(0, 0, w, h, 4, fill="#FFFFFF")
    p.rrect(0, 0, w, 44, 0, fill=L.paper_frame)
    ui.draw_kind_icon(p, "doc", 12, 12, 20, L)
    p.rrect(40, 18, w * 0.4, 8, 4, fill=L.track)
    p.text(16, 82, title, max(14, w * 0.075), 500, L.text)
    r = rng(int(w))
    y = 104
    while y < h - 20:
        p.rrect(16, y, (w - 32) * r.uniform(0.5, 0.95), 6, 3, fill="#E3E6EA")
        y += 20
    return p.sprite()


# ===========================================================================
# flow
# ===========================================================================

@register("flow")
class Flow(Scene):
    weight = 1.5

    def setup(self):
        f, d = self.f, self.data
        nodes = d.get("nodes") or [dict(label="Trigger", kind="link"), dict(label="Enrich", kind="sheet"),
                                   dict(label="Approve", kind="task"), dict(label="Notify", kind="chat")]
        self.nodes = nodes[:5]
        n = len(self.nodes)
        self.nw, self.nh = 260.0, 96.0
        if f.portrait:
            k = min(1.0, (f.DW - 160) / self.nw)
            ys = np.linspace(f.CY - 560, f.CY + 560, n)
            xs = [f.CX + (180 if i % 2 else -180) for i in range(n)]
        else:
            span = min(f.DW - 2 * 200, 1560)
            xs = np.linspace(f.CX - span / 2, f.CX + span / 2, n)
            ys = [f.CY + (-80 if i % 2 == 0 else 80) for i in range(n)]
            k = 1.0
        self.k = k
        self.pos = [np.array([x, y]) for x, y in zip(xs, ys)]
        self.t_nodes = [self.a + 0.32 + 0.08 * i for i in range(n)]
        t_run0 = self.land + 0.15
        self.key = max(self.a + 0.82 * self.L, t_run0 + 0.4 * n)
        seg = (self.key - t_run0) / max(1, n - 1)
        self.t_arrive = [t_run0 + i * seg for i in range(n)]
        self.links = []
        for i in range(n - 1):
            p0 = self.port(i, out=True)
            p3 = self.port(i + 1, out=False)
            if f.portrait:
                c = 140
                self.links.append(Path.beziers([(p0, p0 + (0, c), p3 - (0, c), p3)]))
            else:
                c = abs(p3[0] - p0[0]) * 0.5
                self.links.append(Path.beziers([(p0, p0 + (c, 0), p3 - (c, 0), p3)]))
        self.result = d.get("result", "Done")

    def port(self, i, out):
        p = self.pos[i]
        if self.f.portrait:
            return p + (0, (self.nh / 2 + 6) * (1 if out else -1)) * np.array([0, self.k])
        return p + ((self.nw / 2 + 6) * (1 if out else -1), 0)

    def entry(self):
        return tuple(self.port(0, out=False))

    def exit(self):
        return tuple(self.port(len(self.nodes) - 1, out=True))

    def anchors(self):
        return {"key": self.key, "start": self.t_arrive[0]}

    def cues(self):
        c = [dict(t=t0, kind="tick", gain=-25, pitch=1 + 0.08 * i) for i, t0 in enumerate(self.t_nodes)]
        for i, ta in enumerate(self.t_arrive):
            c.append(dict(t=ta, kind="pop", gain=-17, pitch=0.9 + 0.12 * i, pan=-0.5 + i / max(1, len(self.t_arrive) - 1)))
            if i:
                c.append(dict(t=ta - 0.25, kind="whoosh_short", gain=-24))
        c.append(dict(t=self.key + 0.1, kind="success", gain=-15))
        return c

    def draw(self, cv, t):
        f, T, k = self.f, self.T, self.k
        out = 1 - smooth(ramp(t, self.b - 0.15, self.b + 0.2))
        if out <= 0 or t < self.a - 0.3:
            return
        for i, nd in enumerate(self.nodes):
            a_in = ease_out(ramp(t, self.t_nodes[i] - 0.1, self.t_nodes[i] + 0.3), 2.5)
            if a_in <= 0:
                continue
            sp = self.sp(("node", i), lambda nd=nd: ui.node_card(nd["label"], nd.get("kind", "doc"), self.ppd,
                                                                 self.fonts, T))
            hit = pulse(t, self.t_arrive[i], 0.02, 0.18)
            cv.draw(sp, self.pos[i][0], self.pos[i][1] + 18 * (1 - a_in), scale=k * (1 + 0.04 * hit), alpha=a_in * out)
            if t >= self.t_arrive[i]:
                chk = self.sp(("chk",), lambda: _check_sprite(self, self.color))
                pop = spring_in(t - self.t_arrive[i], 5, 0.45)
                cv.draw(chk, self.pos[i][0] + (self.nw / 2 - 10) * k, self.pos[i][1] - (self.nh / 2 - 10) * k,
                        scale=k * 0.8 * max(0.01, pop), alpha=out)
        # the light runs node to node, leaving a lit connector behind
        for i, path in enumerate(self.links):
            t0, t1 = self.t_arrive[i], self.t_arrive[i + 1]
            if t < t0:
                break
            u = ease_in_out(ramp(t, t0 + 0.04, t1), 1.8)
            cv.light_poly(path.span(0, u, 60), lin(self.color), 0.42 * out, width=3.0)
            if t <= t1 + 0.1:
                cv.light_path(path, u, tail=0.5, color=self.color, width=4.5, energy=1.1 * out)
        if t >= self.key:
            ch = self.sp("res", lambda: ui.chip(self.result, self.ppd, self.fonts, T, color=self.color))
            pop = spring_in(t - self.key - 0.08, 4.2, 0.5)
            last = self.pos[-1]
            y = last[1] + (self.nh / 2 + 60) * k
            cv.draw(ch, last[0], y, scale=max(0.01, pop), alpha=out)


# ===========================================================================
# chart
# ===========================================================================

@register("chart")
class Chart(Scene):
    weight = 1.35

    def setup(self):
        f, d = self.f, self.data
        self.fit = k = min(1.0, (f.DW - 100) / 1100)
        self.cw, self.ch = 1100 * k, 600 * k
        self.C = np.array([f.CX, f.CY + 10])
        vals = d.get("series") or [12, 18, 15, 24, 22, 31, 29, 38, 44, 52]
        self.value = d.get("value", "+38%")
        self.label = d.get("label", "Revenue this quarter")
        self.title = d.get("title", "")
        x0 = self.C[0] - self.cw / 2 + 36 * k
        x1 = self.C[0] + self.cw / 2 - 36 * k
        y0 = self.C[1] - self.ch / 2 + 190 * k
        y1 = self.C[1] + self.ch / 2 - 50 * k
        v = np.array(vals, float)
        v = (v - v.min()) / max(1e-6, v.max() - v.min())
        xs = np.linspace(x0, x1, len(v))
        ys = y1 - v * (y1 - y0)
        pts = np.stack([xs, ys], 1)
        self.curve = Path(_catmull(pts, 24))
        self.area_box = (x0, y0, x1, y1)
        self.t_draw0 = self.land + 0.1
        self.key = max(self.a + 0.75 * self.L, self.t_draw0 + 1.1)

    def entry(self):
        return tuple(self.curve.at(0))

    def exit(self):
        return tuple(self.curve.at(1))

    def anchors(self):
        return {"key": self.key}

    def cues(self):
        return [dict(t=self.t_draw0, kind="riser_short", gain=-20, dur=self.key - self.t_draw0),
                dict(t=self.key, kind="pop_low", gain=-13), dict(t=self.key + 0.02, kind="success", gain=-17)]

    def draw(self, cv, t):
        f, T, k = self.f, self.T, self.fit
        out = 1 - smooth(ramp(t, self.b - 0.15, self.b + 0.2))
        a_in = ease_out(ramp(t, self.t_in, self.t_in + 0.4))
        al = a_in * out
        if al <= 0:
            return
        card, _ = self.sp("card", lambda: ui.chart_card(1100, 600, self.ppd, self.fonts, T, self.title))
        cv.draw(card, self.C[0], self.C[1] + 20 * (1 - a_in), scale=k, alpha=al)
        u = ease_in_out(ramp(t, self.t_draw0, self.key), 1.7)
        x0, y0, x1, y1 = self.area_box
        if u > 0:
            area = self.sp("area", lambda: _area_sprite(self))
            head = self.curve.at(u)
            cv.draw(area, x0, y0 - 40 * k, anchor=(0, 0), alpha=al * 0.9, wipe=(head[0], 30))
            cv.light_poly(self.curve.span(0, u, 120), lin(self.color), 0.8 * out, width=5.0 * k)
            cv.light_dot(head, lin(self.color), energy=1.6 * out, radius=6 * k)
        # big number rolls to its value
        num_t = ease_out(ramp(t, self.t_draw0, self.key), 3.0)
        txt = _roll(self.value, num_t)
        sp = self.text(txt, 84, 700)
        bx, by = self.C[0] - self.cw / 2 + 50 * k, self.C[1] - self.ch / 2 + 110 * k
        cv.draw(sp, bx, by, scale=k * (1 + 0.05 * pulse(t, self.key, 0.02, 0.2)), alpha=al,
                anchor=(0, sp.baseline / sp.h))
        lb = self.text(self.label, 26, 400, f.text2)
        cv.draw(lb, bx + 4, by + 44 * k, scale=k, alpha=al, anchor=(0, lb.baseline / lb.h))


def _catmull(pts, per=20):
    out = []
    P = np.vstack([pts[:1], pts, pts[-1:]])
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        for s in np.linspace(0, 1, per, endpoint=False):
            s2, s3 = s * s, s * s * s
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * s + (2 * p0 - 5 * p1 + 4 * p2 - p3) * s2 +
                              (-p0 + 3 * p1 - 3 * p2 + p3) * s3))
    out.append(pts[-1])
    return np.array(out)


def _area_sprite(sc):
    x0, y0, x1, y1 = sc.area_box
    top = y0 - 40 * sc.fit
    w, h = x1 - x0, y1 - top
    p = ui.Painter(w, h, sc.ppd, sc.fonts)
    pts = sc.curve.span(0, 1, 200)
    poly = [(x - x0, y - top) for x, y in pts] + [(w, h), (0, h)]
    p.poly(poly, sc.color, alpha=70)
    arr = np.asarray(p.im).copy()
    g = np.linspace(1.0, 0.1, arr.shape[0])[:, None]
    arr[..., 3] = (arr[..., 3] * g).astype(np.uint8)
    p.im = Image.fromarray(arr)
    return p.sprite()


def _roll(value, k):
    """Roll the digits inside a value string ('+38%', '$1.2M', '4,096')."""
    import re
    m = re.search(r"[\d.,]+", value)
    if not m:
        return value
    raw = m.group(0)
    num = float(raw.replace(",", ""))
    cur = num * k
    dec = len(raw.split(".")[1]) if "." in raw else 0
    s = f"{cur:,.{dec}f}" if "," in raw else f"{cur:.{dec}f}"
    return value[:m.start()] + s + value[m.end():]


# ===========================================================================
# chat (AI)
# ===========================================================================

@register("chat")
class Chat(Scene):
    weight = 1.7

    def setup(self):
        f, d = self.f, self.data
        self.fit = k = min(1.0, (f.DW - 80) / 1000)
        self.prompt = d.get("prompt", "Summarise this week's feedback")
        self.answer = d.get("answer") or ["Three themes stand out:", "faster onboarding,", "clearer pricing,",
                                          "and offline mode."]
        self.pw, self.ph = 1000 * k, 92 * k
        self.P = np.array([f.CX, f.CY + (300 if not f.portrait else 520) * k])
        self.t_type = self.land + 0.2
        self.cps = 0.045
        self.t_sent = self.t_type + self.cps * len(self.prompt) + 0.15
        self.t_ans = self.t_sent + 0.45
        self.lps = 0.32  # seconds per answer line
        self.key = max(self.a + 0.8 * self.L, self.t_ans + self.lps * len(self.answer))
        self.lps = (self.key - self.t_ans) / len(self.answer)
        self.line_y = self.P[1] + self.ph / 2 + 7 * k
        self.lx0, self.lx1 = self.P[0] - self.pw / 2 + 46 * k, self.P[0] + self.pw / 2 - 46 * k
        self.ax = self.P[0] - self.pw / 2 + 40 * k
        self.ay0 = f.CY - (170 if not f.portrait else 260) * k

    def entry(self):
        return (self.lx0, self.line_y)

    def exit(self):
        n = len(self.answer)
        sp = self.text(self.answer[-1], 34, 400)
        return (self.ax + 60 * self.fit + sp.w * self.fit, self.ay0 + (n - 1) * 50 * self.fit + 70 * self.fit)

    def anchors(self):
        return {"key": self.key, "send": self.t_sent}

    def cues(self):
        c = [dict(t=self.t_type + i * self.cps, kind="key", gain=-24, pitch=1 + 0.2 * ((i * 0.37) % 1))
             for i, ch in enumerate(self.prompt) if ch != " " and i % 2 == 0]
        c.append(dict(t=self.t_sent, kind="send", gain=-16))
        c += [dict(t=self.t_ans + i * self.lps, kind="shimmer", gain=-23, pitch=1 + 0.1 * i) for i in range(len(self.answer))]
        c.append(dict(t=self.key, kind="success", gain=-16))
        return c

    def draw(self, cv, t):
        f, T, k = self.f, self.T, self.fit
        out = 1 - smooth(ramp(t, self.b - 0.15, self.b + 0.2))
        a_in = ease_out(ramp(t, self.t_in, self.t_in + 0.4), 2.5)
        if a_in * out <= 0:
            return
        pill = self.sp("pill", lambda: ui.pill(1000, 92, self.ppd, self.fonts, T.field))
        cv.draw(pill, self.P[0], self.P[1], scale=k * (0.94 + 0.06 * a_in), alpha=a_in * out)
        tx, ty = self.P[0] - self.pw / 2 + 46 * k, self.P[1] + 10 * k
        n = int(clamp((t - self.t_type) / self.cps, 0, len(self.prompt))) if t >= self.t_type else 0
        sent = smooth(ramp(t, self.t_sent, self.t_sent + 0.2))
        q = self.text(self.prompt, 30, 400)
        if n and sent < 1:
            cv.draw(q, tx, ty, scale=k, alpha=out * (1 - sent), anchor=(0, q.baseline / q.h),
                    wipe=(tx + q.meta["advance"][n] * k + 1, 1.0))
        if t < self.t_type + 0.1:
            ph = self.text(self.data.get("placeholder", f"Ask {f.name}"), 28, 400, f.text2)
            cv.draw(ph, tx, ty, scale=k, alpha=a_in * out * (1 - ramp(t, self.t_type - 0.05, self.t_type + 0.1)),
                    anchor=(0, ph.baseline / ph.h))
        if t >= self.land - 0.02:
            g = ease_out(ramp(t, self.land, self.land + 0.35), 2.0)
            xs = np.linspace(self.lx0, lerp(self.lx0, self.lx1, g), 40)
            cv.light_poly(np.stack([xs, np.full(40, self.line_y)], 1), lin(self.color), 0.55 * out, width=4.0 * k)
        # sent prompt becomes a user bubble
        if sent > 0:
            bw = q.w * k + 56 * k
            bub = self.sp(("ub", round(bw)), lambda: ui.chat_bubble(q.w + 56, 70, self.ppd, self.fonts, T, user=True))
            bx = self.P[0] + self.pw / 2 - bw / 2
            by = self.ay0 - 110 * k + 30 * (1 - sent)
            cv.draw(bub, bx, by, scale=k, alpha=sent * out)
            cv.draw(q, bx - bw / 2 + 28 * k, by + 10 * k, scale=k, alpha=sent * out, anchor=(0, q.baseline / q.h))
        # answer streams in; the light is the streaming caret
        if t >= self.t_ans - 0.2:
            spark = self.sp("spark", lambda: _sparkle_sprite(self, self.color))
            cv.draw(spark, self.ax + 14 * k, self.ay0 - 12 * k, scale=k * (1 + 0.08 * math.sin(t * 6)),
                    rot=t * 30, alpha=out * smooth(ramp(t, self.t_ans - 0.2, self.t_ans)))
            for i, line in enumerate(self.answer):
                t0 = self.t_ans + i * self.lps
                if t < t0:
                    break
                sp = self.text(line, 34, 400)
                prog = clamp((t - t0) / self.lps)
                nch = int(prog * len(line) + 0.999)
                edge = self.ax + 60 * k + sp.meta["advance"][min(nch, len(line))] * k
                y = self.ay0 + i * 50 * k + 12 * k
                cv.draw(sp, self.ax + 60 * k, y, scale=k, alpha=out, anchor=(0, sp.baseline / sp.h), wipe=(edge, 14))
                if prog < 1 or (i == len(self.answer) - 1 and t < self.b):
                    ly = y + 10 * k
                    cv.light_poly([(max(self.ax + 60 * k, edge - 90 * k), ly), (edge, ly)], lin(self.color),
                                  [0.0, 1.0 * out], width=3.5 * k)
                    cv.light_dot((edge, ly), lin(self.color), energy=1.3 * out, radius=4 * k)


def _sparkle_sprite(sc, color):
    p = ui.Painter(40, 40, sc.ppd, sc.fonts)
    c = 20
    pts = []
    for i in range(8):
        r = 18 if i % 2 == 0 else 5
        a = i * math.pi / 4 - math.pi / 2
        pts.append((c + r * math.cos(a), c + r * math.sin(a)))
    p.poly(pts, color)
    return p.sprite()


# ===========================================================================
# screen (a real screenshot)
# ===========================================================================

@register("screen")
class Screen(Scene):
    """A real screenshot of the product in a clean browser window, shot like a keynote:
    3D tilt-in, a glass sheen, then the camera pushes in on each highlight while the rest of
    the UI dims, the light traces the detail and a label pops beside it.

    data: {"image": "assets/screens/x.png", "url": "app.com",
           "highlights": [{"rect": [x, y, w, h] (screenshot px), "label": "...",
                           "zoom": 1.35, "click": false}],
           "tilt": 14 (degrees of yaw on entry, sign = side), "enter": "tilt" | "rise" | "slide",
           "frame": "browser" | "bare", "sharpen": true}
    """
    weight = 1.8

    def setup(self):
        f, d = self.f, self.data
        img = d.get("image")
        if not img:
            raise ValueError("screen scene needs data.image (a screenshot path)")
        self.img_path = img if os.path.isabs(img) else os.path.join(f.dir, img)
        shot = Image.open(self.img_path).convert("RGB")
        if d.get("crop"):
            shot = shot.crop(tuple(d["crop"]))
        if d.get("sharpen", True):
            from PIL import ImageFilter
            shot = shot.filter(ImageFilter.UnsharpMask(radius=1.2, percent=60, threshold=2))
        self.shot = shot
        self.bar = 52.0 if d.get("frame", "browser") == "browser" else 0.0
        self.fw = min(1300.0, f.DW - 160) if not f.portrait else f.DW - 80
        self.scale_px = (self.fw - 2.4) / shot.width
        self.fh = shot.height * self.scale_px + self.bar
        if self.fh > f.DH - 170:
            self.fw *= (f.DH - 170) / self.fh
            self.scale_px = (self.fw - 2.4) / shot.width
            self.fh = shot.height * self.scale_px + self.bar
        self.C = np.array([f.CX, f.CY + 6])
        self.hl = d.get("highlights") or []
        n = max(1, len(self.hl))
        t0 = self.land + 0.35
        self.key = max(self.a + 0.86 * self.L, t0 + 0.9 * n)
        self.t_hl = [t0 + i * (self.key - t0) / n for i in range(n)]
        self.tilt = float(d.get("tilt", 14 if self.i % 2 == 0 else -14))
        self.enter = d.get("enter", "tilt")
        self.focal = 2600.0

    # -- geometry -------------------------------------------------------------
    def frame(self):
        if self.bar:
            return self.sp("frame", lambda: ui.browser_frame(self.shot, self.ppd * 1.4, self.fonts, self.T,
                                                             w=self.fw, url=self.data.get("url", "")))
        def bare():
            p = ui.Painter(self.fw, self.fh, self.ppd * 1.4, self.fonts)
            k = p.k
            tgt = self.shot.convert("RGBA").resize((int(self.fw * k), int(self.fh * k)), Image.LANCZOS)
            mask = Image.new("L", tgt.size, 0)
            from PIL import ImageDraw
            ImageDraw.Draw(mask).rounded_rectangle([0, 0, tgt.size[0] - 1, tgt.size[1] - 1], radius=18 * k, fill=255)
            p.im.paste(tgt, (0, 0), mask)
            return p.sprite(), {}
        return self.sp("frame", bare)

    def local_rect(self, i):
        """Highlight rect in frame-local design px (origin = frame centre)."""
        x, y, w, h = self.hl[i]["rect"]
        if self.data.get("crop"):
            cx0, cy0 = self.data["crop"][:2]
            x, y = x - cx0, y - cy0
        s = self.scale_px
        ox, oy = -self.fw / 2 + 1.2, -self.fh / 2 + self.bar
        return (ox + x * s, oy + y * s, w * s, h * s)

    def pose(self, t):
        """(yaw, pitch, zoom, focus) of the shot at t. focus is a frame-local point kept near centre."""
        e = ease_out(ramp(t, self.a + 0.05, self.a + 0.85), 2.6)
        yaw = math.radians(self.tilt * (1 - e) - 3.0 * (self.tilt > 0) * e + 3.0 * (self.tilt < 0) * e) \
            if self.enter == "tilt" else 0.0
        yaw += math.radians(1.5 * math.sin((t - self.a) * 0.6))
        pitch = math.radians(4.0 * (1 - e) + 1.2) if self.enter == "tilt" else 0.0
        z, focus = 1.0, np.zeros(2)
        if self.hl:
            keys = []
            for i, th in enumerate(self.t_hl):
                x, y, w, h = self.local_rect(i)
                zz = float(self.hl[i].get("zoom", 1.3))
                keys.append((th - 0.35, zz, np.array([x + w / 2, y + h / 2])))
            z_prev, f_prev = 1.0, np.zeros(2)
            for (tk, zz, fc) in keys:
                k = ease_in_out(ramp(t, tk, tk + 0.6), 2.2)
                z_prev = lerp(z_prev, zz, k)
                f_prev = lerp(f_prev, fc, k)
            # release before the hand-over so the next scene starts wide
            k = ease_in_out(ramp(t, self.b - 0.5, self.b + 0.05), 2.0)
            z, focus = lerp(z_prev, 1.0, k), lerp(f_prev, np.zeros(2), k)
        if self.enter == "rise":
            focus = focus + np.array([0.0, -80.0 * (1 - e)])
        elif self.enter == "slide":
            focus = focus + np.array([-260.0 * (1 - e) * (1 if self.tilt > 0 else -1), 0.0])
        return yaw, pitch, z, focus

    def M(self, p, t, pose=None):
        yaw, pitch, z, focus = pose or self.pose(t)
        # keep the focus point near the centre, more strongly the further we zoom
        X, Y = (p[0] - focus[0] * min(1.0, (z - 1) * 3)) * z, (p[1] - focus[1] * min(1.0, (z - 1) * 3)) * z
        X1 = X * math.cos(yaw)
        Z1 = -X * math.sin(yaw)
        Y2 = Y * math.cos(pitch) - Z1 * math.sin(pitch)
        Z2 = Y * math.sin(pitch) + Z1 * math.cos(pitch)
        q = self.focal / (self.focal + Z2)
        return np.array([self.C[0] + X1 * q, self.C[1] + Y2 * q])

    def quad(self, t, pose=None):
        w, h = self.fw / 2, self.fh / 2
        return [self.M(c, t, pose) for c in ((-w, -h), (w, -h), (w, h), (-w, h))]

    # -- interface --------------------------------------------------------------
    def entry(self):
        if self.hl:
            x, y, w, h = self.local_rect(0)
            return tuple(self.M((x, y + h / 2), self.land))
        return tuple(self.C)

    def exit(self):
        if self.hl:
            x, y, w, h = self.local_rect(len(self.hl) - 1)
            return tuple(self.M((x + w, y + h / 2), self.b))
        return tuple(self.C)

    def anchors(self):
        a = {"key": self.key}
        for i, th in enumerate(self.t_hl):
            a[f"h{i + 1}"] = th
        return a

    def cues(self):
        c = [dict(t=self.a + 0.05, kind="swoosh", gain=-17, dur=0.8), dict(t=self.a + 0.45, kind="shimmer", gain=-24)]
        for i, (h, th) in enumerate(zip(self.hl, self.t_hl)):
            c.append(dict(t=th - 0.35, kind="whoosh_soft", gain=-23, pan=-0.3 + 0.3 * i))
            c.append(dict(t=th + 0.5, kind="tick", gain=-20, pitch=1.1 + 0.1 * i))
            if h.get("click"):
                c += [dict(t=th + 0.1, kind="click", gain=-15), dict(t=th + 0.12, kind="pop_soft", gain=-22)]
        return c

    def draw(self, cv, t):
        f, T = self.f, self.T
        out = 1 - smooth(ramp(t, self.b - 0.15, self.b + 0.2))
        a_in = smooth(ramp(t, self.a + 0.05, self.a + 0.45))
        al = a_in * out
        if al <= 0:
            return
        pose = self.pose(t)
        sp, _ = self.frame()
        q = self.quad(t, pose)
        cv.draw_quad(sp, q, alpha=al)
        # glass sheen on entry
        sk = ramp(t, self.a + 0.25, self.a + 1.05)
        if 0 < sk < 1:
            gl = self.sp("glass", lambda: ui.card(self.fw, self.fh, 18, self.ppd * 0.5, self.fonts, T, fill="#FFFFFF",
                                                  border=False))
            cx = lerp(min(p[0] for p in q) - 300, max(p[0] for p in q) + 300, ease_in_out(sk))
            cv.draw(gl, self.C[0], self.C[1], sx=(max(p[0] for p in q) - min(p[0] for p in q)) / self.fw,
                    sy=(max(p[1] for p in q) - min(p[1] for p in q)) / self.fh, alpha=0.32 * al * math.sin(math.pi * sk),
                    band=(cx, 110, 22))
        for i, h in enumerate(self.hl):
            th = self.t_hl[i]
            if t < th - 0.3:
                break
            x, y, w, hh = self.local_rect(i)
            nxt = self.t_hl[i + 1] - 0.35 if i + 1 < len(self.t_hl) else self.b - 0.45
            act = smooth(ramp(t, th - 0.3, th + 0.1)) * (1 - smooth(ramp(t, nxt, nxt + 0.3)))
            if act <= 0.002:
                continue
            # spotlight: dim the UI around the detail
            key = ("spot", i)
            spot = self.sp(key, lambda x=x, y=y, w=w, hh=hh: _spot_sprite(self, x, y, w, hh))
            cv.draw_quad(spot, q, alpha=0.62 * act * al)
            pts = np.array([self.M(p, t, pose) for p in rrect_points(x - 8, y - 8, w + 16, hh + 16, 12, 220)])
            u = ease_out(ramp(t, th, th + 0.6), 2.2)
            cv.light_path(Path(pts), u, tail=0.5, color=self.color, width=4.0, energy=1.05 * act * out)
            hold = smooth(ramp(t, th + 0.3, th + 0.75))  # the resting outline fades in under the trace
            if hold > 0:
                cv.light_poly(pts, lin(self.color), 0.45 * act * out * hold, width=3.2)
            if h.get("click"):
                rr = ease_out(ramp(t, th + 0.1, th + 0.55), 2.0)
                if 0 < rr < 1:
                    c = self.M((x + w * 0.5, y + hh * 0.5), t, pose)
                    ring = [(c[0] + 64 * rr * math.cos(a_), c[1] + 64 * rr * math.sin(a_)) for a_ in np.linspace(0, 6.3, 64)]
                    cv.light_poly(ring, lin(self.color), 1.0 * (1 - rr) * out, width=4.0)
            lab = h.get("label")
            if lab:
                la = smooth(ramp(t, th + 0.35, th + 0.7)) * act * out
                ch = self.sp(("lab", i), lambda lab=lab: ui.chip(lab, self.ppd, self.fonts, T, icon="dot",
                                                                color=self.color, size=24, h=52))
                below = self.M((x + w / 2, y + hh + 16), t, pose)
                p = below + (0, 40)
                if p[1] > f.DH - 70:
                    p = self.M((x + w / 2, y - 16), t, pose) - (0, 40)
                cv.draw(ch, p[0], p[1] + 10 * (1 - la), alpha=la, scale=0.96 + 0.04 * la)


def _spot_sprite(sc, x, y, w, h):
    """Frame-sized veil with a soft rounded hole around one highlight."""
    T, f = sc.T, sc.f
    veil = "#FFFFFF" if f.stage_mode == "light" else "#000000"
    p = ui.Painter(sc.fw, sc.fh, sc.ppd * 0.6, sc.fonts)
    p.rrect(0, 0, sc.fw, sc.fh, 18, fill=veil, alpha=200)
    hx, hy = x + sc.fw / 2 - 14, y + sc.fh / 2 - 14
    p.hole(hx, hy, w + 28, h + 28, 16)
    from PIL import ImageFilter
    p.im = p.im.filter(ImageFilter.GaussianBlur(10 * p.k))
    return p.sprite()


# ===========================================================================
# statement (kinetic type)
# ===========================================================================

@register("statement")
class Statement(Scene):
    """Big words, written by light. data: {"lines": [...], "highlight": "word"}"""
    weight = 1.1

    def setup(self):
        f, d = self.f, self.data
        self.lines = d.get("lines") or [d.get("text", f.brief.get("tagline", f.name))]
        self.size = float(d.get("size", 118 if not f.portrait else 92))
        self.weight_ = int(d.get("weight", 600))
        self.hl = (d.get("highlight") or "").lower()
        wmax = max(f.type.width(l, self.size, self.weight_) for l in self.lines)
        if wmax > f.DW * 0.86:
            self.size *= f.DW * 0.86 / wmax
        lh = self.size * 1.12
        y0 = f.CY - lh * (len(self.lines) - 1) / 2 + self.size * 0.34
        self.words = []
        t = self.land + 0.05
        for li, line in enumerate(self.lines):
            lw = f.type.width(line, self.size, self.weight_)
            x = f.CX - lw / 2
            y = y0 + li * lh
            for w in line.split(" "):
                ww = f.type.width(w, self.size, self.weight_)
                self.words.append(dict(w=w, x=x, y=y, t=t, line=li, width=ww))
                x += f.type.width(w + " ", self.size, self.weight_)
                t += 0.085
            t += 0.12
        self.t_written = t
        self.key = max(self.a + 0.62 * self.L, self.t_written + 0.15)
        self.hw = next((w for w in self.words if w["w"].lower().strip(".,!?") == self.hl.strip(".,!?")), None) \
            if self.hl else None

    def line_path(self, li):
        ws = [w for w in self.words if w["line"] == li]
        y = ws[0]["y"] + self.size * 0.2
        return Path([(ws[0]["x"] - 20, y), (ws[-1]["x"] + ws[-1]["width"] + 20, y)])

    def entry(self):
        return tuple(self.line_path(0).at(0))

    def exit(self):
        if self.hw:
            return (self.hw["x"] + self.hw["width"] + 30, self.hw["y"] - self.size * 0.35)
        return tuple(self.line_path(len(self.lines) - 1).at(1))

    def anchors(self):
        return {"key": self.key, "written": self.t_written}

    def cues(self):
        c = [dict(t=w["t"], kind="tick", gain=-26, pitch=0.8 + 0.05 * i) for i, w in enumerate(self.words)]
        if self.hw:
            c.append(dict(t=self.key, kind="whoosh_short", gain=-19))
            c.append(dict(t=self.key + 0.45, kind="confirm", gain=-21))
        return c

    def draw(self, cv, t):
        f = self.f
        out = 1 - smooth(ramp(t, self.b - 0.15, self.b + 0.2))
        if out <= 0 or t < self.land - 0.1:
            return
        hk = smooth(ramp(t, self.key, self.key + 0.3)) if self.hw else 0
        for w in self.words:
            a = ease_out(ramp(t, w["t"], w["t"] + 0.32), 2.6)
            if a <= 0:
                continue
            sp = self.text(w["w"], self.size, self.weight_, f.text_color)
            is_hl = self.hw is w
            cv.draw(sp, w["x"], w["y"] + 26 * (1 - a), alpha=a * out * (1 - (hk if is_hl else 0)), blur=5 * (1 - a),
                    anchor=(0, sp.baseline / sp.h))
            if is_hl and hk > 0:
                sp2 = self.text(w["w"], self.size, self.weight_, self.color)
                cv.draw(sp2, w["x"], w["y"], alpha=hk * out, anchor=(0, sp2.baseline / sp2.h))
        for li in range(len(self.lines)):
            ws = [w for w in self.words if w["line"] == li]
            t0, t1 = ws[0]["t"] - 0.05, ws[-1]["t"] + 0.3
            if t < t0:
                continue
            u = ease_in_out(ramp(t, t0, t1), 1.8) + 0.6 * ramp(t, t1, t1 + 0.5)
            cv.light_path(self.line_path(li), u, tail=0.55, color=self.color, width=4.0, energy=1.0 * out)
        if self.hw and t >= self.key:
            w = self.hw
            cx, cy = w["x"] + w["width"] / 2, w["y"] - self.size * 0.33
            rx, ry = w["width"] / 2 + 34, self.size * 0.62
            pts = self.f.sprite((self.i, "loop"), lambda: np.array(
                [(cx + rx * (1 + 0.04 * math.sin(3 * a_)) * math.cos(a_ - 2.4),
                  cy + ry * (1 + 0.05 * math.cos(2 * a_)) * math.sin(a_ - 2.4) + 6 * a_ / 7)
                 for a_ in np.linspace(0, 2 * math.pi * 1.08, 300)]))
            u = ease_out(ramp(t, self.key, self.key + 0.5), 2.0)
            cv.light_path(Path(pts), u, tail=0.9, color=self.color, width=5.0, energy=1.1 * out, head_glow=0.8)


# ===========================================================================
# features (bento grid)
# ===========================================================================

@register("features")
class Features(Scene):
    """data: {"items": [{"title": "...", "desc": "...", "kind": "link"}, ...]} (3-6 items)"""
    weight = 1.6

    def setup(self):
        f, d = self.f, self.data
        items = d.get("items") or [dict(title="Real-time sync", kind="link"), dict(title="Offline mode", kind="task"),
                                   dict(title="Version history", kind="doc"),
                                   dict(title="End-to-end encrypted", kind="code")]
        self.items = items[:6]
        n = len(self.items)
        gap = 24.0
        if f.portrait:
            rows = [[2]] + [[1, 1] for _ in range(n // 2 + 1)]
            cols, cw, ch = 2, 470.0, 250.0
        else:
            layouts = {3: [[1, 1, 1]], 4: [[2, 1], [1, 2]], 5: [[2, 1], [1, 1, 1]], 6: [[1, 1, 1], [1, 1, 1]]}
            rows = layouts.get(n, [[1] * min(3, n)])
            cols, cw, ch = 3, 440.0, 230.0 if n > 3 else 300.0
        rects, k = [], 0
        used = []
        for row in rows:
            if k >= n:
                break
            used.append(row[: n - k])
            k += len(row[: n - k])
        total_h = len(used) * ch + (len(used) - 1) * gap
        y = f.CY - total_h / 2
        for row in used:
            x = f.CX - (cols * cw + (cols - 1) * gap) / 2
            for span in row:
                w = span * cw + (span - 1) * gap
                rects.append((x, y, w, ch))
                x += w + gap
            y += ch + gap
        self.rects = rects
        self.n = len(rects)
        t0 = self.land
        self.key = max(self.a + 0.8 * self.L, t0 + 0.5 * self.n)
        self.seg = (self.key - t0) / self.n
        self.t_card = [t0 + i * self.seg for i in range(self.n)]

    def entry(self):
        x, y, w, h = self.rects[0]
        return (x, y + h / 2)

    def exit(self):
        x, y, w, h = self.rects[-1]
        return (x, y + h / 2)

    def anchors(self):
        return {"key": self.key}

    def cues(self):
        c = []
        for i, tc in enumerate(self.t_card):
            c.append(dict(t=tc, kind="whoosh_short", gain=-25, pan=-0.5 + i / max(1, self.n - 1)))
            c.append(dict(t=tc + self.seg * 0.8, kind="pop", gain=-19, pitch=0.9 + 0.08 * i))
        c.append(dict(t=self.key + 0.15, kind="shimmer", gain=-20))
        return c

    def draw(self, cv, t):
        f, T = self.f, self.T
        out = 1 - smooth(ramp(t, self.b - 0.15, self.b + 0.2))
        if out <= 0 or t < self.land - 0.05:
            return
        for i, (it, (x, y, w, h)) in enumerate(zip(self.items, self.rects)):
            tc = self.t_card[i]
            if t >= tc + self.seg * 0.75:
                pop = spring_in(t - (tc + self.seg * 0.75), 4.2, 0.55)
                sp = self.sp(("card", i), lambda it=it, w=w, h=h: _feature_card(self, it, w, h))
                cv.draw(sp, x + w / 2, y + h / 2, scale=0.94 + 0.06 * pop, alpha=min(1, pop * 1.4) * out)
            if t >= tc:
                pts = self.sp(("rr", i), lambda x=x, y=y, w=w, h=h: rrect_points(x, y, w, h, 26, 300))
                u = ease_in_out(ramp(t, tc, tc + self.seg * 0.85), 1.6)
                fade = 1 - smooth(ramp(t, tc + self.seg, tc + self.seg + 0.5))
                cv.light_path(Path(pts), u, tail=0.6, color=self.color, width=4.0, energy=1.05 * out * fade)
        if t >= self.key:
            k = ramp(t, self.key, self.key + 0.7)
            x0 = self.rects[0][0]
            x1 = max(r[0] + r[2] for r in self.rects)
            c = lerp(x0 - 200, x1 + 400, ease_in_out(k))
            for i, (x, y, w, h) in enumerate(self.rects):
                sp = self.sp(("glow", i), lambda w=w, h=h: ui.card(w, h, 26, self.ppd, self.fonts, T,
                                                                   fill=mix_hex(self.color, "#FFFFFF", 0.4),
                                                                   border=False))
                cv.draw(sp, x + w / 2, y + h / 2, alpha=0.22 * out * math.sin(math.pi * k), band=(c, 120, 20))


def _feature_card(sc, it, w, h):
    T = sc.T
    p = ui.Painter(w, h, sc.ppd, sc.fonts)
    p.rrect(0, 0, w, h, 26, fill=T.card, outline=T.border, width=1.2)
    ui.draw_kind_icon(p, it.get("kind", "doc"), 30, 30, 52, T)
    p.text(30, 126, it.get("title", ""), 30, 600, T.text)
    if it.get("desc"):
        p.text(30, 166, it["desc"], 22, 400, T.text2)
    else:
        p.rrect(30, 152, w * 0.5, 9, 4.5, fill=T.track)
        p.rrect(30, 172, w * 0.34, 9, 4.5, fill=T.track)
    return p.sprite()


# ===========================================================================
# phone (mobile app)
# ===========================================================================

@register("phone")
class Phone(Scene):
    """data: {"image": app screenshot (optional, tall is fine), "title": "...",
    "rows": ["...", ...], "tap": 1, "notify": {"title": "...", "body": "..."}}"""
    weight = 1.5

    def setup(self):
        f, d = self.f, self.data
        self.sw, self.sh = 360.0, 760.0
        self.k = min(1.0 if not f.portrait else 1.55, (f.DH - 140) / (self.sh + 24))
        self.C = np.array([f.CX + (0 if f.portrait else float(d.get("offset_x", 0))), f.CY])
        self.t_tap = max(self.a + 0.5 * self.L, self.land + 0.9)
        self.key = max(self.a + 0.72 * self.L, self.t_tap + 0.55)
        self.tap_i = int(d.get("tap", 1))

    def screen_rect(self, bob=0.0):
        k = self.k
        return (self.C[0] - self.sw * k / 2, self.C[1] - self.sh * k / 2 + bob, self.sw * k, self.sh * k)

    def content(self):
        d = self.data
        if d.get("image"):
            path = d["image"] if os.path.isabs(d["image"]) else os.path.join(self.f.dir, d["image"])
            im = Image.open(path).convert("RGBA")
            w = int(self.sw * self.ppd * 2)
            im = im.resize((w, int(im.height * w / im.width)), Image.LANCZOS)
            return _sprite_from_pil(im, self.ppd * 2)
        return _app_screen(self, self.sw, self.sh * 1.5)

    def row_rect(self, i, scroll, bob=0.0):
        x, y, w, h = self.screen_rect(bob)
        k = self.k
        return (x + 16 * k, y + (150 + i * 96) * k - scroll * k, w - 32 * k, 84 * k)

    def scroll(self, t):
        return 160 * ease_in_out(ramp(t, self.land, self.t_tap - 0.2), 2.0)

    def bob(self, t, a_in=1.0):
        return 6 * math.sin((t - self.a) * 1.6) + 60 * (1 - a_in)

    def entry(self):
        x, y, w, h = self.screen_rect()
        return (x + w / 2, y + h * 0.75)

    def exit(self):
        x, y, w, h = self.screen_rect()
        return (x + w - 20 * self.k, y + 70 * self.k)

    def anchors(self):
        return {"key": self.key, "tap": self.t_tap}

    def cues(self):
        return [dict(t=self.a, kind="swoosh", gain=-19, dur=0.6), dict(t=self.t_tap, kind="click", gain=-15),
                dict(t=self.t_tap + 0.02, kind="pop_soft", gain=-20), dict(t=self.key, kind="success", gain=-15),
                dict(t=self.key - 0.1, kind="whoosh_short", gain=-22)]

    def draw(self, cv, t):
        f, T, k = self.f, self.T, self.k
        out = 1 - smooth(ramp(t, self.b - 0.15, self.b + 0.2))
        a_in = ease_out(ramp(t, self.t_in, self.t_in + 0.4), 2.6)
        if out * a_in <= 0:
            return
        bob = self.bob(t, a_in)
        x, y, w, h = self.screen_rect(bob)
        clip = (x, y, x + w, y + h)
        cont = self.sp("content", self.content)
        sc = self.scroll(t)
        cv.draw(cont, x, y - sc * k, scale=k * self.sw / cont.w, anchor=(0, 0), alpha=a_in * out, clip=clip)
        if t >= self.t_tap:
            rx, ry, rw, rh = self.row_rect(self.tap_i, sc, bob)
            hl = self.sp("hl", lambda: ui.card(self.sw - 32, 84, 16, self.ppd, self.fonts, T,
                                              fill=mix_hex(self.color, T.card, 0.85), border=False))
            cv.draw(hl, rx + rw / 2, ry + rh / 2, scale=k,
                    alpha=0.9 * out * (1 - 0.4 * ramp(t, self.t_tap + 0.4, self.b)), clip=clip)
            rr = ease_out(ramp(t, self.t_tap, self.t_tap + 0.45), 2.0)
            if rr < 1:
                c = (rx + rw * 0.7, ry + rh / 2)
                ring = [(c[0] + 70 * rr * math.cos(a_), c[1] + 70 * rr * math.sin(a_)) for a_ in np.linspace(0, 6.3, 60)]
                cv.light_poly(ring, lin(self.color), 0.9 * (1 - rr) * out, width=4.0)
        fr, _ = self.sp("frame", lambda: ui.handheld(self.sw, self.sh, self.ppd, self.fonts, T, phone=True))
        cv.draw(fr, x + w / 2, y + h / 2, scale=k, alpha=a_in * out)
        if self.land - 0.05 <= t <= self.t_tap + 0.05:
            rx, ry, rw, rh = self.row_rect(self.tap_i, self.scroll(self.t_tap), self.bob(self.t_tap))
            p0 = np.array(self.entry())
            p1 = np.array((rx + rw * 0.7, ry + rh / 2))
            path = self.f.sprite((self.i, "tap_path"), lambda: Path.through(p0, p1, bend=0.3, side=1))
            u = ease_in_out(ramp(t, self.land, self.t_tap), 2.0)
            cv.light_path(path, u, tail=0.5, color=self.color, width=4.0, energy=1.0 * out)
        if t >= self.key - 0.12:
            nb = self.sp("notif", lambda: _notification(self, self.sw - 24, 96))
            drop = spring_in(t - (self.key - 0.12), 3.6, 0.6)
            ny = y + 62 * k - 120 * k * (1 - drop)
            cv.draw(nb, x + w / 2, ny, scale=k, alpha=out * min(1, drop * 1.5), clip=clip)
            u = ease_out(ramp(t, self.key, self.key + 0.55), 2.0)
            pts = rrect_points(x + 12 * k, ny - 48 * k, (self.sw - 24) * k, 96 * k, 22 * k, 240)
            cv.light_path(Path(pts), u, tail=0.6, color=self.color, width=3.6, energy=0.95 * out)


def _app_screen(sc, w, h):
    T, d = sc.T, sc.data
    p = ui.Painter(w, h, sc.ppd, sc.fonts)
    p.rrect(0, 0, w, h, 0, fill=T.card)
    p.text(20, 100, d.get("title", sc.f.name), 34, 700, T.text)
    rows = d.get("rows") or ["Morning standup", "Design review", "Ship v2.4", "Customer call", "Plan sprint",
                             "Write release notes", "Weekly sync"]
    pal = sc.f.palette
    for i, r_ in enumerate(rows):
        y = 150 + i * 96
        p.rrect(16, y, w - 32, 84, 16, fill=T.paper_frame)
        p.circle(56, y + 42, 22, mix_hex(pal[i % len(pal)], T.card, 0.25))
        p.text(92, y + 38, r_, 22, 500, T.text)
        p.rrect(92, y + 52, (w - 140) * (0.4 + 0.1 * (i % 3)), 8, 4, fill=T.track)
    return p.sprite()


def _notification(sc, w, h):
    T, d = sc.T, sc.data
    n = d.get("notify") or {}
    p = ui.Painter(w, h, sc.ppd, sc.fonts)
    p.rrect(0, 0, w, h, 22, fill=T.card, outline=T.border, width=1.2)
    p.rrect(16, 20, 40, 40, 10, fill=sc.color)
    p.text(70, 40, n.get("title", sc.f.name), 20, 600, T.text)
    p.text(70, 66, n.get("body", "Done. Everyone's up to date."), 18, 400, T.text2)
    return p.sprite()


# ===========================================================================
# product (hero image: a car, a device, a bottle...)
# ===========================================================================

@register("product")
class Product(Scene):
    """data: {"image": "assets/product.png" (transparent PNG best),
    "specs": [{"value": "0-60 in 3.1 s", "at": [x, y]}]}  (at: 0-1 on the image)"""
    weight = 1.8

    def setup(self):
        f, d = self.f, self.data
        path = d["image"] if os.path.isabs(d["image"]) else os.path.join(f.dir, d["image"])
        im = Image.open(path).convert("RGBA")
        bb = im.getchannel("A").getbbox()
        if bb:
            im = im.crop(bb)
        self.im = im
        maxw, maxh = f.DW * (0.72 if not f.portrait else 0.9), f.DH * (0.55 if not f.portrait else 0.42)
        self.scale_fit = min(maxw / im.width, maxh / im.height)
        self.pw, self.ph = im.width * self.scale_fit, im.height * self.scale_fit
        self.C = np.array([f.CX, f.CY + (20 if not f.portrait else -80)])
        a = np.asarray(im.getchannel("A")).astype(np.float32) / 255
        self.has_alpha = (a < 0.5).mean() > 0.05
        if self.has_alpha:
            import cv2
            m = (a > 0.5).astype(np.uint8) * 255
            cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            c = max(cs, key=cv2.contourArea)[:, 0, :].astype(np.float64)
            c = resample(c, 700, closed=True)
            c = np.roll(c, -int(np.argmin(c[:, 0])), axis=0)
            self.contour = c * self.scale_fit - (self.pw / 2, self.ph / 2)
        else:
            self.contour = rrect_points(-self.pw / 2, -self.ph / 2, self.pw, self.ph, 24, 700)
        self.specs = d.get("specs") or []
        self.t_rim = self.land
        self.t_sheen = self.land + 0.9
        n = max(1, len(self.specs))
        self.key = max(self.a + 0.8 * self.L, self.t_sheen + 0.4 + 0.45 * n)
        self.t_specs = [self.t_sheen + 0.35 + i * (self.key - self.t_sheen - 0.35) / n for i in range(len(self.specs))]

    def sprite(self):
        return self.sp("img", lambda: _sprite_from_pil(self.im, self.ppd * self.scale_fit * 1.08))

    def zoom(self, t):
        return 0.94 + 0.08 * smooth(ramp(t, self.a - 0.2, self.b + 0.4))

    def world(self, p, t):
        return self.C + np.asarray(p) * self.zoom(t)

    def entry(self):
        return tuple(self.world(self.contour[0], self.land))

    def exit(self):
        return tuple(self.world(self.contour[len(self.contour) // 2], self.b))

    def anchors(self):
        a = {"key": self.key, "sheen": self.t_sheen}
        for i, ts in enumerate(self.t_specs):
            a[f"s{i + 1}"] = ts
        return a

    def cues(self):
        c = [dict(t=self.a, kind="swoosh", gain=-17, dur=0.9), dict(t=self.t_sheen, kind="shimmer", gain=-19, pitch=0.8),
             dict(t=self.land + 0.05, kind="whoosh_soft", gain=-20)]
        for i, ts in enumerate(self.t_specs):
            c.append(dict(t=ts, kind="tick", gain=-20, pitch=1.0 + 0.1 * i))
        return c

    def draw(self, cv, t):
        f, T = self.f, self.T
        out = 1 - smooth(ramp(t, self.b - 0.15, self.b + 0.2))
        a_in = ease_out(ramp(t, self.t_in, self.t_in + 0.4), 2.2)
        al = a_in * out
        if al <= 0:
            return
        sp = self.sprite()
        z = self.zoom(t)
        sc = z * self.pw / sp.w
        cx, cy = self.C
        sh = self.sp("shadow", lambda: _ellipse_sprite(self, self.pw * 0.8, 60,
                                                      "#000000" if f.stage_mode == "light" else self.color))
        cv.draw(sh, cx, cy + self.ph * z / 2 + 8, scale=z, alpha=0.22 * al, blur=14)
        cv.draw(sp, cx, cy + self.ph * z + 4, scale=sc, alpha=0.16 * al, flip=True,
                grad=(cx, cy + self.ph * z / 2, cx, cy + self.ph * z * 0.95))
        cv.draw(sp, cx, cy + 20 * (1 - a_in), scale=sc, alpha=al)
        if t >= self.t_sheen:
            k = ramp(t, self.t_sheen, self.t_sheen + 0.9)
            c = lerp(cx - self.pw, cx + self.pw, ease_in_out(k))
            white = self.sp("white", lambda: _sprite_from_pil(_whiten(self.im), self.ppd * self.scale_fit * 1.08))
            cv.draw(white, cx, cy, scale=sc, alpha=0.55 * out * math.sin(math.pi * k), band=(c, 90, 18))
        if t >= self.t_rim:
            pts = np.array([self.world(p, t) for p in self.contour])
            u = ease_out(ramp(t, self.t_rim, self.t_rim + 1.4), 2.0) + 0.25 * max(0, t - self.t_rim - 1.4)
            fade = 1 - 0.6 * ramp(t, self.t_rim + 1.4, self.b)
            cv.light_path(Path(pts, closed=True), u, tail=0.35, color=self.color, width=3.6, energy=1.0 * out * fade)
        for i, (s_, ts) in enumerate(zip(self.specs, self.t_specs)):
            if t < ts:
                continue
            ax, ay = s_.get("at", [0.25 + 0.5 * (i % 2), 0.3 + 0.2 * (i // 2)])
            p0 = self.world((ax * self.pw - self.pw / 2, ay * self.ph - self.ph / 2), t)
            side = -1 if ax < 0.5 else 1
            p1 = p0 + (np.array([side * 90, -150]) if f.portrait else np.array([side * 160, -110]))
            u = ease_out(ramp(t, ts, ts + 0.35), 2.0)
            cv.light_poly([p0, p0 + (p1 - p0) * u], lin(self.color), 0.8 * out, width=3.0)
            cv.light_dot(p0, lin(self.color), energy=1.4 * out, radius=5)
            if u > 0.6:
                chp = self.sp(("spec", i), lambda s_=s_: ui.chip(s_["value"], self.ppd, self.fonts, T, icon="dot",
                                                                color=self.color, size=24, h=52))
                pop = spring_in(t - ts - 0.2, 4.0, 0.55)
                cv.draw(chp, p1[0] + side * chp.w / 2, p1[1], scale=max(0.01, pop), alpha=out)


def _sprite_from_pil(im, ppd):
    from .gfx import Sprite
    return Sprite.from_pil(im, ppd)


def _whiten(im):
    arr = np.asarray(im).copy()
    arr[..., :3] = 255
    return Image.fromarray(arr)


def _ellipse_sprite(sc, w, h, color):
    p = ui.Painter(w, h, sc.ppd, sc.fonts)
    p.d.ellipse([0, 0, w * p.k - 1, h * p.k - 1], fill=ui._rgba(color))
    return p.sprite()


# ===========================================================================
# signature (the product's motif as a hero moment)
# ===========================================================================

@register("signature")
class Signature(Scene):
    """The product's own shape, drawn by the light across the whole frame like a live monitor
    trace (heartbeat, sound wave, circuit, route...). Older trace fades behind a hot head; every
    event in the motif (a spike, a node, a pin) flashes a ripple and makes its sound.

    data: {"lines": ["Every second", "counts."], "highlight": "counts.", "kind": "ecg"}
    (kind defaults to the film's brief.signature)
    """
    weight = 1.3

    def setup(self):
        from .motifs import beats, motif_points
        f, d = self.f, self.data
        self.motif = d.get("kind") or (f.signature if f.signature != "none" else "wave")
        self.lines = d.get("lines") or []
        self.hl = (d.get("highlight") or "").lower().strip(".,!?")
        has_text = bool(self.lines)
        self.y = f.CY + (110 if has_text and not f.portrait else 160 if has_text else 0)
        x0, x1 = 70.0, f.DW - 70.0
        amp = {"ecg": 150.0, "wave": 95.0, "ripple": 80.0, "route": 170.0, "spark": 110.0, "rise": 230.0}.get(self.motif)
        if self.motif == "rise":
            p0, p1 = (x0, self.y + 120), (x1, self.y - 140)
        else:
            p0, p1 = (x0, self.y), (x1, self.y)
        # a real heart rhythm: about three beats across the frame
        self.period = (f.DW - 140) / 3.3 if self.motif == "ecg" else None
        self.pts = motif_points(self.motif, p0, p1, amp=amp, seed=self.i + 17, envelope=self.motif != "ecg",
                                period=self.period)
        self.path = Path(self.pts)
        self.t0 = self.land + 0.05
        self.key = max(self.a + 0.78 * self.L, self.t0 + 1.4)
        self.events = beats(self.motif, p0, p1, seed=self.i + 17, period=self.period)
        # type: one or two short lines above the trace, word by word
        self.size = 84.0 if not f.portrait else 66.0
        if self.lines:
            wmax = max(f.type.width(l, self.size, 700) for l in self.lines)
            if wmax > f.DW * 0.86:
                self.size *= f.DW * 0.86 / wmax
        self.words = []
        lh = self.size * 1.12
        top = self.y - 150 - lh * (len(self.lines) - 1)
        t = self.land + 0.25
        for li, line in enumerate(self.lines):
            lw = f.type.width(line, self.size, 700)
            x = f.CX - lw / 2
            for w in line.split(" "):
                self.words.append(dict(w=w, x=x, y=top + li * lh, t=t))
                x += f.type.width(w + " ", self.size, 700)
                t += 0.09

    def u_at(self, t):
        return ease_in_out(ramp(t, self.t0, self.key), 1.35)

    def event_times(self):
        ts = np.linspace(self.t0, self.key, 600)
        us = np.array([self.u_at(t) for t in ts])
        return [float(ts[min(len(ts) - 1, int(np.searchsorted(us, e)))]) for e in self.events]

    def entry(self):
        return tuple(self.pts[0])

    def exit(self):
        return tuple(self.pts[-1])

    def anchors(self):
        a = {"key": self.key}
        for i, te in enumerate(self.event_times()):
            a[f"e{i + 1}"] = te
        return a

    def cues(self):
        kind = {"ecg": "beep", "circuit": "tick", "route": "pop", "wave": "shimmer", "ripple": "pop_soft",
                "spark": "sparkle", "rise": "tick"}.get(self.motif, "tick")
        c = [dict(t=te, kind=kind, gain=-17, pitch=1.0 + 0.04 * i) for i, te in enumerate(self.event_times())]
        c += [dict(t=w["t"], kind="tick", gain=-29, pitch=0.8) for w in self.words]
        c.append(dict(t=self.key, kind="success", gain=-17))
        return c

    def draw(self, cv, t):
        f = self.f
        out = 1 - smooth(ramp(t, self.b - 0.15, self.b + 0.2))
        if out <= 0 or t < self.land - 0.05:
            return
        u = self.u_at(t)
        col = lin(self.color)
        n = len(self.pts)
        if u > 0:
            # the trace: older parts dim like a monitor, the newest part burns
            m = max(2, int(n * u))
            seg = self.pts[:m]
            age = np.linspace(0, 1, m)
            inten = (0.18 + 0.55 * age ** 2.2) * out
            if t > self.key:
                inten = np.maximum(inten, 0.32 * out)
            cv.light_poly(seg, col, inten, width=4.2)
            if t <= self.key + 0.05:
                cv.light_path(self.path, u, tail=0.06, color=self.color, width=5.0, energy=1.35 * out, head_glow=1.3)
        # each event flashes a ripple where it happens
        for e, te in zip(self.events, self.event_times()):
            k = ramp(t, te, te + 0.6)
            if 0 < k < 1:
                p = self.path.at(e)
                rr = ease_out(k, 2.2)
                ring = [(p[0] + 70 * rr * math.cos(a_), p[1] + 70 * rr * math.sin(a_)) for a_ in np.linspace(0, 6.3, 64)]
                cv.light_poly(ring, col, 0.9 * (1 - k) * out, width=3.4)
                cv.light_dot(p, col, energy=1.6 * (1 - k) * out, radius=7)
        for w in self.words:
            a = ease_out(ramp(t, w["t"], w["t"] + 0.3), 2.6)
            if a <= 0:
                continue
            is_hl = self.hl and w["w"].lower().strip(".,!?") == self.hl
            sp = self.text(w["w"], self.size, 700, self.color if is_hl else f.text_color)
            cv.draw(sp, w["x"], w["y"] + 22 * (1 - a), alpha=a * out, blur=4 * (1 - a), anchor=(0, sp.baseline / sp.h))


# ===========================================================================

def build_scenes(film, acts):
    try:
        import importlib.util
        cp = os.path.join(film.dir, "custom_scenes.py")
        if os.path.exists(cp):
            spec = importlib.util.spec_from_file_location("custom_scenes", cp)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
    except Exception as e:
        raise RuntimeError(f"custom_scenes.py failed to load: {e}")
    if not acts:
        acts = [dict(scene="upload"), dict(scene="search"), dict(scene="share"), dict(scene="devices")]
    start, end = film.acts_start, film.race_t + (0.16 if film.ending == "cut" else 0.0)
    ft = film.format_transitions
    if ft:  # the format's rhythm fills transitions the brief does not set
        acts = [dict(a) for a in acts]
        for i, a in enumerate(acts):
            if i and "transition" not in a:
                a["transition"] = ft[(i - 1) % len(ft)]
    ws = [float(a.get("weight", SCENES[a["scene"]].weight)) for a in acts]
    tot = sum(ws)
    out = []
    t = start
    for i, (a, w) in enumerate(zip(acts, ws)):
        cls = SCENES.get(a["scene"])
        if cls is None:
            raise ValueError(f"unknown scene '{a['scene']}'. Known: {sorted(SCENES)}")
        dur = (end - start) * w / tot
        sc = cls(film, a, i, t, t + dur)
        out.append(sc)
        t += dur
    for i, sc in enumerate(out):
        sc.prev = out[i - 1] if i else None
        sc.next = out[i + 1] if i + 1 < len(out) else None
    for sc in out:
        sc.setup()
    return out
