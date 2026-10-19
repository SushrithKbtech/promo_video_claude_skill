"""Custom scenes for the FirstDropAI HealthCareSim film.

roleplay  a patient conversation; the light is the tension meter: it fills hot as the
          patient escalates and drains to calm when the trainee answers well.
report    the feedback report; the light draws the overall score ring and fills
          the five communication subscores.
"""
import math

import numpy as np

from adkit import ui
from adkit.core import Path, ease_in_out, ease_out, lerp, lin, mix_hex, pulse, ramp, smooth, spring_in
from adkit.scenes import Scene, register

WARM = "#F2784B"


def _bubble(sc, text, w, mine):
    T = sc.T
    h = 64
    p = ui.Painter(w, h, sc.ppd, sc.fonts)
    fill = mix_hex(sc.color, "#FFFFFF", 0.86) if mine else mix_hex(WARM, "#FFFFFF", 0.88)
    p.rrect(0, 0, w, h, 22, fill=fill)
    p.text(24, h / 2 + 8, text, 23, 500, T.text)
    return p.sprite()


@register("roleplay")
class Roleplay(Scene):
    weight = 2.0

    def setup(self):
        f, d = self.f, self.data
        self.W, self.H = 1040.0, 600.0
        self.C = np.array([f.CX, f.CY + 10])
        self.x0, self.y0 = self.C[0] - self.W / 2, self.C[1] - self.H / 2
        self.lines = d.get("lines") or [
            ("patient", "Why is nobody telling me anything?!"),
            ("trainee", "I hear you. Let me find out exactly where things are."),
            ("patient", "Okay. Thank you."),
        ]
        self.mx0, self.mx1 = self.x0 + self.W - 300, self.x0 + self.W - 48
        self.my = self.y0 + 52
        L = self.land
        self.t_p1 = L + 0.15
        self.t_type = self.t_p1 + 0.75
        self.cps = 0.032
        self.t_send = self.t_type + self.cps * len(self.lines[1][1]) + 0.12
        self.key = max(self.t_send + 0.3, self.a + 0.62 * self.L)
        self.t_p2 = self.key + 0.75
        self.hi, self.lo = 0.84, 0.26

    def level(self, t):
        up = ease_out(ramp(t, self.t_p1, self.t_p1 + 0.6), 2.2) * self.hi
        down = ease_in_out(ramp(t, self.key, self.key + 0.9), 2.0)
        return lerp(up, self.lo, down) if t >= self.key else up

    def calm(self, t):
        return ease_in_out(ramp(t, self.key, self.key + 0.9), 2.0)

    def entry(self):
        return (self.mx0, self.my)

    def exit(self):
        return (self.mx0 + (self.mx1 - self.mx0) * self.lo, self.my)

    def anchors(self):
        return {"key": self.key, "send": self.t_send, "patient": self.t_p1}

    def tint_color(self):
        return self.color

    def cues(self):
        c = [dict(t=self.t_p1, kind="pop", gain=-16, pitch=0.8),
             dict(t=self.t_p1 + 0.05, kind="riser_short", gain=-24, dur=0.55)]
        c += [dict(t=self.t_type + i * self.cps, kind="key", gain=-29, pitch=1.05 + 0.2 * ((i * 0.37) % 1))
              for i, ch in enumerate(self.lines[1][1]) if ch != " " and i % 2 == 0]
        c += [dict(t=self.t_send, kind="send", gain=-17), dict(t=self.key, kind="whoosh_soft", gain=-19),
              dict(t=self.key + 0.85, kind="success", gain=-17), dict(t=self.t_p2, kind="pop", gain=-19, pitch=1.1)]
        return c

    def draw(self, cv, t):
        f, T = self.f, self.T
        out = 1 - smooth(ramp(t, self.b - 0.15, self.b + 0.2))
        a_in = ease_out(ramp(t, self.a + 0.18, self.a + 0.58), 2.4)
        al = a_in * out
        if al <= 0:
            return
        win = self.sp("win", lambda: ui.card(self.W, self.H, 26, self.ppd, self.fonts, T))
        cv.draw(win, self.C[0], self.C[1] + 24 * (1 - a_in), alpha=al)
        dy = 24 * (1 - a_in)
        # header: patient, context, turn, tension meter
        av = self.sp("av", lambda: ui.avatar(WARM, "P", self.ppd, self.fonts, r=20))
        cv.draw(av, self.x0 + 52, self.my + dy, alpha=al)
        h1 = self.text(self.data.get("who", "Parent, ER waiting room"), 24, 600)
        cv.draw(h1, self.x0 + 86, self.my + 8 + dy, alpha=al, anchor=(0, h1.baseline / h1.h))
        turn = self.text("Turn 2 of 6" if t < self.key else "Turn 3 of 6", 19, 500, f.text2)
        cv.draw(turn, self.mx0 - 28, self.my + 7 + dy, alpha=al, anchor=(1, turn.baseline / turn.h))
        lab = self.text("Tension", 18, 600, f.text2)
        cv.draw(lab, self.mx0, self.my - 20 + dy, alpha=al, anchor=(0, lab.baseline / lab.h))
        state = "High" if self.calm(t) < 0.5 else "Calm"
        col_state = WARM if state == "High" else self.color
        st = self.text(state, 18, 700, col_state)
        cv.draw(st, self.mx1, self.my - 20 + dy, alpha=al * (1 if t >= self.t_p1 + 0.2 else 0),
                anchor=(1, st.baseline / st.h))
        trk = self.sp("trk", lambda: ui.bar(self.mx1 - self.mx0, 10, T.track, self.ppd, self.fonts))
        cv.draw(trk, self.mx0, self.my + dy, anchor=(0, 0.5), alpha=al)
        # the light IS the meter: warm while escalating, cooling to the brand colour
        lv = self.level(t)
        if t >= self.land - 0.02 and lv > 0.005:
            xh = self.mx0 + (self.mx1 - self.mx0) * lv
            k = self.calm(t)
            col = np.asarray(lin(WARM)) * (1 - k) + np.asarray(lin(self.color)) * k
            n = 40
            xs = np.linspace(self.mx0, xh, n)
            cv.light_poly(np.stack([xs, np.full(n, self.my + dy)], 1), col, 0.82 * out, width=10)
            cv.light_dot((xh, self.my + dy), col, energy=1.5 * out, radius=7)
        # conversation
        bx = self.x0 + 44
        by = self.y0 + 140 + dy
        if t >= self.t_p1:
            pop = spring_in(t - self.t_p1, 4.5, 0.6)
            sp = self.sp("p1", lambda: _bubble(self, self.lines[0][1], 520, False))
            cv.draw(sp, bx, by + 14 * (1 - min(1, pop)), anchor=(0, 0), alpha=al * min(1, pop * 1.5))
        # trainee typing in the composer
        cx0, cy = self.x0 + 44, self.y0 + self.H - 70 + dy
        comp = self.sp("comp", lambda: ui.pill(self.W - 88, 64, self.ppd, self.fonts, T.field))
        cv.draw(comp, self.C[0], cy, alpha=al)
        msg = self.lines[1][1]
        if self.t_type <= t < self.t_send + 0.1:
            n = int(min(len(msg), max(0, (t - self.t_type) / self.cps)))
            q = self.text(msg, 23, 400)
            if n:
                cv.draw(q, cx0 + 28, cy + 8, alpha=al * (1 - ramp(t, self.t_send, self.t_send + 0.1)),
                        anchor=(0, q.baseline / q.h), wipe=(cx0 + 28 + q.meta["advance"][n] + 1, 1.0))
                cxp = cx0 + 30 + q.meta["advance"][n]
                cv.light_poly([(cxp, cy - 14), (cxp, cy + 14)], lin(self.color), 0.9 * out, width=3)
        elif t < self.t_type:
            ph = self.text("Type your response...", 22, 400, f.text2)
            cv.draw(ph, cx0 + 28, cy + 8, alpha=al, anchor=(0, ph.baseline / ph.h))
        if t >= self.t_send:
            pop = spring_in(t - self.t_send, 4.5, 0.6)
            w = 700
            sp = self.sp("t1", lambda: _bubble(self, msg, w, True))
            cv.draw(sp, self.x0 + self.W - 44, by + 92 + 14 * (1 - min(1, pop)), anchor=(1, 0),
                    alpha=al * min(1, pop * 1.5))
        if t >= self.t_p2:
            pop = spring_in(t - self.t_p2, 4.5, 0.6)
            sp = self.sp("p2", lambda: _bubble(self, self.lines[2][1], 260, False))
            cv.draw(sp, bx, by + 184 + 14 * (1 - min(1, pop)), anchor=(0, 0), alpha=al * min(1, pop * 1.5))


@register("report")
class Report(Scene):
    weight = 1.6

    def setup(self):
        f, d = self.f, self.data
        self.W, self.H = 1060.0, 580.0
        self.C = np.array([f.CX, f.CY + 10])
        self.x0, self.y0 = self.C[0] - self.W / 2, self.C[1] - self.H / 2
        self.score = int(d.get("score", 86))
        self.dims = d.get("dims") or [("Empathy", 92), ("Clarity", 84), ("Structure & next steps", 80),
                                      ("Professionalism", 90), ("Pressure & de-escalation", 78)]
        self.rc = np.array([self.x0 + 230, self.C[1] + 30])
        self.R = 128.0
        self.t_ring = self.land + 0.05
        self.key = max(self.a + 0.62 * self.L, self.t_ring + 1.1)
        self.t_bars = [self.t_ring + 0.35 + 0.16 * i for i in range(len(self.dims))]
        self.bx0 = self.x0 + 470
        self.bx1 = self.x0 + self.W - 110

    def ring_path(self):
        a = np.linspace(-math.pi / 2, -math.pi / 2 + 2 * math.pi, 400)
        return Path(np.stack([self.rc[0] + self.R * np.cos(a), self.rc[1] + self.R * np.sin(a)], 1))

    def entry(self):
        return (self.rc[0], self.rc[1] - self.R)

    def exit(self):
        return (self.bx1, self.y0 + 150 + (len(self.dims) - 1) * 72)

    def anchors(self):
        return {"key": self.key}

    def cues(self):
        c = [dict(t=self.t_ring, kind="riser_short", gain=-22, dur=self.key - self.t_ring)]
        c += [dict(t=tb + 0.35, kind="tick", gain=-22, pitch=1.0 + 0.08 * i) for i, tb in enumerate(self.t_bars)]
        c += [dict(t=self.key, kind="pop_low", gain=-14), dict(t=self.key + 0.02, kind="success", gain=-16)]
        return c

    def draw(self, cv, t):
        f, T = self.f, self.T
        out = 1 - smooth(ramp(t, self.b - 0.15, self.b + 0.2))
        a_in = ease_out(ramp(t, self.a + 0.18, self.a + 0.58), 2.4)
        al = a_in * out
        if al <= 0:
            return
        dy = 24 * (1 - a_in)
        card = self.sp("card", lambda: ui.card(self.W, self.H, 26, self.ppd, self.fonts, T))
        cv.draw(card, self.C[0], self.C[1] + dy, alpha=al)
        ttl = self.text("Feedback report", 30, 700)
        cv.draw(ttl, self.x0 + 48, self.y0 + 70 + dy, alpha=al, anchor=(0, ttl.baseline / ttl.h))
        # score ring: grey track + the light drawing the score
        trk = self.sp("ring", lambda: _ring(self, self.R, 14, T.track))
        cv.draw(trk, self.rc[0], self.rc[1] + dy, alpha=al)
        u = ease_in_out(ramp(t, self.t_ring, self.key), 1.8) * self.score / 100
        if u > 0:
            path = self.ring_path()
            cv.light_poly(path.span(0, u, 160) + (0, dy), lin(self.color), 0.8 * out, width=14)
            cv.light_dot(path.at(u) + (0, dy), lin(self.color), energy=1.5 * out, radius=9)
        val = int(round(self.score * ease_out(ramp(t, self.t_ring, self.key), 3.0)))
        num = self.text(str(val), 84, 700)
        cv.draw(num, self.rc[0], self.rc[1] + 22 + dy, scale=1 + 0.06 * pulse(t, self.key, 0.02, 0.2), alpha=al,
                anchor=(0.5, num.baseline / num.h))
        lb = self.text("Overall", 22, 500, f.text2)
        cv.draw(lb, self.rc[0], self.rc[1] + 62 + dy, alpha=al, anchor=(0.5, lb.baseline / lb.h))
        # subscores
        for i, ((name, v), tb) in enumerate(zip(self.dims, self.t_bars)):
            y = self.y0 + 150 + i * 72 + dy
            nm = self.text(name, 22, 500)
            cv.draw(nm, self.bx0, y - 16, alpha=al, anchor=(0, nm.baseline / nm.h))
            bt = self.sp(("bt",), lambda: ui.bar(self.bx1 - self.bx0, 10, T.track, self.ppd, self.fonts))
            cv.draw(bt, self.bx0, y, anchor=(0, 0.5), alpha=al)
            k = ease_out(ramp(t, tb, tb + 0.6), 2.4)
            if k > 0:
                xe = self.bx0 + (self.bx1 - self.bx0) * v / 100 * k
                n = 30
                xs = np.linspace(self.bx0, xe, n)
                cv.light_poly(np.stack([xs, np.full(n, y)], 1), lin(self.color), 0.72 * out, width=10)
                cv.light_dot((xe, y), lin(self.color), energy=1.3 * out, radius=6)
                vt = self.text(str(int(round(v * k))), 22, 600, f.text2)
                cv.draw(vt, self.bx1 + 54, y + 8, alpha=al, anchor=(1, vt.baseline / vt.h))


def _ring(sc, R, w, color):
    s = 2 * R + w + 4
    p = ui.Painter(s, s, sc.ppd, sc.fonts)
    p.ring(s / 2, s / 2, R, w, color)
    return p.sprite()
