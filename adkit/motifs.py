"""Signature motifs: the shape the light takes, chosen from what the product *is*.

A health product's light travels as a heartbeat, an audio product's as a sound wave,
a dev tool's as circuit traces, a maps or delivery product's as a winding route.
The same motif shapes the hand-overs, the peel, the opening and the end-card line, and
gets a hero moment of its own (the ``signature`` scene).

    motif_points(kind, p0, p1, amp, seed) -> (n, 2) points from p0 to p1
    beats(kind, p0, p1)                   -> arc fractions of the motif's "events"
                                             (heartbeat spikes, circuit nodes, route pins)
"""
from __future__ import annotations

import math

import numpy as np

KINDS = ("ecg", "wave", "circuit", "route", "ripple", "rise", "spark", "none")

# what each motif suits, for the director (SKILL.md repeats this)
SUITS = {
    "ecg": "health, care, fitness, medical training, wellbeing",
    "wave": "music, audio, voice, podcasts, calls",
    "circuit": "dev tools, AI, hardware, infra, security",
    "route": "maps, travel, delivery, logistics, mobility, cars",
    "ripple": "messaging, social, notifications, community",
    "rise": "finance, analytics, growth, sales, investing",
    "spark": "creative tools, design, writing, education",
}


def _frame(p0, p1):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    L = float(np.linalg.norm(d)) or 1.0
    u = d / L
    n = np.array([-u[1], u[0]])
    if n[1] > 0:  # keep "up" on screen as the positive side
        n = -n
    return p0, p1, u, n, L


def _gauss(x, c, w):
    return np.exp(-((x - c) / w) ** 2)


def ecg_offset(x, period=230.0, amp=70.0, phase=0.5):
    """Heartbeat trace offset (px, +up) at distance x (px) along the line."""
    k = np.floor(x / period - phase + 0.5)
    c = (k + phase) * period
    r = x - c
    y = (0.10 * _gauss(r, -62, 13)           # P wave
         - 0.16 * _gauss(r, -14, 4.5)        # Q
         + 1.00 * _gauss(r, 0, 4.2)          # R spike
         - 0.34 * _gauss(r, 11, 4.8)         # S
         + 0.20 * _gauss(r, 58, 17))         # T wave
    return y * amp


def motif_points(kind, p0, p1, amp=None, seed=0, n=None, envelope=True, period=None):
    p0, p1, u, nrm, L = _frame(p0, p1)
    n = n or int(max(120, min(1400, L * (1.6 if kind in ("ecg", "circuit") else 0.8))))
    s = np.linspace(0, 1, n)
    x = s * L
    env = np.sin(np.pi * s) ** 0.6 if envelope else np.ones_like(s)
    r = np.random.default_rng(seed)
    if kind == "ecg":
        amp = amp or min(90.0, L * 0.12)
        period = period or max(160.0, min(260.0, L / 2.6))
        off = ecg_offset(x, period, amp, phase=0.5 + 0.25 * r.random()) * (env > 0.25)
    elif kind == "wave":
        amp = amp or min(70.0, L * 0.09)
        f1, f2 = 1 / 38.0, 1 / 61.0
        off = amp * env * (0.7 * np.sin(2 * np.pi * f1 * x + r.random() * 6) * (0.5 + 0.5 * np.sin(2 * np.pi * x / L * 3))
                           + 0.3 * np.sin(2 * np.pi * f2 * x))
    elif kind == "ripple":
        amp = amp or min(60.0, L * 0.08)
        off = amp * np.sin(2 * np.pi * x / 70.0) * np.exp(-3.2 * s) * (s > 0.02)
    elif kind == "route":
        amp = amp or min(140.0, L * 0.18)
        ph = r.random(3) * 6
        off = amp * env * (0.6 * np.sin(2 * np.pi * s * 1.3 + ph[0]) + 0.3 * np.sin(2 * np.pi * s * 2.7 + ph[1])
                           + 0.1 * np.sin(2 * np.pi * s * 5.1 + ph[2]))
    elif kind == "rise":
        amp = amp or min(160.0, L * 0.2)
        steps = np.cumsum(r.normal(0.6, 1.0, n)) / n * 6
        off = amp * (s ** 1.5 + 0.12 * np.sin(s * 19) * env) + steps * 8 * env
        off -= off[-1] * s  # still land exactly on p1
    elif kind == "spark":
        amp = amp or min(90.0, L * 0.12)
        off = amp * env * np.sign(np.sin(2 * np.pi * s * 4.5)) * np.abs(np.sin(2 * np.pi * s * 4.5)) ** 0.35 * 0.6
    elif kind == "circuit":
        return _circuit(p0, p1, u, nrm, L, seed, n)
    else:
        off = np.zeros_like(s)
    pts = p0 + np.outer(x, u) + np.outer(off, nrm)
    pts[0], pts[-1] = p0, p1
    return pts


def _circuit(p0, p1, u, nrm, L, seed, n):
    """Orthogonal PCB-style traces with 45-degree corners between p0 and p1."""
    r = np.random.default_rng(seed)
    d = p1 - p0
    horiz = abs(d[0]) >= abs(d[1])
    pts = [p0]
    cur = p0.copy()
    legs = 3
    for i in range(legs):
        rem = p1 - cur
        if horiz:
            step = rem[0] * (r.uniform(0.25, 0.45) if i < legs - 1 else 1.0)
            nxt = cur + np.array([step, 0.0])
            jog = rem[1] * (r.uniform(0.3, 0.6) if i < legs - 1 else 1.0)
        else:
            step = rem[1] * (r.uniform(0.25, 0.45) if i < legs - 1 else 1.0)
            nxt = cur + np.array([0.0, step])
            jog = rem[0] * (r.uniform(0.3, 0.6) if i < legs - 1 else 1.0)
        pts.append(nxt)
        cur = nxt
        if abs(jog) > 1 and i < legs - 1:
            diag = np.array([abs(jog), jog]) if horiz else np.array([jog, abs(jog)])
            if horiz:
                diag = np.array([math.copysign(abs(jog), d[0]), jog])
            else:
                diag = np.array([jog, math.copysign(abs(jog), d[1])])
            cur = cur + diag
            pts.append(cur)
    if np.linalg.norm(pts[-1] - p1) > 0.5:
        pts.append(p1)
    from .core import resample
    return resample(np.array(pts), n)


def beats(kind, p0, p1, seed=0, period=None):
    """Arc fractions of the motif's events (for sounds and flashes)."""
    _, _, _, _, L = _frame(p0, p1)
    if kind == "ecg":
        period = period or max(160.0, min(260.0, L / 2.6))
        r = np.random.default_rng(seed)
        phase = 0.5 + 0.25 * r.random()
        xs = [(k + phase) * period for k in range(int(L / period) + 2)]
        return [x / L for x in xs if 0.08 < x / L < 0.92]
    if kind == "circuit":
        return [0.22, 0.47, 0.72]
    if kind == "route":
        return [0.33, 0.66]
    if kind in ("wave", "spark"):
        return [0.25, 0.5, 0.75]
    return []
