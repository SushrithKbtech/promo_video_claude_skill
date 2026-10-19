"""Sound design: every visible move gets a sound.

Each generator returns (stereo_clip, offset) where offset is the time from the
clip start to the *event* (the peak of a whoosh, the end of a riser, the
transient of an impact). The mixer places clip at cue_time - offset.
"""
from __future__ import annotations

import math

import numpy as np

from .dsp import (N, SR, expdec, filt, noise, place, reverb, sat, saw, sine, square, stereo, sweep)


def _pan_sweep(m, p0, p1):
    n = len(m)
    p = np.linspace(p0, p1, n)
    a = (p + 1) * math.pi / 4
    return np.stack([m * np.cos(a), m * np.sin(a)], 1) * math.sqrt(2)


def whoosh(dur=0.7, peak=0.62, lo=260, hi=4200, pan=(-0.6, 0.6), seed=1, body=0.5, verb=0.18):
    n = N(dur)
    t = np.linspace(0, 1, n)
    env = np.where(t < peak, (t / peak) ** 2.4, np.exp(-(t - peak) / 0.11))
    f = np.where(t < peak, lo * (hi / lo) ** (t / peak), hi * (lo * 1.6 / hi) ** ((t - peak) / (1 - peak + 1e-6)))
    nz = noise(n, seed)
    a = sweep(nz, "bp", f, q=1.1)
    b = sweep(nz, "lp", f * 0.5, q=0.6) * body
    m = (a * 1.3 + b) * env
    st = _pan_sweep(m, *pan)
    st += reverb(st, 1.2, verb)
    return st * 0.9, dur * peak


def impact(size=1.0, seed=2, dark=False):
    dur = 2.6 * size
    n = N(dur)
    t = np.arange(n) / SR
    f_sub = 32 + 30 * np.exp(-t / 0.35)
    sub = sat(sine(f_sub, n) * np.exp(-t / (0.95 * size)), 1.8) * 0.9
    f_p = 48 + 110 * np.exp(-t / 0.03) + 500 * np.exp(-t / 0.002)
    punch = sat(sine(f_p, n) * np.exp(-t / 0.22), 2.2) * 0.8
    crack = filt(noise(n, seed) * expdec(n, 0.018), "hp", 1100) * 0.9
    rumble = filt(noise(n, seed + 1) * expdec(n, 0.45), "lp", 380) * 0.55
    m = sub + punch + crack + rumble
    st = stereo(m)
    st += reverb(stereo(crack + punch * 0.3), 3.0, 0.42, bright=0.5)
    if dark:
        st += braam([38, 45, 50], 1.8 * size, seed) * 0.5
    return st * 0.8, 0.0


def braam(notes, dur, seed=0):
    from .dsp import midi
    n = N(dur)
    t = np.arange(n) / SR
    m = np.zeros(n)
    for i, nt in enumerate(notes):
        for d in (-0.08, 0.0, 0.08):
            m += saw(midi(nt) * 2 ** (d / 12), n, (i * 0.3 + d) % 1)
    m /= len(notes) * 3
    env = np.clip(t / 0.04, 0, 1) * np.exp(-t / (dur * 0.45))
    cut = 180 + 1400 * np.clip(t / 0.25, 0, 1) * np.exp(-t / (dur * 0.6))
    m = sweep(sat(m * 2.5, 1.5), "lp", cut, q=0.9) * env
    return stereo(m) + reverb(stereo(m), 2.4, 0.25)


def transient(seed=3):
    n = N(0.12)
    t = np.arange(n) / SR
    m = sine(90 + 200 * np.exp(-t / 0.01), n) * np.exp(-t / 0.05) + filt(noise(n, seed) * expdec(n, 0.006), "hp", 2000)
    return stereo(sat(m, 1.4) * 0.8), 0.0


def ring(dur=2.5, f0=880.0):
    n = N(dur)
    t = np.arange(n) / SR
    m = np.zeros(n)
    for k, (r, a, tau) in enumerate(((1.0, 1, 1.2), (2.76, 0.5, 0.7), (5.4, 0.25, 0.4), (8.93, 0.12, 0.25))):
        m += sine(f0 * r, n, k * 0.13) * a * np.exp(-t / tau)
    m *= np.clip(t / 0.02, 0, 1) * (1 + 0.15 * np.sin(2 * np.pi * 5 * t))
    st = stereo(m * 0.25)
    return st + reverb(st, 2.5, 0.5), 0.0


def air(dur=0.6, seed=4):
    n = N(dur)
    t = np.linspace(0, 1, n)
    nz = noise(n, seed)
    m = sweep(nz, "bp", 500 * (5000 / 500) ** t, q=0.8) * (0.25 + 0.75 * t ** 2)
    m += sweep(nz, "lp", 300 * (2400 / 300) ** t, q=0.7) * 0.5 * t
    return _pan_sweep(m * 0.6, -0.3, 0.3), 0.0


def riser(dur=1.0, seed=5, tone=True):
    """Ends at the event (cut, never fading out)."""
    n = N(dur)
    t = np.linspace(0, 1, n)
    nz = noise(n, seed)
    m = sweep(nz, "bp", 400 * (9000 / 400) ** (t ** 1.3), q=1.4) * (t ** 2.2)
    if tone:
        f = 180 * 2 ** (2.2 * t ** 1.6)
        m += saw(f, n) * 0.18 * t ** 2 * (1 + 0.3 * np.sin(2 * np.pi * (2 + 14 * t) * t * 3))
    m[-N(0.004):] *= np.linspace(1, 0, N(0.004))
    st = stereo(m * 0.7)
    return st, dur


def keystroke(pitch=1.0, seed=6):
    n = N(0.11)
    t = np.arange(n) / SR
    r = np.random.default_rng(seed)
    down = filt(noise(n, seed) * expdec(n, 0.004), "bp", 3200 * pitch * r.uniform(0.9, 1.1), q=1.3)
    body = sine(190 * pitch, n) * np.exp(-t / 0.012) * 0.35
    up = np.zeros(n)
    j = N(0.055 + r.uniform(-0.01, 0.01))
    up[j:] = filt(noise(n - j, seed + 9) * expdec(n - j, 0.003), "bp", 4200 * pitch, q=1.5) * 0.35
    m = down + body + up
    return stereo(m * 0.9), 0.0


def tick(pitch=1.0):
    n = N(0.05)
    t = np.arange(n) / SR
    m = sine(2400 * pitch, n) * np.exp(-t / 0.006) + filt(noise(n, 7) * expdec(n, 0.001), "hp", 3000) * 0.5
    st = stereo(m * 0.6)
    return st + reverb(st, 0.6, 0.12), 0.0


def click():
    n = N(0.08)
    t = np.arange(n) / SR
    m = sine(1150, n) * np.exp(-t / 0.012) * 0.6 + filt(noise(n, 8) * expdec(n, 0.002), "bp", 2200, q=0.9)
    return stereo(m * 0.8), 0.0


def pop(pitch=1.0, soft=False):
    n = N(0.16)
    t = np.arange(n) / SR
    f = (420 + 520 * (1 - np.exp(-t / 0.012))) * pitch
    m = sine(f, n) * np.exp(-t / (0.04 if not soft else 0.03)) * np.clip(t / 0.002, 0, 1)
    st = stereo(m * (0.7 if not soft else 0.45))
    return st + reverb(st, 0.9, 0.15), 0.0


def pop_low():
    n = N(0.35)
    t = np.arange(n) / SR
    m = sine(70 + 120 * np.exp(-t / 0.04), n) * np.exp(-t / 0.12)
    m += filt(noise(n, 9) * expdec(n, 0.01), "lp", 900) * 0.3
    return stereo(sat(m, 1.3) * 0.8), 0.0


def bell(freq, dur=1.4, amp=1.0):
    n = N(dur)
    t = np.arange(n) / SR
    mod = sine(freq * 3.5, n) * 1.6 * np.exp(-t / 0.25)
    p = np.cumsum(np.full(n, freq) / SR) + mod / (2 * np.pi)
    m = np.sin(2 * np.pi * p) * np.exp(-t / (dur * 0.35)) * amp
    m += sine(freq * 2, n) * 0.25 * np.exp(-t / (dur * 0.15)) * amp
    return m * np.clip(t / 0.002, 0, 1)


def success():
    from .dsp import midi
    n = N(1.8)
    m = np.zeros(n)
    for i, note in enumerate((88, 92, 95)):
        j = N(0.045 * i)
        b = bell(midi(note), 1.6, 0.5 - 0.08 * i)
        m[j:j + len(b)] += b[:n - j]
    st = stereo(m * 0.55)
    return st + reverb(st, 1.8, 0.35), 0.0


def confirm():
    from .dsp import midi
    n = N(0.6)
    m = np.zeros(n)
    for i, note in enumerate((84, 91)):
        j = N(0.05 * i)
        b = bell(midi(note), 0.5, 0.4)
        m[j:j + len(b)] += b[:n - j]
    st = stereo(m * 0.5)
    return st + reverb(st, 1.2, 0.25), 0.0


def sparkle(seed=10):
    r = np.random.default_rng(seed)
    n = N(1.2)
    st = np.zeros((n, 2))
    for i in range(7):
        f = r.uniform(3000, 8000)
        j = N(r.uniform(0, 0.4))
        k = N(0.25)
        t = np.arange(k) / SR
        m = sine(f, k) * np.exp(-t / 0.06) * 0.18
        place(st, stereo(m, r.uniform(-0.8, 0.8)), j / SR)
    return st + reverb(st, 1.5, 0.4), 0.0


def shimmer(pitch=1.0):
    n = N(0.5)
    t = np.arange(n) / SR
    m = (sine(1760 * pitch, n) + sine(2637 * pitch, n) * 0.6) * np.sin(np.pi * t / t[-1]) ** 2 * 0.12
    m *= 1 + 0.4 * np.sin(2 * np.pi * 14 * t)
    st = stereo(m)
    return st + reverb(st, 1.4, 0.4), 0.0


def send():
    w, off = whoosh(0.4, 0.75, 600, 6500, (-0.2, 0.5), seed=12, body=0.2)
    p, _ = tick(1.1)
    place(w, p * 0.7, off)
    return w * 0.8, off


def swirl(dur=1.3, seed=13):
    n = N(dur)
    t = np.linspace(0, 1, n)
    nz = noise(n, seed)
    f = 600 * (1 + 0.8 * np.sin(2 * np.pi * 1.6 * t * dur)) * (1 + 2 * t)
    m = sweep(nz, "bp", f, q=1.6) * np.sin(np.pi * t) ** 1.5
    pan = np.sin(2 * np.pi * 1.1 * t * dur)
    a = (pan + 1) * math.pi / 4
    st = np.stack([m * np.cos(a), m * np.sin(a)], 1) * 1.1
    return st + reverb(st, 1.6, 0.3), 0.0


def digital(seed=14):
    n = N(0.35)
    t = np.arange(n) / SR
    f = 500 * (4.0 ** (t / t[-1]))
    m = square(f, n, pw=0.3) * np.exp(-t / 0.12) * 0.25
    step = 8
    m = np.repeat(m[::step], step)[:n]  # sample-rate reduction
    m = np.round(m * 24) / 24
    w, off = whoosh(0.35, 0.4, 900, 7000, (0.4, -0.4), seed=seed, body=0.1)
    st = stereo(m) * 0.7
    place(w, st, 0)
    return w, 0.0


def breath(dur=0.6, seed=15):
    st, _ = riser(dur, seed, tone=False)
    st = filt(st, "lp", 2500)
    return st * 0.6, 0.0


SFX = {
    "air": lambda c: air(c.get("dur", 0.6)),
    "riser_end": lambda c: riser(c.get("dur", 0.6)),
    "riser_short": lambda c: riser(c.get("dur", 0.8), seed=21, tone=True),
    "impact": lambda c: impact(c.get("size", 1.0), dark=c.get("dark", False)),
    "transient": lambda c: transient(),
    "ring": lambda c: ring(c.get("dur", 2.5)),
    "whoosh_soft": lambda c: whoosh(0.6, 0.6, 220, 2600, (-0.5, 0.4), seed=31, body=0.6),
    "whoosh_hit": lambda c: whoosh(0.55, 0.9, 300, 5200, (-0.4, 0.4), seed=32, body=0.5),
    "whoosh_short": lambda c: whoosh(0.32, 0.7, 500, 6000, (c.get("pan", 0) - 0.4, c.get("pan", 0) + 0.4), seed=33),
    "whoosh_fast": lambda c: whoosh(0.45, 0.85, 400, 7000, (-0.7, 0.7), seed=34, body=0.4),
    "swoosh": lambda c: whoosh(c.get("dur", 0.7), 0.55, 200, 3000, (-0.3, 0.3), seed=35, body=0.7, verb=0.25),
    "key": lambda c: keystroke(c.get("pitch", 1.0), seed=int(c.get("t", 0) * 997) % 50),
    "tick": lambda c: tick(c.get("pitch", 1.0)),
    "select": lambda c: tick(0.8),
    "click": lambda c: click(),
    "pop": lambda c: pop(c.get("pitch", 1.0)),
    "pop_soft": lambda c: pop(c.get("pitch", 1.2), soft=True),
    "pop_low": lambda c: pop_low(),
    "success": lambda c: success(),
    "confirm": lambda c: confirm(),
    "sparkle": lambda c: sparkle(),
    "shimmer": lambda c: shimmer(c.get("pitch", 1.0)),
    "send": lambda c: send(),
    "swirl": lambda c: swirl(c.get("dur", 1.3)),
    "digital": lambda c: digital(),
    "breath": lambda c: breath(c.get("dur", 0.6)),
}


def render_cues(cues, total_s, dark=False):
    out = np.zeros((N(total_s + 3), 2))
    for c in cues:
        fn = SFX.get(c["kind"])
        if fn is None:
            continue
        if c["kind"] == "impact" and dark:
            c = {**c, "dark": True}
        clip, off = fn(c)
        if clip.ndim == 1:
            clip = stereo(clip)
        pan = c.get("pan", 0.0)
        if pan and c["kind"] not in ("whoosh_short",):
            a = (pan + 1) * math.pi / 4
            clip = clip * np.array([math.cos(a), math.sin(a)]) * math.sqrt(2)
        place(out, clip, c["t"] - off, 10 ** (c.get("gain", -12) / 20))
    return out
