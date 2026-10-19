"""The score: a built-in composer that writes to the picture's beat grid.

Shape (from the method): quiet and airy under the hook, building from the opening hit,
the groove kicking in on the drop, a short thin breath before the logo hit,
everything back on the hit, then ringing out under the end card.

Every product gets its own song. A seed derived from the product picks the chord
progression, key, arpeggio pattern, lead instrument and drum variations from the
mood's banks, so two films in the same mood still sound different.

Moods
    hype       trailer-pop: 808 drop, trap hats, supersaw stabs            (energetic launches)
    uplifting  clean and bright, Apple-keynote feel, four-on-the-floor     (consumer, SaaS)
    dark       car-ad: half-time, distorted 808, toms, braams              (auto, gaming, security)
    cinematic  taiko ostinato, string runs, brass braams, no 808           (epic launches, hardware)
    chill      lo-fi: swung drums, FM electric piano, vinyl, warm sub      (lifestyle, notes, calm)
    house      four-on-the-floor, offbeat bass, organ stabs                (fashion, fintech, social)
    acoustic   plucked guitar, snaps, shaker, warm pad                     (health, education, food)
    minimal    glassy bells, clicks, sub pulses, lots of air               (dev tools, AI, B2B)
    epic       alias of cinematic

Or bring a real track: brief.music.file (+ optional drop_at); see audio.py.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np
from scipy import signal

from .dsp import (N, SR, adsr, compress, expdec, filt, midi, noise, normalize_lufs, pingpong, place, reverb, sat,
                  saw, sine, square, stereo, supersaw, sweep)

# ---------------------------------------------------------------------------
# harmony banks (voicings around C / A minor; transposed per product)
# ---------------------------------------------------------------------------

MAJOR = [  # (chords, bass pitch classes)
    ([[60, 64, 67], [59, 62, 67], [57, 60, 64], [57, 60, 65]], [0, 7, 9, 5]),        # I V vi IV
    ([[57, 60, 64], [57, 60, 65], [55, 60, 64], [55, 59, 62]], [9, 5, 0, 7]),        # vi IV I V
    ([[57, 60, 65], [55, 60, 64], [55, 59, 62], [57, 60, 64]], [5, 0, 7, 9]),        # IV I V vi
    ([[60, 64, 67], [57, 60, 64], [57, 60, 65], [55, 59, 62]], [0, 9, 5, 7]),        # I vi IV V
    ([[60, 64, 67], [59, 64, 67], [57, 60, 65], [55, 59, 62]], [0, 4, 5, 7]),        # I iii IV V
    ([[57, 60, 65], [55, 59, 62], [57, 60, 64], [55, 60, 64]], [5, 7, 9, 0]),        # IV V vi I
]
MINOR = [
    ([[57, 60, 64], [57, 60, 65], [55, 60, 64], [55, 59, 62]], [9, 5, 0, 7]),        # i VI III VII
    ([[57, 60, 64], [57, 60, 65], [57, 62, 65], [56, 59, 64]], [9, 5, 2, 4]),        # i VI iv V
    ([[57, 60, 64], [57, 62, 65], [57, 60, 65], [56, 59, 64]], [9, 2, 5, 4]),        # i iv VI V
    ([[57, 60, 64], [55, 59, 62], [57, 60, 65], [55, 59, 62]], [9, 7, 5, 7]),        # i VII VI VII
    ([[57, 60, 64], [55, 60, 64], [55, 59, 62], [57, 62, 65]], [9, 0, 7, 2]),        # i III VII iv
    ([[57, 60, 64], [57, 60, 64], [57, 60, 65], [55, 59, 62]], [9, 9, 5, 7]),        # i i VI VII (pedal)
]
SEVENTHS = [
    ([[57, 60, 62, 65], [55, 59, 62, 65], [55, 59, 60, 64], [55, 57, 60, 64]], [2, 7, 0, 9]),   # ii7 V7 Imaj7 vi7
    ([[57, 60, 64, 65], [55, 59, 62, 64], [57, 60, 64, 67], [57, 60, 62, 65]], [5, 4, 9, 2]),   # IVmaj7 iii7 vi7 ii7
    ([[59, 62, 64, 67], [57, 60, 64, 65], [55, 59, 62, 65], [55, 59, 60, 64]], [4, 5, 7, 0]),   # iii7 IVmaj7 V7 Imaj7
    ([[57, 60, 64, 67], [57, 60, 62, 65], [55, 59, 62, 65], [55, 59, 60, 64]], [9, 2, 7, 0]),   # vi7 ii7 V7 Imaj7
]

STYLES = {
    "hype": dict(bank="minor", kit="trap", bass="808", pad="saw", leads=["pluck", "bell"], stabs=True,
                 arp_rates=[1, 2], roll="snare", gap=0.11),
    "uplifting": dict(bank="major", kit="pop4", bass="offbeat", pad="saw", leads=["pluck", "keys", "bell"],
                      stabs=False, arp_rates=[2, 1], roll="snare", gap=0.11),
    "dark": dict(bank="minor", kit="halftime", bass="808dist", pad="dark", leads=["pulse"], stabs=True,
                 arp_rates=[1], roll="toms", gap=0.12, braams=True),
    "cinematic": dict(bank="minor", kit="taiko", bass="sub", pad="strings", leads=["strings"], stabs=False,
                      arp_rates=[1], roll="toms", gap=0.14, braams=True),
    "chill": dict(bank="sevenths", kit="lofi", bass="sub", pad="warm", leads=["keys"], stabs=False,
                  arp_rates=[2, 4], roll=None, gap=0.0, swing=0.58, vinyl=True),
    "house": dict(bank="major", kit="house", bass="offbeat", pad="saw", leads=["keys", "pluck"], stabs=True,
                  arp_rates=[2], roll="snare", gap=0.1, organ=True),
    "acoustic": dict(bank="major", kit="acoustic", bass="sub", pad="warm", leads=["guitar"], stabs=False,
                     arp_rates=[2, 1], roll=None, gap=0.0, swing=0.54),
    "minimal": dict(bank="minor", kit="minimal", bass="sub", pad="glass", leads=["glass"], stabs=False,
                    arp_rates=[4, 2], roll=None, gap=0.08),
}
ALIASES = {"epic": "cinematic", "lofi": "chill", "lo-fi": "chill", "edm": "hype", "trap": "hype",
           "corporate": "uplifting", "apple": "uplifting", "car": "dark", "folk": "acoustic", "ambient": "minimal",
           "tech": "minimal"}

ARP_PATTERNS = [[0, 1, 2, 3], [0, 2, 1, 3], [0, 1, 2, 3, 2, 1], [0, 2, 3, 1], [3, 2, 1, 0], [0, 3, 1, 2], [0, 0, 2, 3]]


def bass_freq(pc, lo=41.0):
    f = midi(24 + pc)
    while f < lo:
        f *= 2
    while f >= lo * 2:
        f /= 2
    return f


def seed_of(*parts):
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return int(h[:8], 16)


# ---------------------------------------------------------------------------
# instruments
# ---------------------------------------------------------------------------

def _peak1(x):
    p = np.abs(x).max()
    return x / p if p > 1e-9 else x


KIT = dict(kick_f=46.0, kick_dec=0.32, snare_p=1.0, hat_f=9500.0, clap_f=1300.0, hat_dec=0.028)


def kick(vel=1.0, hard=False, soft=False):
    n = N(0.5)
    t = np.arange(n) / SR
    kf = KIT["kick_f"]
    f = kf + (80 if soft else 120) * np.exp(-t / 0.028) + (200 if soft else 600) * np.exp(-t / 0.0025)
    m = sine(f, n) * np.exp(-t / (KIT["kick_dec"] if not hard else KIT["kick_dec"] + 0.08))
    if not soft:
        m += filt(noise(n, 1) * expdec(n, 0.0015), "bp", 3500, q=0.8) * 0.4
    return sat(m * 1.4, 1.6) * vel


def clap(vel=1.0, seed=2):
    n = N(0.45)
    t = np.arange(n) / SR
    nz = noise(n, seed)
    env = np.zeros(n)
    for d in (0.0, 0.009, 0.019, 0.028):
        j = N(d)
        env[j:] += np.exp(-(t[:n - j]) / 0.005) * (0.8 if d < 0.028 else 1.0)
    env += np.exp(-np.maximum(0, t - 0.028) / 0.14) * (t > 0.028) * 0.55
    m = filt(filt(nz * env, "bp", KIT["clap_f"], q=0.7), "hp", 500)
    return m * 1.6 * vel


def snap(vel=1.0, seed=3):
    n = N(0.2)
    t = np.arange(n) / SR
    m = filt(noise(n, seed) * np.exp(-t / 0.02), "bp", 2400, q=1.2) + sine(1800, n) * np.exp(-t / 0.008) * 0.3
    return _peak1(m) * vel


def rim(vel=1.0):
    n = N(0.12)
    t = np.arange(n) / SR
    m = sine(1700, n) * np.exp(-t / 0.012) + filt(noise(n, 5) * expdec(n, 0.004), "bp", 3200, q=2) * 0.6
    return _peak1(m) * vel


def snare(vel=1.0, pitch=1.0, seed=3, tail=0.18):
    n = N(0.6)
    t = np.arange(n) / SR
    pitch = pitch * KIT["snare_p"]
    body = (sine(185 * pitch, n) * 0.7 + sine(330 * pitch, n) * 0.35) * np.exp(-t / 0.06)
    nz = filt(noise(n, seed), "hp", 1400) * np.exp(-t / tail)
    return sat((body + nz * 0.9) * vel, 1.3)


_HAT_F = (205.3, 304.4, 369.6, 522.7, 540.0, 800.0)


def hat(vel=1.0, open_=False):
    n = N(0.4 if open_ else 0.09)
    t = np.arange(n) / SR
    m = sum(square(f * 2.2, n, i * 0.17) for i, f in enumerate(_HAT_F)) / 6
    m = filt(filt(m, "bp", KIT["hat_f"], q=0.9), "hp", 7000, order=2)
    m *= np.exp(-t / (0.16 if open_ else KIT["hat_dec"])) * np.clip(t / 0.0008, 0, 1)
    return _peak1(m) * 0.9 * vel


def shaker(vel=1.0, seed=4):
    n = N(0.09)
    t = np.arange(n) / SR
    env = np.clip(t / 0.012, 0, 1) * np.exp(-t / 0.03)
    m = filt(noise(n, seed) * env, "bp", 6500, q=0.9)
    return _peak1(m) * 0.7 * vel


def crash(vel=1.0, dur=2.2):
    n = N(dur)
    t = np.arange(n) / SR
    m = sum(square(f * 3.1, n, i * 0.11) for i, f in enumerate(_HAT_F)) / 6 * 0.6 + noise(n, 4) * 0.5
    m = filt(m, "hp", 4500, order=2) * np.exp(-t / (dur * 0.32)) * np.clip(t / 0.002, 0, 1)
    return m * vel


def tom(freq=90.0, vel=1.0, dur=0.6):
    n = N(dur)
    t = np.arange(n) / SR
    m = sine(freq * (1 + 0.6 * np.exp(-t / 0.04)), n) * np.exp(-t / (dur * 0.37))
    m += filt(noise(n, 5) * expdec(n, 0.02), "lp", 1800) * 0.3
    return sat(m * 1.3 * vel, 1.4)


def taiko(vel=1.0, low=True):
    n = N(0.9)
    t = np.arange(n) / SR
    f0 = 62 if low else 95
    m = sine(f0 * (1 + 0.35 * np.exp(-t / 0.05)), n) * np.exp(-t / 0.3)
    m += filt(noise(n, 8) * expdec(n, 0.03), "lp", 900) * 0.5
    return sat(m * 1.5 * vel, 1.5)


def bass808(f0, f1, dur, vel=1.0, drive=2.4, glide=0.06):
    n = N(dur + 0.08)
    t = np.arange(n) / SR
    f = f1 + (f0 - f1) * np.exp(-t / glide) if f0 != f1 else np.full(n, f1)
    f = f * (1 + 0.5 * np.exp(-t / 0.012))
    m = sine(f, n)
    env = np.clip(t / 0.004, 0, 1) * np.where(t < dur, np.exp(-t / (dur * 1.4 + 0.2)), 0)
    rel = (t >= dur)
    env[rel] = np.exp(-(t[rel] - dur) / 0.03) * (np.exp(-dur / (dur * 1.4 + 0.2)))
    m = sat(m * env * drive, 1.0)
    grit = filt(sat(m * 3.0, 1.0) - m * 0.8, "bp", 700, q=0.7) * 0.35
    return filt(m + grit, "lp", 4200) * vel


def sub_note(f, dur, vel=1.0):
    n = N(dur + 0.1)
    m = sine(f, n) + 0.25 * sine(f * 2, n) + 0.08 * sine(f * 3, n)
    env = adsr(n, 0.01, 0.2, 0.8, 0.12, gate=dur)
    return sat(m * env * 1.2, 1.1) * vel


def stab(notes, dur, vel=1.0, bright=1.0, seed=0, organ=False):
    n = N(dur + 0.25)
    t = np.arange(n) / SR
    out = np.zeros((n, 2))
    if organ:
        for i, nt in enumerate(notes + [notes[0] + 12]):
            f = midi(nt)
            m = sine(f, n) + 0.5 * sine(f * 2, n) + 0.25 * sine(f * 3, n) + 0.12 * square(f * 4, n, 0.2)
            out += stereo(m, -0.3 + 0.2 * i)
    else:
        for i, nt in enumerate(notes + [notes[0] + 12]):
            out += supersaw(midi(nt), n, 7, 0.2, seed + i)
    out /= len(notes) + 1
    env = np.clip(t / 0.003, 0, 1) * np.where(t < dur, np.exp(-t / (dur * 2.0)), 0)
    rel = t >= dur
    env[rel] = np.exp(-(t[rel] - dur) / 0.06) * np.exp(-dur / (dur * 2.0))
    cut = 900 + 6500 * bright * np.exp(-t / 0.12)
    out = sweep(out * env[:, None], "lp", cut, q=0.9)
    return _peak1(filt(out, "hp", 180)) * vel


def pad(notes, dur, cut0=600.0, cut1=2400.0, vel=1.0, seed=0, kind="saw"):
    n = N(dur + 0.6)
    out = np.zeros((n, 2))
    for i, nt in enumerate(notes):
        f = midi(nt)
        if kind in ("saw", "dark"):
            out += supersaw(f, n, 5, 0.12 if kind == "saw" else 0.2, seed + i) * 0.8
            out += stereo(sine(midi(nt - 12), n) * 0.35)
        elif kind == "strings":
            out += supersaw(f, n, 7, 0.09, seed + i) * 0.7
            out += supersaw(f / 2, n, 5, 0.06, seed + 7 + i) * 0.4
        elif kind == "warm":
            ph = 2 * np.pi * np.cumsum(np.full(n, f)) / SR + i
            tri = (2 / np.pi) * np.arcsin(np.sin(ph))
            out += stereo(tri * 0.6 + sine(f, n) * 0.4, -0.4 + 0.3 * i)
        else:  # glass
            out += stereo(sine(f * 2, n) * 0.5 + sine(f * 4.01, n) * 0.18 + sine(f * 2.997, n) * 0.12, -0.5 + 0.35 * i)
    out /= len(notes)
    att = {"strings": 0.6, "warm": 0.25, "glass": 0.4}.get(kind, 0.35)
    env = adsr(n, att, 0.4, 0.85, 0.6, gate=dur)
    cut = np.geomspace(max(cut0, 50), max(cut1, 50), n)
    if kind == "dark":
        cut = cut * 0.6
    out = sweep(out * env[:, None], "lp", cut, q=0.6)
    return _peak1(filt(out, "hp", 120)) * vel


def pluck(note, dur=0.22, vel=1.0, bright=1.0):
    n = N(dur + 0.2)
    t = np.arange(n) / SR
    f = midi(note)
    m = saw(f, n) * 0.6 + square(f * 2, n, 0.1, 0.35) * 0.25
    env = np.exp(-t / (dur * 0.6)) * np.clip(t / 0.002, 0, 1)
    cut = 600 + 5200 * bright * np.exp(-t / 0.07)
    m = sweep(m * env, "lp", cut, q=1.2)
    return _peak1(m) * vel


def keys(note, dur=0.5, vel=1.0, bright=1.0):
    """FM electric piano (tine-like)."""
    n = N(dur + 0.6)
    t = np.arange(n) / SR
    f = midi(note)
    idx = (1.6 + 1.2 * bright) * np.exp(-t / 0.35)
    mod = np.sin(2 * np.pi * f * t) * idx
    m = np.sin(2 * np.pi * f * t + mod) * np.exp(-t / (0.9 + dur))
    m += 0.12 * np.sin(2 * np.pi * f * 14 * t) * np.exp(-t / 0.02)  # tine click
    m *= 1 + 0.12 * np.sin(2 * np.pi * 5.2 * t)
    return _peak1(m * np.clip(t / 0.002, 0, 1)) * vel


def bellnote(note, dur=0.8, vel=1.0, bright=1.0):
    from .sfx import bell
    return _peak1(bell(midi(note), dur + 0.6, 1.0)) * vel


def guitar(note, dur=0.6, vel=1.0, bright=1.0, seed=0):
    """Karplus-Strong plucked string."""
    f = midi(note)
    n = N(dur + 1.2)
    Nd = max(2, int(round(SR / f)))
    burst = np.random.default_rng(seed + int(note)).uniform(-1, 1, Nd)
    burst = filt(burst, "lp", 2500 + 4000 * bright)
    x = np.zeros(n)
    x[:Nd] = burst
    g = 0.996
    a = np.zeros(Nd + 2)
    a[0] = 1.0
    a[Nd] = -g * 0.5
    a[Nd + 1] = -g * 0.5
    y = signal.lfilter([1.0], a, x)
    y = filt(y, "hp", 90)
    return _peak1(y) * vel


def glass(note, dur=0.8, vel=1.0, bright=1.0):
    n = N(dur + 1.0)
    t = np.arange(n) / SR
    f = midi(note)
    m = (sine(f, n) + 0.35 * sine(f * 2.76, n) * np.exp(-t / 0.25) + 0.15 * sine(f * 5.4, n) * np.exp(-t / 0.1))
    return _peak1(m * np.exp(-t / (0.5 + dur)) * np.clip(t / 0.003, 0, 1)) * vel


def pulse_note(note, dur, vel=1.0, cut=900):
    n = N(dur + 0.05)
    t = np.arange(n) / SR
    m = (saw(midi(note), n) + saw(midi(note) * 1.004, n, 0.3)) * 0.5
    env = np.clip(t / 0.004, 0, 1) * np.exp(-t / (dur * 0.9))
    m = filt(m * env, "lp", cut, q=1.4)
    return _peak1(m) * vel


def string_note(note, dur, vel=1.0, cut=1800):
    n = N(dur + 0.08)
    t = np.arange(n) / SR
    out = supersaw(midi(note), n, 5, 0.08, int(note))
    env = np.clip(t / 0.01, 0, 1) * np.exp(-t / (dur * 1.2))
    return _peak1(filt(out * env[:, None], "lp", cut, q=0.8)) * vel


def braam(notes, dur, seed=0):
    from .sfx import braam as _braam
    return _braam(notes, dur, seed)


def rev_cymbal(dur):
    c = _peak1(crash(1.0, dur + 0.4))[::-1][-N(dur):]
    return c * np.linspace(0.2, 1.0, len(c)) ** 2


LEADS = {"pluck": pluck, "keys": keys, "bell": bellnote, "guitar": guitar, "glass": glass}


# ---------------------------------------------------------------------------
# the arrangement
# ---------------------------------------------------------------------------

class Score:
    def __init__(self, plan, mood="hype", seed=0, transpose=None):
        self.p = plan
        mood = ALIASES.get(mood, mood)
        self.mood = mood if mood in STYLES else "hype"
        self.st = STYLES[self.mood]
        r = np.random.default_rng(seed)
        bank = {"major": MAJOR, "minor": MINOR, "sevenths": SEVENTHS}[self.st["bank"]]
        chords, bass = bank[int(r.integers(len(bank)))]
        k = int(transpose) if transpose is not None else int(r.integers(-4, 6))
        self.key_offset = k
        self.prog = dict(chords=[[n_ + k for n_ in ch] for ch in chords], bass=[(pc + k) % 12 for pc in bass])
        self.lead = self.st["leads"][int(r.integers(len(self.st["leads"])))]
        self.arp = ARP_PATTERNS[int(r.integers(len(ARP_PATTERNS)))]
        self.arp_rate = self.st["arp_rates"][int(r.integers(len(self.st["arp_rates"])))]
        self.var = int(r.integers(0, 4))
        # every product gets its own drum kit tuning
        self.kit = dict(kick_f=float(r.uniform(42, 56)), kick_dec=float(r.uniform(0.26, 0.42)),
                        snare_p=float(r.uniform(0.85, 1.15)), hat_f=float(r.uniform(8000, 11000)),
                        clap_f=float(r.uniform(1000, 1700)), hat_dec=float(r.uniform(0.02, 0.04)))
        KIT.update(self.kit)
        self.octave = int(r.choice([12, 12, 24])) if self.mood not in ("acoustic", "chill") else 12
        self.bar = 240.0 / plan["bpm"]
        self.step = self.bar / 16
        self.swing = self.st.get("swing", 0.5)
        self.total = plan["end"] + 0.5
        n = N(self.total + 3)
        self.stems = {k_: np.zeros((n, 2)) for k_ in ("drums", "bass", "chords", "pad", "arp", "fx")}
        self.kicks = []
        self.seed = seed

    def describe(self):
        return dict(mood=self.mood, key_offset=self.key_offset, lead=self.lead, arp=self.arp,
                    arp_rate=self.arp_rate, drum_variation=self.var,
                    kit={k: round(v, 3) for k, v in self.kit.items()})

    def st_t(self, tb, s):
        """Time of 16th step s in the bar starting at tb, with swing on the off 8ths."""
        t = tb + s * self.step
        if self.swing != 0.5 and s % 4 == 2:
            t += (self.swing - 0.5) * 2 * self.step
        return t

    def chord_at(self, t):
        k = int(math.floor((t - self.p["slam"]) / self.bar + 1e-6))
        return k % len(self.prog["chords"])

    def add(self, stem, clip, t, gain=1.0, pan=0.0):
        if clip.ndim == 1:
            clip = stereo(clip, pan)
        place(self.stems[stem], clip, t, gain)

    def lead_note(self, note, dur, vel, bright, s=0):
        if self.lead == "guitar":
            return guitar(note, dur, vel, bright, seed=s)
        fn = LEADS.get(self.lead, pluck)
        return fn(note, dur, vel, bright)

    # -- sections ------------------------------------------------------------
    def compose(self):
        p = self.p
        self.build(p["slam"], p["drop"])
        self.groove(p["drop"], p["breath"])
        self.breathe(p["breath"], p["hit"])
        self.finale(p["hit"], p["end"])
        return self

    def build(self, t0, t1):
        st = self.st
        bars = max(1, int(round((t1 - t0) / self.bar)))
        for b in range(bars):
            tb = t0 + b * self.bar
            ch = self.prog["chords"][self.chord_at(tb + 0.01)]
            prog = (b + 1) / bars
            self.add("pad", pad(ch, self.bar, 400 + 900 * b / bars, 900 + 2600 * prog, 0.9, b, st["pad"]), tb, 0.8)
            self._arp_bar(tb, ch, t1, 0.4 + 0.6 * prog, build=True)
            if prog > 0.5 or bars == 1:
                self._drums_bar(tb, t1, light=True, b=b)
        roll = st.get("roll")
        if roll:
            roll0 = t1 - self.bar
            t = roll0
            while t < t1 - self.step * 0.6:
                rel = (t - roll0) / self.bar
                dt = self.step * (2 if rel < 0.5 else 1 if rel < 0.85 else 0.5)
                if roll == "snare":
                    self.add("drums", snare(0.25 + 0.6 * rel, 1 + 0.35 * rel, int(t * 100)), t, 0.55 + 0.4 * rel)
                else:
                    self.add("drums", tom(70 + 60 * rel, 0.4 + 0.6 * rel, 0.4), t, 0.5 + 0.4 * rel)
                t += dt
        self.add("fx", rev_cymbal(self.bar / 2), t1 - self.bar / 2 - 0.12, 0.45 if roll else 0.3)

    def _arp_bar(self, tb, ch, t_end, bright, build=False, gain=None):
        tones = ch + [ch[0] + 12]
        rate = self.arp_rate
        if self.mood == "dark":
            for s in range(16):
                ts = tb + s * self.step
                if ts < t_end - self.step * 0.5:
                    self.add("arp", pulse_note(ch[0] - 12, self.step * 0.9, 0.9, 500 + 1600 * bright), ts, 0.55)
            return
        if self.mood == "cinematic":
            for s in range(16):
                ts = tb + s * self.step
                if ts < t_end - self.step * 0.5:
                    nt = tones[self.arp[s % len(self.arp)] % len(tones)]
                    self.add("arp", string_note(nt, self.step * 0.85, 0.8 + 0.2 * (s % 4 == 0), 1200 + 1800 * bright),
                             ts, 0.5, pan=0.3 * math.sin(s))
            return
        for j, s in enumerate(range(0, 16, rate)):
            ts = self.st_t(tb, s)
            if ts >= t_end - self.step * 0.5:
                break
            nt = tones[self.arp[j % len(self.arp)] % len(tones)] + self.octave
            dur = self.step * rate * (1.6 if self.lead in ("keys", "guitar", "glass", "bell") else 1.0)
            g = gain if gain is not None else (0.5 if build else 0.3)
            self.add("arp", self.lead_note(nt, dur, 0.75, bright, s=j), ts, g, pan=0.3 * math.sin(j * 1.3))

    def _drums_bar(self, tb, t_end, light=False, b=0):
        kit, v = self.st["kit"], self.var
        hits = []
        if kit == "trap":
            kp = [[0, 6, 10], [0, 7, 10, 13], [0, 3, 10], [0, 6, 9, 14]][v] if not light else [0, 8]
            if b % 4 == 3 and not light:
                kp = sorted(set(kp + [13]))
            hits += [("k", s) for s in kp]
            if not light:
                hits += [("c", s) for s in ([4, 12] if v % 2 == 0 else [8])]
            hits += [("h", s) for s in range(0, 16, 1 if not light else 2)]
            if not light and b % 2 == 1:
                hits += [("h32", 14), ("h32", 15)]
        elif kit == "pop4":
            hits += [("k", s) for s in ([0, 4, 8, 12] if v != 3 else [0, 4, 8, 10, 12])]
            if not light:
                hits += [("c", 4), ("c", 12)]
                hits += [("oh", s) for s in (2, 6, 10, 14)]
            hits += [("h", s) for s in range(0, 16, 2)]
        elif kit == "halftime":
            hits += [("k", s) for s in ([0, 10] if b % 2 == 0 else [0, 7, 10])]
            if not light:
                hits += [("sn", 8)]
            hits += [("h", s) for s in range(0, 16, 2)]
        elif kit == "taiko":
            pat = [[0, 3, 6, 8, 11, 14], [0, 2, 4, 8, 10, 12, 14], [0, 3, 8, 11, 12, 14]][v % 3]
            hits += [("tk", s) for s in pat]
            if not light and b % 2 == 1:
                hits += [("sn", 8)]
        elif kit == "lofi":
            hits += [("ks", s) for s in ([0, 7, 10] if v % 2 == 0 else [0, 6, 9])]
            hits += [("rim", s) for s in (4, 12)]
            hits += [("h", s) for s in range(0, 16, 2)]
            if not light:
                hits += [("sh", s) for s in range(1, 16, 2)]
        elif kit == "house":
            hits += [("k", s) for s in (0, 4, 8, 12)]
            if not light:
                hits += [("c", 4), ("c", 12)]
                hits += [("sh", s) for s in range(16)]
            hits += [("oh", s) for s in (2, 6, 10, 14)]
        elif kit == "acoustic":
            hits += [("ks", s) for s in ([0, 8] if v % 2 == 0 else [0, 7, 10])]
            if not light:
                hits += [("snap", s) for s in (4, 12)]
            hits += [("sh", s) for s in range(16)]
        elif kit == "minimal":
            hits += [("ks", s) for s in (0, 8)]
            rr = np.random.default_rng(self.seed + b)
            hits += [("rim", s) for s in (6, 14)]
            hits += [("click", int(s)) for s in rr.choice(16, 3 + v, replace=False)]
        g_light = 0.75 if light else 1.0
        for kind, s in hits:
            ts = self.st_t(tb, s)
            if ts >= t_end - 0.01:
                continue
            if kind == "k":
                self.add("drums", kick(1.0, self.mood == "dark"), ts, g_light)
                self.kicks.append(ts)
            elif kind == "ks":
                self.add("drums", kick(0.9, soft=True), ts, g_light * 0.9)
                self.kicks.append(ts)
            elif kind == "c":
                self.add("drums", clap(1.0, b + s), ts, 0.85 * g_light)
            elif kind == "sn":
                self.add("drums", snare(1.0, 0.85, b, tail=0.3), ts, 0.95 * g_light)
            elif kind == "h":
                self.add("drums", hat(0.75 if s % 4 == 0 else 0.5), ts, 0.42 * g_light, 0.25)
            elif kind == "h32":
                self.add("drums", hat(0.4), ts + self.step / 2, 0.3, 0.25)
            elif kind == "oh":
                self.add("drums", hat(0.8, True), ts, 0.45 * g_light, -0.2)
            elif kind == "tk":
                self.add("drums", taiko(0.9 if s % 8 == 0 else 0.6, low=s % 8 == 0), ts, 0.9 * g_light)
                if s % 8 == 0:
                    self.kicks.append(ts)
            elif kind == "rim":
                self.add("drums", rim(0.7), ts, 0.5 * g_light, -0.15)
            elif kind == "sh":
                self.add("drums", shaker(0.8 if s % 4 == 2 else 0.5, s + b), ts, 0.38 * g_light, 0.35)
            elif kind == "snap":
                self.add("drums", snap(1.0, b + s), ts, 0.7 * g_light, 0.1)
            elif kind == "click":
                self.add("drums", rim(0.35), ts, 0.3, 0.6 * math.sin(s))

    def groove(self, t0, t1):
        st = self.st
        bars = int(math.ceil((t1 - t0) / self.bar - 1e-6))
        if st.get("roll"):
            self.add("drums", crash(1.0), t0, 0.55)
        prev_bass = None
        for b in range(bars):
            tb = t0 + b * self.bar
            ci = self.chord_at(tb + 0.01)
            ch = self.prog["chords"][ci]
            bf = bass_freq(self.prog["bass"][ci])
            self.add("pad", pad(ch, self.bar, 1600, 2600, 0.7, b + 10, st["pad"]), tb, 0.55)
            k0 = len(self.kicks)
            self._drums_bar(tb, t1, light=False, b=b)
            bk = st["bass"]
            if bk == "offbeat":
                note = int(round(12 * math.log2(bf * 2 / 8.1758)))
                for s in (2, 6, 10, 14):
                    ts = tb + s * self.step
                    if ts < t1:
                        self.add("bass", pluck(note, 0.18, 0.9, 0.5), ts, 0.6)
                self.add("bass", sub_note(bf, self.bar * 0.9, 0.9), tb, 0.9)
            elif bk in ("808", "808dist"):
                kp = sorted(set(int(round((k - tb) / self.step)) for k in self.kicks[k0:]))
                kp = kp or [0]
                for j, s in enumerate(kp):
                    ts = tb + s * self.step
                    nxt = kp[j + 1] if j + 1 < len(kp) else 16
                    dur = max(self.step, (nxt - s) * self.step * 0.95)
                    f0 = prev_bass if (prev_bass and j == 0) else bf
                    if ts < t1:
                        self.add("bass", bass808(f0, bf, dur, 1.0, 3.0 if bk == "808dist" else 2.4), ts, 1.0)
                prev_bass = bf
            else:
                self.add("bass", sub_note(bf, self.bar * 0.95, 1.0), tb, 1.0)
                if self.mood in ("chill", "acoustic"):
                    self.add("bass", sub_note(bf * 1.5, self.step * 3, 0.6), tb + 10 * self.step, 0.7)
            if st.get("stabs"):
                spat = [0, 3, 6, 8, 11, 14] if self.mood == "hype" else [0] if self.mood == "dark" else [2, 6, 10, 14]
                for s in spat:
                    ts = tb + s * self.step
                    if ts < t1:
                        self.add("chords", stab(ch, self.step * (1.6 if self.mood != "dark" else 6), 1.0, 1.0, b + s,
                                                organ=st.get("organ", False)), ts, 0.55)
            elif self.mood == "uplifting":
                self.add("chords", stab(ch, self.bar * 0.92, 0.9, 0.7, b), tb, 0.45)
            elif self.mood in ("chill", "acoustic", "minimal"):
                for i, nt in enumerate(ch):
                    strum = 0.014 if self.mood == "acoustic" else 0.006 if self.mood == "chill" else 0.0
                    self.add("chords", self.lead_note(nt, self.bar * 0.8, 0.6, 0.6, s=b * 7 + i), tb + i * strum, 0.35)
            if st.get("braams") and b % 2 == 0:
                self.add("fx", braam([ch[0] - 24, ch[0] - 12, ch[0] - 17], self.bar * 0.9, b), tb, 0.6)
            self._arp_bar(tb, ch, t1, 0.8, gain=0.28)
        if st.get("vinyl"):
            n = N(t1 - t0)
            nz = noise(n, 99)
            crackle = (np.abs(nz) > 3.2) * nz * 0.3 + filt(nz, "lp", 3000) * 0.02
            self.add("fx", stereo(crackle), t0, 0.5)

    def breathe(self, t0, t1):
        ch = self.prog["chords"][self.chord_at(t0 + 0.01)]
        self.add("pad", pad(ch, t1 - t0, 350, 900, 0.7, 77, self.st["pad"]), t0, 0.5)
        r = filt(stereo(noise(N(t1 - t0), 9)), "bp", 2500, q=0.6) * np.linspace(0, 0.25, N(t1 - t0))[:, None]
        self.add("fx", r, t0, 0.6)

    def finale(self, t0, end):
        ch0 = self.prog["chords"][0]
        bf = bass_freq(self.prog["bass"][0])
        st = self.st
        if self.p.get("soft_hit"):
            # understated endings: no crash, no kick; a warm chord and a lifted lead
            self.add("bass", sub_note(bf, 2.2, 0.8), t0, 0.8)
            self.add("pad", pad(ch0 + [ch0[0] + 12], end - t0 + 0.5, 900, 2400, 0.9, 123, st["pad"]), t0, 0.8)
            for i, nt in enumerate(ch0 + [ch0[0] + 12]):
                self.add("chords", self.lead_note(nt + 12, 1.2, 0.6, 0.6, s=i), t0 + i * 0.06, 0.4,
                         pan=-0.3 + 0.2 * i)
            for i, nt in enumerate(ch0 + [ch0[0] + 12, ch0[1] + 12]):
                self.add("arp", self.lead_note(nt + 12, 0.5, 0.45, 0.5, s=i),
                         t0 + 1.2 + i * self.step * 1.5, 0.28, pan=-0.4 + 0.2 * i)
            return
        self.add("drums", crash(1.0, 2.6), t0, 0.55)
        self.add("drums", taiko(1.0) if st["kit"] == "taiko" else kick(1.0, True), t0, 1.0)
        self.kicks.append(t0)
        if st["bass"] in ("808", "808dist"):
            self.add("bass", bass808(bf * 2, bf, 1.6, 1.0, 2.6, glide=0.09), t0, 1.0)
        else:
            self.add("bass", sub_note(bf, 1.8, 1.0), t0, 1.0)
        if st.get("stabs") or self.mood == "uplifting":
            self.add("chords", stab(ch0, 0.9, 1.0, 1.0, 99, organ=st.get("organ", False)), t0, 0.6)
        if st.get("braams"):
            self.add("fx", braam([ch0[0] - 24, ch0[0] - 12, ch0[0] - 17], 2.2, 5), t0, 0.7)
        one = min(t0 + self.bar, end - 1.0)
        if one - t0 > self.bar * 0.6 and st["kit"] != "minimal":
            self._drums_bar(t0, one, light=True, b=1)
        self.add("pad", pad(ch0 + [ch0[0] + 12], end - t0 + 0.5, 1800, 900, 0.9, 123, st["pad"]), t0, 0.7)
        for i, nt in enumerate(ch0 + [ch0[0] + 12, ch0[1] + 12]):
            self.add("arp", self.lead_note(nt + 12, 0.5, 0.45, 0.5, s=i), one + i * self.step * 1.5, 0.3,
                     pan=-0.4 + 0.2 * i)

    # -- mixdown ---------------------------------------------------------------
    def mix(self):
        S = self.stems
        n = len(S["drums"])
        a, b = int(self.p["drop"] * SR), int(self.p["breath"] * SR)
        if b - a < SR:
            a, b = int(self.p["slam"] * SR), int(self.p["hit"] * SR)

        def rms(x):
            seg = x[a:b]
            return math.sqrt(float((seg ** 2).mean()) + 1e-12)
        ref = rms(S["drums"])
        targets = {"bass": -4.5, "chords": -6.5, "pad": -10.5, "arp": -11.5, "fx": -9.0}
        targets.update({
            "uplifting": dict(bass=-5.0, chords=-6.0, arp=-10.5),
            "chill": dict(bass=-3.0, chords=-3.5, pad=-8.0, arp=-7.0, fx=-14.0),
            "acoustic": dict(bass=-5.0, chords=-3.0, pad=-9.0, arp=-4.0),
            "minimal": dict(bass=-2.0, chords=-4.0, pad=-6.0, arp=-5.0),
            "cinematic": dict(bass=-6.0, arp=-4.0, pad=-7.0, fx=-5.0),
            "house": dict(bass=-4.0, chords=-5.0),
        }.get(self.mood, {}))
        if ref > 1e-6:
            for k, tdb in targets.items():
                r = rms(S[k])
                if r > 1e-6:
                    S[k] = S[k] * (ref * 10 ** (tdb / 20) / r)
        t = np.arange(n) / SR
        depth = {"chill": 0.3, "acoustic": 0.25, "minimal": 0.35, "cinematic": 0.2}.get(self.mood, 0.62)
        sc = np.ones(n)
        for tk in self.kicks:
            i0, i1 = int(tk * SR), min(n, int((tk + 0.3) * SR))
            if i0 >= n:
                continue
            tt = t[i0:i1] - tk
            sc[i0:i1] = np.minimum(sc[i0:i1], 1 - depth * np.exp(-tt / 0.09) * np.clip(tt / 0.004, 0, 1))
        pump = sc[:, None]
        drums = S["drums"] + reverb(S["drums"], 1.4, 0.12, bright=0.6, hp=400)
        bass = compress(S["bass"], -14, 3, 0.003, 0.08, 2)
        if self.st["bass"] == "offbeat":
            bass = bass * (0.55 + 0.45 * pump)
        chords = S["chords"] * pump
        chords = chords + reverb(chords, 2.2, 0.22) + pingpong(chords, self.step * 3, 0.3, 0.12)
        padm = S["pad"] * (0.45 + 0.55 * pump)
        padm = padm + reverb(padm, 3.0, 0.3)
        arp = S["arp"] * (0.6 + 0.4 * pump)
        arp = arp + pingpong(arp, self.step * 3, 0.38, 0.32) + reverb(arp, 2.0, 0.2)
        fx = S["fx"] + reverb(S["fx"], 2.5, 0.25)
        mix = drums + bass + chords + padm + arp + fx
        g = np.ones(n)
        for te, gap in ((self.p["drop"], self.st.get("gap", 0.1)), (self.p["hit"], 0.015)):
            if gap <= 0:
                continue
            i0, i1 = int((te - gap) * SR), int(te * SR)
            if 0 <= i0 < n:
                g[i0:i1] = np.minimum(g[i0:i1], 0.04)
        mix *= uniform(g, N(0.006))[:, None]
        mix = filt(mix, "hp", 28, order=2)
        mix = filt(mix, "lowshelf", 90, 0.7, -2.0)
        mix = filt(mix, "peak", 2800, 0.8, 1.5)
        if self.mood == "chill":
            mix = filt(mix, "lp", 9000)
        else:
            mix = filt(mix, "highshelf", 5000, 0.7, 4.0)
        mix = compress(mix, -16, 2.2, 0.01, 0.12, 1.5)
        mix = sat(mix * 1.1, 1.15)
        fade = np.clip((self.p["end"] - t) / 1.4, 0, 1) ** 1.5
        mix *= fade[:, None]
        return normalize_lufs(mix, -16.0)


def uniform(x, k):
    from scipy.ndimage import uniform_filter1d
    return uniform_filter1d(x, size=max(1, k))
