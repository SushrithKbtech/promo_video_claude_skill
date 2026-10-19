"""Audio DSP building blocks (numpy/scipy, 48 kHz, stereo arrays shaped (n, 2)).

Band-limited oscillators, RBJ biquads, time-varying filters, convolution
reverb with frequency-dependent decay, ping-pong delay, compressor, sidechain,
saturation, look-ahead limiter and BS.1770 loudness.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import signal
from scipy.ndimage import maximum_filter1d, minimum_filter1d, uniform_filter1d

SR = 48000


def secs(n):
    return n / SR


def N(sec):
    return max(1, int(round(sec * SR)))


def midi(n):
    return 440.0 * 2 ** ((n - 69) / 12)


def db(x):
    return 10 ** (x / 20)


def stereo(m, pan=0.0, width=0.0):
    """Mono -> stereo with equal-power pan in [-1, 1]."""
    a = (pan + 1) * math.pi / 4
    return np.stack([m * math.cos(a), m * math.sin(a)], 1) * math.sqrt(2)


def place(buf, x, t, gain=1.0):
    """Add stereo clip x into buf at time t (clips at both ends)."""
    i = int(round(t * SR))
    if x.ndim == 1:
        x = stereo(x)
    j0, j1 = max(0, i), min(len(buf), i + len(x))
    if j1 > j0:
        buf[j0:j1] += x[j0 - i:j1 - i] * gain


# ---------------------------------------------------------------------------
# oscillators
# ---------------------------------------------------------------------------

def phase(freq, n, ph0=0.0):
    f = np.broadcast_to(np.asarray(freq, np.float64), (n,))
    return (ph0 + np.cumsum(f) / SR) % 1.0, f / SR


def sine(freq, n, ph0=0.0):
    p, _ = phase(freq, n, ph0)
    return np.sin(2 * np.pi * p)


def _blep(t, dt):
    y = np.zeros_like(t)
    m = t < dt
    x = t[m] / dt[m]
    y[m] = x + x - x * x - 1.0
    m2 = t > 1 - dt
    x = (t[m2] - 1.0) / dt[m2]
    y[m2] = x * x + x + x + 1.0
    return y


def saw(freq, n, ph0=0.0):
    """PolyBLEP band-limited sawtooth."""
    p, dt = phase(freq, n, ph0)
    return (2.0 * p - 1.0) - _blep(p, dt)


def square(freq, n, ph0=0.0, pw=0.5):
    p, dt = phase(freq, n, ph0)
    s = np.where(p < pw, 1.0, -1.0)
    s += _blep(p, dt)
    s -= _blep((p + 1 - pw) % 1.0, dt)
    return s


def noise(n, seed=0):
    return np.random.default_rng(seed).standard_normal(n)


def supersaw(freq, n, voices=7, detune=0.22, seed=0, stereo_out=True):
    """Detuned saw stack; voices alternate left/right for width."""
    r = np.random.default_rng(seed)
    offs = np.linspace(-1, 1, voices) * detune  # semitones
    L = np.zeros(n)
    R = np.zeros(n)
    for i, o in enumerate(offs):
        f = np.asarray(freq) * 2 ** (o / 12)
        s = saw(f, n, r.random())
        g = 1.0 if i == voices // 2 else 0.62
        if i % 2:
            L += s * g * 0.75
            R += s * g * 0.25
        else:
            L += s * g * 0.25
            R += s * g * 0.75
    out = np.stack([L, R], 1) / (voices * 0.55)
    return out if stereo_out else out.mean(1)


# ---------------------------------------------------------------------------
# envelopes
# ---------------------------------------------------------------------------

def adsr(n, a=0.005, d=0.1, s=0.7, r=0.2, gate=None):
    gate = gate if gate is not None else max(0.0, secs(n) - r)
    t = np.arange(n) / SR
    e = np.where(t < a, t / max(a, 1e-6), 0.0)
    m = (t >= a) & (t < a + d)
    e[m] = 1 - (1 - s) * (t[m] - a) / max(d, 1e-6)
    m = (t >= a + d) & (t < gate)
    e[m] = s
    m = t >= gate
    lvl = s if gate >= a + d else (gate / max(a, 1e-6) if gate < a else 1 - (1 - s) * (gate - a) / max(d, 1e-6))
    e[m] = lvl * np.exp(-(t[m] - gate) / max(r / 4.6, 1e-4))
    return e


def expdec(n, tau, attack=0.0015):
    t = np.arange(n) / SR
    e = np.exp(-t / max(tau, 1e-5))
    if attack > 0:
        e *= np.clip(t / attack, 0, 1)
    return e


# ---------------------------------------------------------------------------
# filters (RBJ cookbook -> second-order sections)
# ---------------------------------------------------------------------------

def _biquad(kind, f, q=0.707, gain_db=0.0):
    f = min(max(f, 10.0), SR * 0.49)
    w0 = 2 * math.pi * f / SR
    cw, sw = math.cos(w0), math.sin(w0)
    alpha = sw / (2 * q)
    A = 10 ** (gain_db / 40)
    if kind == "lp":
        b = [(1 - cw) / 2, 1 - cw, (1 - cw) / 2]
        a = [1 + alpha, -2 * cw, 1 - alpha]
    elif kind == "hp":
        b = [(1 + cw) / 2, -(1 + cw), (1 + cw) / 2]
        a = [1 + alpha, -2 * cw, 1 - alpha]
    elif kind == "bp":
        b = [alpha, 0, -alpha]
        a = [1 + alpha, -2 * cw, 1 - alpha]
    elif kind == "peak":
        b = [1 + alpha * A, -2 * cw, 1 - alpha * A]
        a = [1 + alpha / A, -2 * cw, 1 - alpha / A]
    elif kind == "lowshelf":
        sq = 2 * math.sqrt(A) * alpha
        b = [A * ((A + 1) - (A - 1) * cw + sq), 2 * A * ((A - 1) - (A + 1) * cw), A * ((A + 1) - (A - 1) * cw - sq)]
        a = [(A + 1) + (A - 1) * cw + sq, -2 * ((A - 1) + (A + 1) * cw), (A + 1) + (A - 1) * cw - sq]
    elif kind == "highshelf":
        sq = 2 * math.sqrt(A) * alpha
        b = [A * ((A + 1) + (A - 1) * cw + sq), -2 * A * ((A - 1) + (A + 1) * cw), A * ((A + 1) + (A - 1) * cw - sq)]
        a = [(A + 1) - (A - 1) * cw + sq, 2 * ((A - 1) - (A + 1) * cw), (A + 1) - (A - 1) * cw - sq]
    else:
        raise ValueError(kind)
    return np.array([[b[0] / a[0], b[1] / a[0], b[2] / a[0], 1.0, a[1] / a[0], a[2] / a[0]]])


def filt(x, kind, f, q=0.707, gain_db=0.0, order=1):
    sos = np.vstack([_biquad(kind, f, q, gain_db)] * order)
    return signal.sosfilt(sos, x, axis=0)


def sweep(x, kind, f_curve, q=0.707, block=128):
    """Time-varying filter: f_curve is an array of cutoffs (one per sample) or (f0, f1)."""
    n = len(x)
    if isinstance(f_curve, tuple):
        f_curve = np.geomspace(max(f_curve[0], 10), max(f_curve[1], 10), n)
    y = np.zeros_like(x)
    zshape = (1, 2) + x.shape[1:]
    zi = np.zeros(zshape)
    for i in range(0, n, block):
        sos = _biquad(kind, float(f_curve[min(n - 1, i + block // 2)]), q)
        y[i:i + block], zi = signal.sosfilt(sos, x[i:i + block], axis=0, zi=zi)
    return y


def sat(x, drive=1.5):
    return np.tanh(x * drive) / np.tanh(drive)


# ---------------------------------------------------------------------------
# space
# ---------------------------------------------------------------------------

_IR = {}


def reverb_ir(rt60=2.2, predelay=0.018, bright=0.6, seed=3):
    key = (round(rt60, 2), round(predelay, 3), round(bright, 2), seed)
    if key in _IR:
        return _IR[key]
    n = N(rt60 * 1.15)
    t = np.arange(n) / SR
    r = np.random.default_rng(seed)
    out = []
    for ch in range(2):
        nz = r.standard_normal(n)
        lo = filt(nz, "lp", 900)
        mid = filt(filt(nz, "hp", 900), "lp", 4500)
        hi = filt(nz, "hp", 4500)
        e = lambda k: np.exp(-6.91 * t / (rt60 * k))  # noqa: E731
        ir = lo * e(1.0) + mid * e(0.8) * 0.9 + hi * e(0.45) * bright
        ir *= np.clip(t / 0.006, 0, 1)
        # early reflections
        for d, g in ((0.011, 0.5), (0.019, 0.38), (0.027, 0.3), (0.041, 0.22)):
            j = N(d + 0.002 * ch)
            if j < n:
                ir[j] += g * (1 if r.random() < 0.5 else -1)
        out.append(ir)
    ir = np.stack(out, 1)
    ir /= np.sqrt((ir ** 2).sum(0, keepdims=True)) + 1e-9
    pd = N(predelay)
    ir = np.vstack([np.zeros((pd, 2)), ir])
    _IR[key] = ir
    return ir


def reverb(x, rt60=2.2, mix=1.0, predelay=0.018, bright=0.6, hp=180.0):
    """Wet signal only (mix scales it). x stereo (n, 2)."""
    ir = reverb_ir(rt60, predelay, bright)
    src = filt(x, "hp", hp) if hp else x
    wet = np.stack([signal.fftconvolve(src[:, c], ir[:, c])[:len(x)] for c in range(2)], 1)
    return wet * mix


def pingpong(x, delay_s, feedback=0.38, mix=0.3, taps=6, lp=5000.0):
    y = np.zeros_like(x)
    d = N(delay_s)
    cur = filt(x, "lp", lp)
    for k in range(1, taps + 1):
        cur = cur * feedback if k > 1 else cur * 0.8
        sh = d * k
        if sh >= len(x):
            break
        ch = (k + 1) % 2
        y[sh:, ch] += cur[:len(x) - sh].mean(1)
    return y * mix


# ---------------------------------------------------------------------------
# dynamics
# ---------------------------------------------------------------------------

def gain_curve_from_events(n, events, depth_db, attack=0.005, release=0.18):
    """Duck curve: at each event time drop by depth_db, recover with release."""
    g = np.ones(n)
    t = np.arange(n) / SR
    for te in events:
        i0 = int((te - attack) * SR)
        if i0 >= n:
            continue
        i0 = max(0, i0)
        i1 = min(n, int((te + release * 6) * SR))
        tt = t[i0:i1] - te
        dep = 1 - db(-abs(depth_db))
        env = np.where(tt < 0, 1 - dep * (tt + attack) / attack, 1 - dep * np.exp(-tt / release))
        g[i0:i1] = np.minimum(g[i0:i1], env)
    return g


def smooth_gain(target_db, attack=0.06, release=0.3, rate=1000):
    """Asymmetric one-pole smoothing of a gain curve given at ``rate`` Hz (dB)."""
    out = np.empty_like(target_db)
    aa = math.exp(-1 / (attack * rate))
    ar = math.exp(-1 / (release * rate))
    y = target_db[0]
    for i, v in enumerate(target_db):
        c = aa if v < y else ar
        y = c * y + (1 - c) * v
        out[i] = y
    return out


def compress(x, thr_db=-18, ratio=3.0, attack=0.005, release=0.1, makeup_db=0.0, knee=6.0):
    """Feed-forward RMS compressor on 1 ms blocks (stereo-linked)."""
    blk = SR // 1000
    n = len(x)
    nb = (n + blk - 1) // blk
    pad = np.zeros((nb * blk,) + x.shape[1:])
    pad[:n] = x
    lvl = pad.reshape(nb, blk, -1) if x.ndim > 1 else pad.reshape(nb, blk, 1)
    rms = np.sqrt((lvl ** 2).mean(axis=(1, 2)) + 1e-12)
    ldb = 20 * np.log10(rms)
    over = ldb - thr_db
    gr = np.where(over <= -knee / 2, 0.0,
                  np.where(over >= knee / 2, over * (1 - 1 / ratio),
                           (1 - 1 / ratio) * (over + knee / 2) ** 2 / (2 * knee)))
    gr_s = smooth_gain(-gr, attack, release, 1000)
    g = db(gr_s + makeup_db)
    gs = np.interp(np.arange(n) / blk, np.arange(nb) + 0.5, g)
    return x * (gs[:, None] if x.ndim > 1 else gs)


def limiter(x, ceiling_db=-1.2, lookahead=0.003, release=0.08):
    """Look-ahead brick-wall limiter (sample-peak), smooth gain."""
    c = db(ceiling_db)
    peak = np.abs(x).max(1) if x.ndim > 1 else np.abs(x)
    la = N(lookahead)
    held = maximum_filter1d(peak, size=2 * la + 1)
    g = np.minimum(1.0, c / np.maximum(held, 1e-9))
    g = minimum_filter1d(g, size=N(release))
    g = uniform_filter1d(g, size=la * 2 + 1)
    y = x * (g[:, None] if x.ndim > 1 else g)
    return np.clip(y, -c, c)


def true_peak_db(x):
    up = signal.resample_poly(x, 4, 1, axis=0)
    return 20 * np.log10(np.abs(up).max() + 1e-12)


# ---------------------------------------------------------------------------
# loudness (ITU-R BS.1770-4)
# ---------------------------------------------------------------------------

def _k_weight(x):
    # pre-filter (high shelf) and RLB high-pass, 48 kHz coefficients from the spec
    b1 = [1.53512485958697, -2.69169618940638, 1.19839281085285]
    a1 = [1.0, -1.69065929318241, 0.73248077421585]
    b2 = [1.0, -2.0, 1.0]
    a2 = [1.0, -1.99004745483398, 0.99007225036621]
    y = signal.lfilter(b1, a1, x, axis=0)
    return signal.lfilter(b2, a2, y, axis=0)


def lufs(x):
    if x.ndim == 1:
        x = x[:, None]
    y = _k_weight(x)
    blk, hop = N(0.4), N(0.1)
    if len(y) < blk:
        return -70.0
    ms = []
    for i in range(0, len(y) - blk + 1, hop):
        ms.append((y[i:i + blk] ** 2).mean(0).sum())
    ms = np.array(ms)
    L = -0.691 + 10 * np.log10(ms + 1e-12)
    g = ms[L > -70]
    if len(g) == 0:
        return -70.0
    rel = -0.691 + 10 * np.log10(g.mean()) - 10
    g2 = ms[(L > -70) & (L > rel)]
    return float(-0.691 + 10 * np.log10(g2.mean() + 1e-12))


def normalize_lufs(x, target):
    cur = lufs(x)
    return x * db(target - cur) if cur > -69 else x
