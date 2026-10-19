"""Math, timing, easing, curves and colour helpers.

Everything here is pure and deterministic: the same input always gives the same
output, so any frame can be rendered in isolation, in any order, on any worker.
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

# ---------------------------------------------------------------------------
# scalars
# ---------------------------------------------------------------------------

def clamp(x, a=0.0, b=1.0):
    return np.minimum(np.maximum(x, a), b) if isinstance(x, np.ndarray) else (a if x < a else b if x > b else x)


def lerp(a, b, t):
    return a + (b - a) * t


def ramp(x, a, b):
    """0 before a, 1 after b, linear between. Safe when a == b."""
    if b == a:
        return 1.0 if x >= b else 0.0
    return clamp((x - a) / (b - a))


def window(t, t0, t1, fade_in=0.2, fade_out=0.2):
    """1 inside [t0, t1] with soft fades on both ends (smoothstep)."""
    if t < t0 or t > t1:
        return 0.0
    a = smooth(ramp(t, t0, t0 + fade_in)) if fade_in > 0 else 1.0
    b = 1.0 - smooth(ramp(t, t1 - fade_out, t1)) if fade_out > 0 else 1.0
    return a * b


# ---------------------------------------------------------------------------
# easing: smooth for general moves, ease_out for arrivals, ease_in for exits.
# Never put a long move on a steep ease_in: nothing happens, then it lurches.
# ---------------------------------------------------------------------------

def smooth(t):
    t = clamp(t)
    return t * t * (3 - 2 * t)


def smoother(t):
    t = clamp(t)
    return t * t * t * (t * (6 * t - 15) + 10)


def ease_out(t, p=3.0):
    t = clamp(t)
    return 1 - (1 - t) ** p


def ease_in(t, p=2.0):
    t = clamp(t)
    return t ** p


def ease_in_out(t, p=3.0):
    t = clamp(t)
    return 0.5 * (2 * t) ** p if t < 0.5 else 1 - 0.5 * (2 - 2 * t) ** p


def ease_out_back(t, s=1.70158):
    t = clamp(t) - 1
    return 1 + t * t * ((s + 1) * t + s)


def damped(t, freq=6.0, zeta=0.35):
    """Damped oscillation starting at 1 when t=0, settling to 0 (0 for t<0)."""
    if t < 0:
        return 0.0
    w = 2 * math.pi * freq
    wd = w * math.sqrt(max(1e-6, 1 - zeta * zeta))
    return math.exp(-zeta * w * t) * math.cos(wd * t)


def spring_in(t, freq=4.0, zeta=0.45):
    """0 -> 1 with a springy overshoot. Use for pops (avatars, chips)."""
    if t <= 0:
        return 0.0
    return 1.0 - damped(t, freq, zeta)


def pulse(t, t0, attack=0.04, decay=0.35):
    """A hit envelope: fast rise at t0, exponential fall."""
    if t < t0 - attack:
        return 0.0
    if t < t0:
        return smooth((t - (t0 - attack)) / attack)
    return math.exp(-(t - t0) / max(1e-4, decay))


# ---------------------------------------------------------------------------
# curves
# ---------------------------------------------------------------------------

def bezier(p0, p1, p2, p3, u):
    u = np.asarray(u, dtype=np.float64)[..., None]
    p0, p1, p2, p3 = (np.asarray(p, dtype=np.float64) for p in (p0, p1, p2, p3))
    v = 1 - u
    return v ** 3 * p0 + 3 * v * v * u * p1 + 3 * v * u * u * p2 + u ** 3 * p3


def chaikin(pts, passes=4, closed=True):
    pts = np.asarray(pts, dtype=np.float64)
    for _ in range(passes):
        a = pts
        b = np.roll(pts, -1, axis=0) if closed else pts[1:]
        if not closed:
            a = pts[:-1]
        q = 0.75 * a + 0.25 * b
        r = 0.25 * a + 0.75 * b
        out = np.empty((len(q) * 2, 2))
        out[0::2], out[1::2] = q, r
        pts = out if closed else np.vstack([pts[:1], out, pts[-1:]])
    return pts


def resample(pts, n, closed=False):
    """Resample a polyline to n points evenly spaced by arc length."""
    pts = np.asarray(pts, dtype=np.float64)
    if closed:
        pts = np.vstack([pts, pts[:1]])
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    total = s[-1] if s[-1] > 0 else 1.0
    targets = np.linspace(0, total, n, endpoint=not closed)
    x = np.interp(targets, s, pts[:, 0])
    y = np.interp(targets, s, pts[:, 1])
    return np.stack([x, y], axis=1)


class Path:
    """A polyline parameterised by arc-length fraction u in [0, 1].

    Build one from cubic Bezier segments (``Path.beziers``) or raw points.
    ``span(u0, u1, n)`` returns the n points between two fractions; that is how
    every light gets its visible tail.
    """

    def __init__(self, pts, closed=False):
        pts = np.asarray(pts, dtype=np.float64)
        if closed:
            pts = np.vstack([pts, pts[:1]])
        self.closed = closed
        self.pts = pts
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        self.s = np.concatenate([[0], np.cumsum(seg)])
        self.length = float(self.s[-1]) or 1.0

    @classmethod
    def beziers(cls, segments, samples=160):
        """segments: list of (p0, p1, p2, p3). Consecutive segments should share ends."""
        out = []
        for i, (p0, p1, p2, p3) in enumerate(segments):
            u = np.linspace(0, 1, samples)
            if i:
                u = u[1:]
            out.append(bezier(p0, p1, p2, p3, u))
        return cls(np.vstack(out))

    @classmethod
    def through(cls, a, b, bend=0.25, side=1.0, samples=160):
        """A soft single-bend Bezier from a to b (the default for hand-overs)."""
        a, b = np.asarray(a, float), np.asarray(b, float)
        d = b - a
        n = np.array([-d[1], d[0]]) * bend * side
        return cls.beziers([(a, a + d * 0.3 + n, a + d * 0.7 + n, b)], samples)

    def at(self, u):
        u = np.asarray(u, dtype=np.float64)
        if self.closed:
            u = np.mod(u, 1.0)
        else:
            u = np.clip(u, 0, 1)
        d = u * self.length
        x = np.interp(d, self.s, self.pts[:, 0])
        y = np.interp(d, self.s, self.pts[:, 1])
        return np.stack([x, y], axis=-1)

    def span(self, u0, u1, n=96):
        return self.at(np.linspace(u0, u1, n))

    def tangent(self, u, eps=1e-3):
        a, b = self.at(u - eps), self.at(u + eps)
        d = b - a
        return d / (np.linalg.norm(d) + 1e-9)


# ---------------------------------------------------------------------------
# colour. Helpers take and return plain float arrays in [0, 1] so they compose.
# Light maths happens in linear light; only the final frame is gamma encoded.
# ---------------------------------------------------------------------------

@lru_cache(maxsize=512)
def _hex(h: str):
    h = h.strip().lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def hex_srgb(h):
    return np.array(_hex(h), dtype=np.float32)


def srgb_to_linear(c):
    c = np.asarray(c, dtype=np.float32)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4).astype(np.float32)


_ENC_N = 16384
_ENC_LUT = None


def linear_to_srgb(c):
    """Gamma-encode linear light. Large arrays go through a 14-bit lookup table."""
    global _ENC_LUT
    c = np.asarray(c, dtype=np.float32)
    if c.size < 4096:
        c = np.clip(c, 0, 1)
        return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055).astype(np.float32)
    if _ENC_LUT is None:
        x = np.linspace(0, 1, _ENC_N, dtype=np.float64)
        _ENC_LUT = np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055).astype(np.float32)
    idx = np.clip(c * (_ENC_N - 1) + 0.5, 0, _ENC_N - 1).astype(np.int32)
    return _ENC_LUT[idx]


@lru_cache(maxsize=1024)
def _lin_t(h):
    return tuple(float(v) for v in srgb_to_linear(hex_srgb(h)))


def lin(h):
    """Hex -> linear RGB (float32[3])."""
    return np.array(_lin_t(h), dtype=np.float32)


def to_hex(rgb_srgb):
    r, g, b = (int(round(float(clamp(v)) * 255)) for v in rgb_srgb)
    return f"#{r:02X}{g:02X}{b:02X}"


def mix_hex(a, b, t):
    return to_hex(lerp(hex_srgb(a), hex_srgb(b), t))


def luminance(h):
    c = lin(h)
    return float(0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2])


def saturate_hex(h, amount=1.0, light=None):
    """Push a colour's chroma (HSL) and optionally set its lightness."""
    import colorsys
    r, g, b = _hex(h)
    hh, ll, ss = colorsys.rgb_to_hls(r, g, b)
    ss = clamp(ss * amount)
    if light is not None:
        ll = light
    return to_hex(colorsys.hls_to_rgb(hh, ll, ss))


# ---------------------------------------------------------------------------
# randomness. Seed once; never draw random numbers at render time.
# ---------------------------------------------------------------------------

def rng(seed):
    return np.random.default_rng(seed)


def hash01(*keys):
    """Stable pseudo-random float in [0, 1) from any hashable keys."""
    h = 2166136261
    for k in keys:
        for ch in str(k):
            h = (h ^ ord(ch)) * 16777619 & 0xFFFFFFFF
    return (h % 100000) / 100000.0


def piecewise(t, table):
    """Piecewise-linear clock map. table = [(film_t, design_t), ...] sorted.

    Use it when the voice moves after a scene is built: map film time onto the
    scene's design clock instead of rewriting the scene.
    """
    xs = [a for a, _ in table]
    ys = [b for _, b in table]
    return float(np.interp(t, xs, ys))
