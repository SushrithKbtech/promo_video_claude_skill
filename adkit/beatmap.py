"""Beat tracking for the user's own music, and the film's beat grid.

    track = analyze(path)   ->  {"bpm", "beats" (s), "downbeats" (s), "drop_at" (s), "duration"}

Beats come from a dynamic-programming tracker over a spectral-flux onset envelope (it follows
small tempo drifts instead of assuming a perfect metronome). Downbeats are the beat phase whose
beats carry the most kick energy. The drop is the biggest rise in low-end energy, snapped to a
downbeat. Results are cached next to the project.

A ``Grid`` turns that into film time: the track is offset so its drop lands on the logo hit, the
film's downbeats are the track's downbeats, and ``snap`` puts any moment on the nearest beat.
With the built-in score the grid is simply a perfect metronome.
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
from scipy import signal

SR = 22050
HOP = 256


def _decode(path):
    import shutil
    import subprocess
    r = subprocess.run([shutil.which("ffmpeg") or "ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", str(SR),
                        "-f", "f32le", "-"], capture_output=True, check=True)
    return np.frombuffer(r.stdout, np.float32).astype(np.float64)


def _onsets(x):
    f, t, Z = signal.stft(x, SR, nperseg=1024, noverlap=1024 - HOP)
    M = np.log1p(np.abs(Z) * 20)
    flux = np.maximum(0, np.diff(M, axis=1))
    env = flux.sum(0)
    low = flux[f[:] < 180][:, :].sum(0) if (f < 180).any() else env
    env = env - signal.medfilt(env, 31)
    env = np.maximum(env, 0)
    env /= env.std() + 1e-9
    low = np.maximum(low - signal.medfilt(low, 31), 0)
    return env, low, HOP / SR


def _tempo(env, dt, prior=120.0):
    e = env - env.mean()
    ac = np.correlate(e, e, "full")[len(e) - 1:]
    lags = np.arange(1, len(ac))
    bpm = 60.0 / (lags * dt)
    ok = (bpm > 60) & (bpm < 200)
    w = np.exp(-0.5 * (np.log2(bpm[ok] / prior) / 0.9) ** 2)  # prefer ~120 BPM
    sc = ac[1:][ok] * w
    return float(bpm[ok][np.argmax(sc)])


def _track(env, dt, bpm, tight=100.0):
    period = 60.0 / bpm / dt
    n = len(env)
    score = env.copy()
    back = -np.ones(n, int)
    lo, hi = int(round(period / 2)), int(round(period * 2))
    for t in range(hi, n):
        prev = np.arange(t - hi, t - lo)
        cost = -tight * (np.log((t - prev) / period)) ** 2
        cand = score[prev] + cost
        j = int(np.argmax(cand))
        score[t] = env[t] + cand[j]
        back[t] = prev[j]
    t = int(np.argmax(score[-int(period * 2):])) + n - int(period * 2)
    beats = []
    while t >= 0:
        beats.append(t)
        t = back[t]
    return np.array(beats[::-1]) * dt


def analyze(path, cache_dir=None):
    st = os.stat(path)
    key = hashlib.sha1(f"{os.path.abspath(path)}|{st.st_size}|{st.st_mtime}".encode()).hexdigest()[:12]
    if cache_dir:
        cp = os.path.join(cache_dir, f"beats_{key}.json")
        if os.path.exists(cp):
            return json.load(open(cp, encoding="utf-8"))
    x = _decode(path)
    env, low, dt = _onsets(x)
    bpm = _tempo(env, dt)
    beats = _track(env, dt, bpm)
    if len(beats) > 4:
        bpm = 60.0 / float(np.median(np.diff(beats)))
    # downbeat phase: the phase whose beats carry the most kick energy
    idx = np.clip((beats / dt).astype(int), 0, len(low) - 1)
    kick = np.array([low[max(0, i - 2):i + 3].max() for i in idx])
    phase = int(np.argmax([kick[p::4].mean() if len(kick[p::4]) else 0 for p in range(4)]))
    downbeats = beats[phase::4]
    # drop: biggest rise of low-end energy (2 s after vs 4 s before), on a downbeat
    rms = np.sqrt(np.convolve(signal.sosfilt(signal.butter(4, 160, "lp", fs=SR, output="sos"), x) ** 2,
                              np.ones(SR // 10) / (SR // 10), "same"))[::SR // 20]  # 20 Hz control rate
    fr = 20.0
    a, bwin = int(2 * fr), int(4 * fr)
    rise = np.array([rms[i:i + a].mean() / (rms[max(0, i - bwin):i].mean() + 1e-9) if i > bwin else 0
                     for i in range(len(rms))])
    t_drop = float(np.argmax(rise[:max(1, len(rise) - a)]) / fr) if len(rise) else 0.0
    if len(downbeats):
        t_drop = float(downbeats[np.argmin(np.abs(downbeats - t_drop))])
    out = {"bpm": round(bpm, 2), "beats": [round(float(b), 4) for b in beats],
           "downbeats": [round(float(b), 4) for b in downbeats], "drop_at": round(t_drop, 4),
           "duration": round(len(x) / SR, 3)}
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        json.dump(out, open(cp, "w", encoding="utf-8"))
    return out


def bar_map(path, track):
    """Loudness per bar (downbeat to downbeat) and where the song's sections change.

    Returns [{"bar", "t", "db", "low_db", "mark"}]; mark is "DROP" where the low end jumps in,
    "break" where it falls away, "build" where it first arrives. Use it to put scene cuts on
    sections: the product reveal on a drop, the reflective beat in the breakdown, the logo on the
    last drop."""
    x = _decode(path)
    low = signal.sosfilt(signal.butter(4, 150, "lp", fs=SR, output="sos"), x)
    db = np.asarray(track["downbeats"], float)
    rows, prev = [], None
    for i in range(len(db) - 1):
        a, b = int(db[i] * SR), int(db[i + 1] * SR)
        if b > len(x):
            break
        lv = lambda s_: float(20 * np.log10(np.sqrt(np.mean(s_[a:b] ** 2)) + 1e-9))
        r = {"bar": i, "t": round(float(db[i]), 3), "db": round(lv(x), 1), "low_db": round(lv(low), 1), "mark": ""}
        if prev is not None:
            d = r["low_db"] - prev["low_db"]
            if d >= 10 and r["low_db"] > -16:
                r["mark"] = "DROP"
            elif d >= 10:
                r["mark"] = "build"
            elif d <= -8:
                r["mark"] = "break"
        rows.append(r)
        prev = r
    for r0, r1 in zip(rows, rows[1:]):   # a dip right before a drop is the riser, not a breakdown
        if r0["mark"] == "break" and r1["mark"] == "DROP":
            r0["mark"] = "pre-drop"
    return rows


def bar_map_cached(path, track, cache_dir):
    st = os.stat(path)
    key = hashlib.sha1(f"{os.path.abspath(path)}|{st.st_size}|{st.st_mtime}|bars".encode()).hexdigest()[:12]
    cp = os.path.join(cache_dir, f"bars_{key}.json")
    if os.path.exists(cp):
        return json.load(open(cp, encoding="utf-8"))
    rows = bar_map(path, track)
    os.makedirs(cache_dir, exist_ok=True)
    json.dump(rows, open(cp, "w", encoding="utf-8"))
    return rows


class Grid:
    """Film-time beats and downbeats (from a track, or a perfect metronome)."""

    def __init__(self, bpm, d0=0.6, beats=None, downbeats=None, horizon=600.0):
        self.bpm = float(bpm)
        self.beat = 60.0 / self.bpm
        if beats is None:
            n = int(horizon / self.beat) + 8
            self.beats = d0 + np.arange(-8, n) * self.beat
            self.downbeats = d0 + np.arange(-2, n // 4 + 2) * self.beat * 4
        else:
            self.beats = self._extend(np.asarray(beats, float), self.beat, horizon)
            self.downbeats = self._extend(np.asarray(downbeats, float), self.beat * 4, horizon)

    @staticmethod
    def _extend(ts, step, horizon):
        """Continue a detected grid before the music starts and after it ends."""
        if len(ts) < 2:
            return np.arange(-step * 4, horizon, step)
        st = float(np.median(np.diff(ts[-8:])))
        before = ts[0] - st * np.arange(40, 0, -1)
        after = ts[-1] + st * np.arange(1, int(horizon / st) + 2)
        return np.concatenate([before, ts, after])

    def downbeats_from(self, t_min):
        return self.downbeats[self.downbeats >= t_min]

    def snap(self, t, grid="beat", within=None):
        arr = self.downbeats if grid == "downbeat" else self.beats if grid == "beat" else \
            np.sort(np.concatenate([self.beats, (self.beats[:-1] + self.beats[1:]) / 2]))
        j = int(np.argmin(np.abs(arr - t)))
        if within is not None and abs(arr[j] - t) > within:
            return t
        return float(arr[j])

    def beats_between(self, t0, t1):
        return [float(b) for b in self.beats if t0 <= b < t1]
