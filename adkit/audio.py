"""The mix: score (or the user's track) + sound design + voice-over.

Rules from the method:
- Put the logo on the song's own drop: a user track is offset so its biggest
  section change lands exactly on the logo hit.
- Duck the music -6 dB under each hit, -8 dB under every word (60 ms attack /
  300 ms release), -4 dB through the drop line and -7 dB under the brand name.
- The voice sits >= 15 dB above the music in 1-4 kHz on every line.
- Master to -14 LUFS integrated, true peak <= -1 dBTP (checked again after AAC).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess

import numpy as np
from scipy.io import wavfile

from .dsp import (N, SR, compress, db, filt, limiter, lufs, normalize_lufs, smooth_gain, true_peak_db)
from .music import Score
from .sfx import render_cues


def decode(path, sr=SR):
    """Any audio file -> float stereo (n, 2) via ffmpeg."""
    r = subprocess.run([shutil.which("ffmpeg") or "ffmpeg", "-v", "error", "-i", path, "-f", "f32le", "-ac", "2",
                        "-ar", str(sr), "-"], capture_output=True, check=True)
    return np.frombuffer(r.stdout, np.float32).reshape(-1, 2).astype(np.float64)


def write_wav(path, x):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    wavfile.write(path, SR, x.astype(np.float32))


def apply_gain_wav(path, gain_db):
    sr, x = wavfile.read(path)
    wavfile.write(path, sr, (x * db(gain_db)).astype(np.float32))


# ---------------------------------------------------------------------------
# user tracks: find the drop and the tempo
# ---------------------------------------------------------------------------

def analyze_track(path):
    x = decode(path).mean(1)
    hop = 512
    lo = filt(x, "lp", 160)
    frames = len(x) // hop
    e = np.sqrt((lo[:frames * hop].reshape(frames, hop) ** 2).mean(1) + 1e-12)
    full = np.sqrt((x[:frames * hop].reshape(frames, hop) ** 2).mean(1) + 1e-12)
    # onset envelope -> tempo by autocorrelation (70-180 BPM)
    on = np.maximum(0, np.diff(np.log(full + 1e-6)))
    on -= on.mean()
    ac = np.correlate(on, on, "full")[len(on) - 1:]
    fr = SR / hop
    lags = np.arange(len(ac))
    bpms = 60 * fr / np.maximum(lags, 1)
    ok = (bpms > 70) & (bpms < 180)
    bpm = float(bpms[ok][np.argmax(ac[ok])]) if ok.any() else 100.0
    # drop: biggest rise in 2-second low-band energy versus the 4 seconds before
    w = int(2 * fr)
    sm = np.convolve(e, np.ones(w) / w, "same")
    prev = np.convolve(e, np.ones(2 * w) / (2 * w), "same")
    score = sm[2 * w:] / (np.roll(prev, 2 * w)[2 * w:] + 1e-9)
    i = int(np.argmax(score[: max(1, len(score) - w)])) + 2 * w - w // 2
    # snap to the strongest onset within 0.25 s
    j0, j1 = max(0, i - int(0.25 * fr)), min(len(on), i + int(0.25 * fr))
    i = j0 + int(np.argmax(on[j0:j1])) if j1 > j0 else i
    return {"bpm": round(bpm, 2), "drop_at": round(i / fr, 3), "duration": round(len(x) / SR, 2)}


def track_bed(film, spec):
    path = spec["file"] if os.path.isabs(spec["file"]) else os.path.join(film.dir, spec["file"])
    x = decode(path)
    drop_at = spec.get("drop_at")
    if drop_at is None:
        drop_at = analyze_track(path)["drop_at"]
    offset = film.hit - float(drop_at)
    n = N(film.duration + 0.5)
    out = np.zeros((n, 2))
    i0 = int(round(offset * SR))
    a0 = max(0, -i0)
    b0 = max(0, i0)
    k = min(len(x) - a0, n - b0)
    if k > 0:
        out[b0:b0 + k] = x[a0:a0 + k]
    t = np.arange(n) / SR
    # the four moves: quiet hook, breath before the hit (thin + filtered), ring out
    g = np.ones(n)
    g *= np.clip(0.35 + 0.65 * (t - film.slam + 0.3) / 0.3, 0.35, 1)
    br0, br1 = film.hit - 0.6, film.hit - 0.015
    m = (t >= br0) & (t < br1)
    thin = filt(out, "hp", 900)
    out[m] = thin[m] * 0.6
    g *= np.clip((film.duration - t) / 1.4, 0, 1) ** 1.5
    return normalize_lufs(out * g[:, None], -16.0)


# ---------------------------------------------------------------------------
# voice
# ---------------------------------------------------------------------------

def voice_chain(x):
    """HPF 80 Hz, -2 dB @ 250, +3.5 dB @ 3.4 kHz, +4 dB shelf @ 7.5 kHz, 3:1 comp."""
    y = filt(x, "hp", 80, order=2)
    y = filt(y, "peak", 250, 1.0, -2.0)
    y = filt(y, "peak", 3400, 1.0, 3.5)
    y = filt(y, "highshelf", 7500, 0.7, 4.0)
    y = compress(y, -24, 3.0, 0.004, 0.08, 4.0)
    return y


def voice_bed(film):
    vj = os.path.join(film.dir, "build", "voice.json")
    n = N(film.duration + 0.5)
    out = np.zeros((n, 2))
    if not os.path.exists(vj):
        return out, None
    meta = json.load(open(vj, encoding="utf-8"))
    for ln in meta["lines"]:
        sr, x = wavfile.read(os.path.join(film.dir, "build", ln["file"]))
        x = x.astype(np.float64)
        if x.ndim == 1:
            x = np.stack([x, x], 1)
        i = int(round(ln["start"] * SR))
        k = min(len(x), n - i)
        if i >= 0 and k > 0:
            out[i:i + k] += x[:k]
    out = voice_chain(out)
    return normalize_lufs(out, -15.0) if np.abs(out).max() > 0 else out, meta


# ---------------------------------------------------------------------------
# the mix
# ---------------------------------------------------------------------------

def build_mix(film, out_path):
    b = film.brief
    mspec = b.get("music", {})
    n = N(film.duration + 0.5)
    t_ctl = np.arange(0, film.duration + 0.5, 0.001)

    if mspec.get("file"):
        music = track_bed(film, mspec)
    elif mspec.get("mood", "hype") == "none":
        music = np.zeros((n, 2))
    else:
        from .music import seed_of
        mood = mspec.get("mood", "hype")
        seed = seed_of(film.name, mood, mspec.get("seed", 0))
        tr = mspec.get("transpose")
        score = Score(film.music_plan(), mood, seed=seed, transpose=None if tr is None else int(tr))
        film.score_info = score.describe()
        print(f"[music] {film.score_info}")
        music = score.compose().mix()
    music = _fit(music, n)
    music *= db(float(mspec.get("gain_db", 0.0)))

    cues = film.cues()
    dark = mspec.get("mood") in ("dark", "epic", "cinematic")
    from .music import seed_of
    from .sfx import MOOD_PALETTE, Palette
    sspec = b.get("sfx", {})
    pname = sspec.get("palette") or MOOD_PALETTE.get(mspec.get("mood", "hype"), "clean")
    palette = Palette(pname, seed_of(film.name, pname, sspec.get("seed", 0), "sfx"))
    film.sfx_info = palette.describe()
    print(f"[sfx] {film.sfx_info}")
    sfx = _fit(render_cues(cues, film.duration, dark=dark, palette=palette), n)
    sfx *= db(float(b.get("sfx_gain_db", 0.0)))

    vo, meta = voice_bed(film)
    vo = _fit(vo, n)

    # ---- ducking curve for the music (dB, 1 kHz control rate) ----------
    duck = np.zeros_like(t_ctl)
    for c in cues:
        if c["kind"] == "impact":
            m = (t_ctl >= c["t"]) & (t_ctl < c["t"] + 0.25)
            duck[m] = np.minimum(duck[m], -6)
    if meta:
        lines = meta["lines"]
        for li, ln in enumerate(lines):
            for w in ln.get("words", []):
                m = (t_ctl >= w["s"] - 0.02) & (t_ctl < w["e"] + 0.05)
                duck[m] = np.minimum(duck[m], -8)
            m = (t_ctl >= ln["start"]) & (t_ctl < ln["end"])
            if ln.get("role") == "brand":
                duck[m] = np.minimum(duck[m], -7 - 8)
            elif ln.get("role") == "drop":
                duck[m] = np.minimum(duck[m], -4 - 8)
    duck_s = smooth_gain(duck, 0.06, 0.3, 1000)
    g = db(np.interp(np.arange(n) / SR, t_ctl, duck_s))
    music_d = music * g[:, None]

    # ---- voice must sit >= 15 dB above the music in 1-4 kHz ----------------
    if meta:
        vb = filt(filt(vo, "hp", 1000), "lp", 4000)
        extra = np.zeros(n)
        for ln in meta["lines"]:
            i0, i1 = int(ln["start"] * SR), int(ln["end"] * SR)
            if i1 <= i0:
                continue
            mb = filt(filt(music_d[i0:i1], "hp", 1000), "lp", 4000)
            dv = 20 * np.log10(np.sqrt((vb[i0:i1] ** 2).mean()) + 1e-9)
            dm = 20 * np.log10(np.sqrt((mb ** 2).mean()) + 1e-9)
            if dv - dm < 15:
                extra[i0:i1] = dv - dm - 15
        if extra.any():
            ec = smooth_gain(np.interp(t_ctl, np.arange(n) / SR, extra), 0.06, 0.3, 1000)
            music_d *= db(np.interp(np.arange(n) / SR, t_ctl, ec))[:, None]

    mix = music_d + sfx + vo
    mix = compress(mix, -12, 1.8, 0.01, 0.15, 0.0)
    mix = limiter(mix, -1.5)
    mix = normalize_lufs(mix, -14.0)
    mix = limiter(mix, -1.3)
    tp = true_peak_db(mix)
    if tp > -1.0:
        mix *= db(-1.0 - tp - 0.1)
    mix = mix[: N(film.duration)]
    write_wav(out_path, mix)
    stems = os.path.join(os.path.dirname(out_path), "stems")
    os.makedirs(stems, exist_ok=True)
    for name, x in (("music", music_d), ("sfx", sfx), ("voice", vo)):
        write_wav(os.path.join(stems, f"{name}.wav"), x[: N(film.duration)])
    print(f"[mix] {out_path}  {lufs(mix):.1f} LUFS  true peak {true_peak_db(mix):.2f} dBTP")
    return out_path


def _fit(x, n):
    if len(x) >= n:
        return x[:n]
    return np.vstack([x, np.zeros((n - len(x), 2))])
