"""Voice-over: record it first, then place each line so its KEY WORD lands on
the picture beat (start = beat - word_offset). Never time-stretch the voice.

Engines
    edge        Microsoft neural voices via edge-tts (free, no key, word timings)
    elevenlabs  ELEVENLABS_API_KEY set; supports audio tags like [upbeat] on v3
    none        no voice (captions off)

brief.vo = [{"text": "Every file.", "key": "file", "at": "act1.w1"}, ...]
``at`` is a film anchor ("slam", "drop", "hit", "act2.key", "act1.w3" ...) with
an optional offset ("hit+0.3"), or seconds.
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import shutil
import subprocess

import numpy as np
from scipy.io import wavfile

from .dsp import SR

DEFAULT_EDGE_VOICE = "en-US-AvaMultilingualNeural"


def _norm(w):
    return re.sub(r"[^a-z0-9']", "", w.lower())


async def _edge(text, voice, rate, pitch, mp3):
    import edge_tts
    try:
        com = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch, boundary="WordBoundary")
    except TypeError:
        com = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
    words = []
    with open(mp3, "wb") as fh:
        async for ch in com.stream():
            if ch["type"] == "audio":
                fh.write(ch["data"])
            elif ch["type"] == "WordBoundary":
                words.append({"w": ch["text"], "s": ch["offset"] / 1e7, "e": (ch["offset"] + ch["duration"]) / 1e7})
    return words


def _eleven(text, voice_id, model, mp3):
    import requests
    key = os.environ["ELEVENLABS_API_KEY"]
    r = requests.post(f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/with-timestamps",
                      headers={"xi-api-key": key, "Content-Type": "application/json"},
                      json={"text": text, "model_id": model}, timeout=120)
    r.raise_for_status()
    js = r.json()
    open(mp3, "wb").write(base64.b64decode(js["audio_base64"]))
    al = js.get("alignment") or js.get("normalized_alignment") or {}
    chars, st, en = al.get("characters", []), al.get("character_start_times_seconds", []), \
        al.get("character_end_times_seconds", [])
    words, cur, s0 = [], "", None
    in_tag = False
    for c, a, b in zip(chars, st, en):
        if c == "[":
            in_tag = True
        if in_tag:
            if c == "]":
                in_tag = False
            continue
        if c.isspace():
            if cur:
                words.append({"w": cur, "s": s0, "e": b})
            cur, s0 = "", None
        else:
            if s0 is None:
                s0 = a
            cur += c
            last = b
    if cur:
        words.append({"w": cur, "s": s0, "e": last})
    return words


def _to_wav(src, dst):
    subprocess.run([shutil.which("ffmpeg") or "ffmpeg", "-y", "-v", "error", "-i", src, "-ac", "1", "-ar", str(SR),
                    "-c:a", "pcm_f32le", dst], check=True)
    sr, x = wavfile.read(dst)
    return x.astype(np.float64)


def _trim(x, thresh_db=-40.0):
    """Trim to the first/last sound; 6 ms fade in, 40 ms fade out."""
    thr = 10 ** (thresh_db / 20) * (np.abs(x).max() + 1e-9)
    idx = np.nonzero(np.abs(x) > thr)[0]
    if len(idx) == 0:
        return x, 0.0
    a, b = max(0, idx[0] - int(0.005 * SR)), min(len(x), idx[-1] + int(0.06 * SR))
    y = x[a:b].copy()
    fi, fo = int(0.006 * SR), int(0.04 * SR)
    y[:fi] *= np.linspace(0, 1, fi)
    y[-fo:] *= np.linspace(1, 0, fo)
    return y, a / SR


def build_voice(film, force=False):
    b = film.brief
    spec = b.get("voice", {})
    engine = spec.get("engine", "edge")
    lines = b.get("vo", [])
    outdir = os.path.join(film.dir, "build")
    os.makedirs(outdir, exist_ok=True)
    if engine == "none" or not lines:
        p = os.path.join(outdir, "voice.json")
        if os.path.exists(p):
            os.remove(p)
        print("[voice] no voice-over")
        return None
    if engine == "elevenlabs" and not os.environ.get("ELEVENLABS_API_KEY"):
        print("[voice] ELEVENLABS_API_KEY not set; falling back to edge-tts")
        engine = "edge"
    out_lines = []
    prev_end = -1.0
    for i, ln in enumerate(lines):
        text = ln["text"]
        mp3 = os.path.join(outdir, f"vo_{i:02d}.mp3")
        wav = os.path.join(outdir, f"vo_{i:02d}.wav")
        meta_p = os.path.join(outdir, f"vo_{i:02d}.json")
        sig = json.dumps([text, engine, spec], sort_keys=True)
        cached = os.path.exists(meta_p) and json.load(open(meta_p, encoding="utf-8")).get("sig") == sig
        if force or not cached:
            if engine == "elevenlabs":
                words = _eleven(spec.get("prefix", "") + text, spec.get("voice_id", "21m00Tcm4TlvDq8ikWAM"),
                                spec.get("model", "eleven_v3"), mp3)
            else:
                words = asyncio.run(_edge(text, spec.get("voice", DEFAULT_EDGE_VOICE), spec.get("rate", "+0%"),
                                          spec.get("pitch", "+0Hz"), mp3))
            json.dump({"sig": sig, "words": words}, open(meta_p, "w", encoding="utf-8"))
        words = json.load(open(meta_p, encoding="utf-8"))["words"]
        x = _to_wav(mp3, wav)
        y, lead = _trim(x)
        wavfile.write(wav, SR, y.astype(np.float32))
        for w in words:
            w["s"] -= lead
            w["e"] -= lead
        key = _norm(ln.get("key", "")) if ln.get("key") else None
        k_off = 0.0
        if key:
            hit = next((w for w in words if _norm(w["w"]) == key), None) or \
                next((w for w in words if key in _norm(w["w"])), None)
            if hit:
                k_off = max(0.0, hit["s"])
        target = film.resolve_time(ln.get("at", f"act{min(i + 1, len(film.scenes))}.key"))
        start = target - k_off
        if start < prev_end + 0.05:
            print(f"[voice] line {i} overlaps the previous one by {prev_end + 0.05 - start:.2f}s; "
                  f"consider a shorter line or a later anchor")
        dur = len(y) / SR
        out_lines.append(dict(text=text, file=os.path.basename(wav), start=round(start, 4), end=round(start + dur, 4),
                              key=ln.get("key"), target=round(target, 4), role=ln.get("role", ""),
                              words=[dict(w=w["w"], s=round(start + w["s"], 4), e=round(start + w["e"], 4))
                                     for w in words]))
        prev_end = start + dur
        print(f"[voice] {i}: '{text}' start {start:.2f}s, key '{ln.get('key', '')}' at {target:.2f}s")
    meta = {"engine": engine, "lines": out_lines}
    json.dump(meta, open(os.path.join(outdir, "voice.json"), "w", encoding="utf-8"), indent=1)
    return meta
