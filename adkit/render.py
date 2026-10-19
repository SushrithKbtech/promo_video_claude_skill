"""Command line: stills, contact sheets, segmented parallel renders, mix, mux, checks.

    python run.py doctor
    python run.py stills  PROJECT --times 0.3,0.6,4.9 [--scale 0.5]
    python run.py sheet   PROJECT [--every 0.5] [--scale 0.25]
    python run.py voice   PROJECT
    python run.py audio   PROJECT
    python run.py preview PROJECT            # half-res, 30 fps, full mix
    python run.py final   PROJECT            # full-res, 60 fps, full mix
    python run.py check   PROJECT [--file out/final.mp4]

Renders are resumable: every 60-frame segment is its own MP4 and is skipped if
it already exists, so a crash or a one-beat change re-renders only that part.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

_FILM = None


def _ffmpeg():
    p = shutil.which("ffmpeg")
    if not p:
        raise SystemExit("ffmpeg not found on PATH. Install it (brew/apt/winget/scoop install ffmpeg).")
    return p


def load_film(project, scale, fps):
    from .film import Film
    return Film(project, scale=scale, fps=fps)


# ---------------------------------------------------------------------------
# stills and contact sheets
# ---------------------------------------------------------------------------

def cmd_stills(a):
    film = load_film(a.project, a.scale, a.fps)
    out = os.path.join(film.dir, "build", "stills")
    os.makedirs(out, exist_ok=True)
    times = [float(x) for x in a.times.split(",")] if a.times else _key_times(film)
    paths = []
    for t in times:
        t0 = time.time()
        img = film.frame(t)
        p = os.path.join(out, f"s_{t:06.2f}.png")
        Image.fromarray(img).save(p)
        paths.append((t, p))
        print(f"  t={t:6.2f}  {time.time() - t0:5.2f}s  {p}")
    sheet = _contact(paths, os.path.join(out, "sheet.png"), cols=a.cols, width=a.thumb)
    print(f"sheet: {sheet}")


def _key_times(film):
    ts = [0.30, film.slam + 0.02, film.slam + 0.12, film.slam + 1.6]
    for sc in film.scenes:
        an = sc.anchors()
        ts += [sc.a + 0.4, an.get("key", sc.a + sc.L / 2) + 0.15, sc.b - 0.1]
    ts += [film.race_t + 0.5, film.hit + 0.03, film.hit + 1.6, film.end_t + 1.0]
    return sorted(set(round(t, 2) for t in ts if 0 <= t < film.duration))


def cmd_sheet(a):
    film = load_film(a.project, a.scale, a.fps)
    out = os.path.join(film.dir, "build", "sheet")
    os.makedirs(out, exist_ok=True)
    ts = np.arange(0, film.duration, a.every)
    paths = []
    for t in ts:
        p = os.path.join(out, f"f_{t:06.2f}.png")
        Image.fromarray(film.frame(float(t))).save(p)
        paths.append((float(t), p))
    print(_contact(paths, os.path.join(out, "sheet.png"), cols=a.cols, width=a.thumb))


def _contact(paths, dest, cols=4, width=480):
    ims = []
    for t, p in paths:
        im = Image.open(p).convert("RGB")
        im = im.resize((width, int(im.height * width / im.width)), Image.LANCZOS)
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, 64, 20], fill=(0, 0, 0))
        d.text((5, 4), f"{t:5.2f}", fill=(255, 220, 0))
        ims.append(im)
    rows = (len(ims) + cols - 1) // cols
    W, H = ims[0].size
    sheet = Image.new("RGB", (cols * W + (cols - 1) * 4, rows * H + (rows - 1) * 4), (40, 40, 40))
    for i, im in enumerate(ims):
        sheet.paste(im, ((i % cols) * (W + 4), (i // cols) * (H + 4)))
    sheet.save(dest)
    return dest


# ---------------------------------------------------------------------------
# video
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# resource safety: never take the machine down with the render
# ---------------------------------------------------------------------------

def mem_available_gb():
    """Physical memory the OS can hand out right now (free + reclaimable), in GB."""
    try:
        if sys.platform == "win32":
            import ctypes

            class MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            m = MS()
            m.dwLength = ctypes.sizeof(MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return m.ullAvailPhys / 2 ** 30, m.ullTotalPhys / 2 ** 30
        if sys.platform == "darwin":
            total = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout)
            vm = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
            page = 4096
            free = 0
            for ln in vm.splitlines():
                if any(k in ln for k in ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable")):
                    free += int(ln.split(":")[1].strip().rstrip("."))
            return free * page / 2 ** 30, total / 2 ** 30
        info = {}
        for ln in open("/proc/meminfo"):
            k, v = ln.split(":")
            info[k] = int(v.split()[0]) * 1024
        return info.get("MemAvailable", info.get("MemFree", 0)) / 2 ** 30, info["MemTotal"] / 2 ** 30
    except Exception:
        return 4.0, 8.0


def auto_workers(W, H, requested=None):
    """Workers that fit in RAM with a safety reserve, capped by CPU. Each 1080p worker needs ~0.9 GB."""
    cpus = os.cpu_count() or 4
    avail, total = mem_available_gb()
    per = 0.35 + 0.55 * (W * H) / (1920 * 1080)          # GB per worker (measured, with headroom)
    reserve = max(1.5, 0.12 * total)                      # leave the user's apps room to breathe
    fit = int(max(0.0, avail - reserve) // per)
    cap = max(1, min(cpus // 2, 8))
    n = max(1, min(fit, cap))
    if requested:
        n = max(1, min(int(requested), max(n, 1) if fit < int(requested) else int(requested)))
    print(f"[render] RAM available {avail:.1f} of {total:.1f} GB -> {n} worker(s) "
          f"(~{per:.1f} GB each, {reserve:.1f} GB kept free, {cpus} CPUs)")
    if fit < 1:
        print("[render] memory is tight: rendering with 1 worker. Closing browsers will make it faster.")
    return n


def _lower_priority():
    try:
        if sys.platform == "win32":
            import ctypes
            BELOW_NORMAL = 0x00004000
            ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), BELOW_NORMAL)
        else:
            os.nice(10)
    except Exception:
        pass


def _init_worker(project, scale, fps):
    global _FILM
    _lower_priority()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import cv2
    cv2.setNumThreads(1)
    _FILM = load_film(project, scale, fps)


def _render_segment(job):
    seg_path, f0, f1, crf, preset = job
    film = _FILM
    tmp = seg_path + ".part.mp4"
    cmd = [_ffmpeg(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{film.W}x{film.H}", "-r", str(film.fps), "-i", "-",
           "-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p",
           "-threads", "2", "-g", str(film.fps), "-bf", "2", tmp]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(f0, f1):
        p.stdin.write(film.frame(i / film.fps, i).tobytes())
    p.stdin.close()
    if p.wait() != 0:
        raise RuntimeError(f"ffmpeg failed on {seg_path}")
    os.replace(tmp, seg_path)
    return seg_path


def render_video(project, scale, fps, workers, seg, crf, preset, tag, only=None):
    _lower_priority()
    film = load_film(project, scale, fps)  # warms caches (logo raster) before forking
    workers = auto_workers(film.W, film.H, workers)
    segdir = os.path.join(film.dir, "build", f"segs_{tag}")
    os.makedirs(segdir, exist_ok=True)
    jobs = []
    for k, f0 in enumerate(range(0, film.nframes, seg)):
        f1 = min(film.nframes, f0 + seg)
        sp = os.path.join(segdir, f"seg_{k:04d}.mp4")
        if only is not None and not (only[0] <= f0 / fps <= only[1] or only[0] <= f1 / fps <= only[1]):
            continue
        if os.path.exists(sp):
            if only is None:
                continue
            os.remove(sp)  # --only forces these segments to re-render
        jobs.append((sp, f0, f1, crf, preset))
    print(f"[render] {film.W}x{film.H} @ {fps} fps, {film.nframes} frames, {len(jobs)} segments to do, "
          f"{workers} workers")
    t0 = time.time()
    if jobs:
        ctx = mp.get_context("spawn")
        with ctx.Pool(workers, initializer=_init_worker, initargs=(project, scale, fps)) as pool:
            for i, sp in enumerate(pool.imap_unordered(_render_segment, jobs)):
                print(f"  [{i + 1}/{len(jobs)}] {os.path.basename(sp)}  {time.time() - t0:6.1f}s", flush=True)
    segs = sorted(os.path.join(segdir, s) for s in os.listdir(segdir) if s.endswith(".mp4") and "part" not in s)
    lst = os.path.join(segdir, "list.txt")
    with open(lst, "w", encoding="utf-8") as fh:
        for s in segs:
            fh.write(f"file '{os.path.abspath(s).replace(os.sep, '/')}'\n")
    out = os.path.join(film.dir, "build", f"video_{tag}.mp4")
    subprocess.run([_ffmpeg(), "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", out],
                   check=True)
    return film, out


def cmd_video(a, tag, scale, fps, crf, preset):
    film, video = render_video(a.project, scale, fps, a.workers, a.seg, crf, preset, tag,
                               only=_parse_range(a.only))
    mix = os.path.join(film.dir, "build", "mix.wav")
    if not a.no_audio:
        from .audio import build_mix
        build_mix(film, mix)
    outdir = os.path.join(film.dir, "out")
    os.makedirs(outdir, exist_ok=True)
    name = film.brief.get("output_name") or film.name.lower().replace(" ", "_")
    dest = os.path.join(outdir, f"{name}_{tag}.mp4")
    mux(video, None if a.no_audio else mix, dest, crf=crf)
    print(f"done: {dest}")
    if tag == "final":
        log_history(film, dest)
    check_file(dest, film)


HISTORY = os.path.join(os.path.expanduser("~"), ".promo-video", "history.jsonl")


def log_history(film, path):
    """Remember the creative choices of every final film so the next one can differ."""
    b = film.brief
    rec = dict(date=time.strftime("%Y-%m-%d %H:%M"), product=film.name, file=path, aspect=b.get("aspect", "16:9"),
               format=film.format, ending=film.ending,
               duration=round(film.duration, 1), opening=b.get("opening", "slam"), hook=b.get("hook", "converge"),
               stage=b.get("stage", {}).get("mode", "light"), mood=b.get("music", {}).get("mood", "hype")
               if not b.get("music", {}).get("file") else "own track",
               score=getattr(film, "score_info", None), voice=b.get("voice", {}).get("voice", b.get("voice", {}).get("engine")),
               bpm=film.bpm, scenes=[sc.kind for sc in film.scenes], transitions=[sc.transition for sc in film.scenes])
    os.makedirs(os.path.dirname(HISTORY), exist_ok=True)
    with open(HISTORY, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")


def cmd_history(a):
    if not os.path.exists(HISTORY):
        print("no films yet")
        return
    rows = [json.loads(l) for l in open(HISTORY, encoding="utf-8") if l.strip()][-a.n:]
    for r in rows:
        print(f"{r['date']}  {r['product']:<18} format={r.get('format', '?'):<9} end={r.get('ending', 'race'):<6} "
              f"{r['aspect']:<5} open={r['opening']:<5} hook={r['hook']:<8} "
              f"stage={r['stage']:<5} mood={r['mood']:<10} voice={r.get('voice')}  scenes={'>'.join(r['scenes'])}")


def _parse_range(s):
    if not s:
        return None
    a, b = s.split("-")
    return float(a), float(b)


def mux(video, mix, dest, crf=18):
    cmd = [_ffmpeg(), "-y", "-loglevel", "error", "-i", video]
    if mix:
        cmd += ["-i", mix, "-map", "0:v", "-map", "1:a", "-c:a", "aac", "-b:a", "320k", "-ar", "48000"]
    cmd += ["-c:v", "copy", "-movflags", "+faststart", "-shortest", dest]
    subprocess.run(cmd, check=True)
    if mix:
        tp = true_peak(dest)
        if tp is not None and tp > -1.0:
            # AAC overshoot: pull the mix down and encode again
            from .audio import apply_gain_wav
            apply_gain_wav(mix, -(tp + 1.1))
            subprocess.run(cmd, check=True)


def true_peak(path):
    r = subprocess.run([_ffmpeg(), "-hide_banner", "-nostats", "-i", path, "-filter_complex",
                        "ebur128=peak=true", "-f", "null", "-"], capture_output=True, text=True)
    lines = [ln for ln in r.stderr.splitlines() if "Peak:" in ln]
    try:
        return float(lines[-1].split("Peak:")[1].split("dBFS")[0])
    except Exception:
        return None


def loudness(path):
    r = subprocess.run([_ffmpeg(), "-hide_banner", "-nostats", "-i", path, "-filter_complex", "ebur128=peak=true",
                        "-f", "null", "-"], capture_output=True, text=True)
    out = {}
    for ln in r.stderr.splitlines():
        ln = ln.strip()
        if ln.startswith("I:"):
            out["lufs"] = float(ln.split()[1])
        if ln.startswith("Peak:"):
            out["true_peak"] = float(ln.split()[1])
    return out


def check_file(path, film=None):
    r = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                        "stream=nb_read_frames,width,height,r_frame_rate:format=duration", "-of", "json", path],
                       capture_output=True, text=True)
    info = json.loads(r.stdout or "{}")
    st = (info.get("streams") or [{}])[0]
    print(f"[check] {os.path.basename(path)}: {st.get('width')}x{st.get('height')} "
          f"{st.get('r_frame_rate')} frames={st.get('nb_read_frames')} duration={info.get('format', {}).get('duration')}")
    ld = loudness(path)
    if ld:
        print(f"[check] loudness {ld.get('lufs')} LUFS, true peak {ld.get('true_peak')} dBTP")


def cmd_check(a):
    film = load_film(a.project, 0.25, 30)
    path = a.file
    if not path:
        outs = sorted((os.path.join(film.dir, "out", f) for f in os.listdir(os.path.join(film.dir, "out"))),
                      key=os.path.getmtime)
        path = outs[-1]
    check_file(path, film)
    # stutter profile: frame-difference jerk; spikes away from known hits are suspects
    w, h = 160, 90
    r = subprocess.run([_ffmpeg(), "-loglevel", "error", "-i", path, "-vf", f"scale={w}:{h}", "-f", "rawvideo",
                        "-pix_fmt", "gray", "-"], capture_output=True)
    fr = np.frombuffer(r.stdout, np.uint8)
    n = len(fr) // (w * h)
    frames = fr[: n * w * h].reshape(n, h, w).astype(np.int16)
    d = np.abs(np.diff(frames, axis=0)).mean(axis=(1, 2))
    fps = n / film.duration if film.duration else 60
    known = film.hits() + [film.race_t, film.end_t, film.fade_t] + [sc.a for sc in film.scenes]
    sus = []
    for i in range(len(d)):
        nb = np.concatenate([d[max(0, i - 6):i], d[i + 1:i + 7]])
        med = max(float(np.median(nb)) if len(nb) else 0.0, 0.25)
        if d[i] > 2.2 * med and d[i] - med > 0.8:
            t = (i + 1) / fps
            whips = [sc.a for sc in film.scenes if sc.transition == "whip"]
            if min(abs(t - k) for k in known) > 0.12 and all(abs(t - w) > 0.3 for w in whips):
                sus.append(round(t, 2))
    print(f"[check] single-frame jumps away from hits/cuts: {len(sus)}" + (f" at {sus[:12]}" if sus else " (clean)"))


def cmd_voice(a):
    from .film import Film
    from .voice import build_voice
    film = Film(a.project, scale=0.25, fps=30)
    build_voice(film, force=a.force)


def cmd_audio(a):
    from .audio import build_mix
    film = load_film(a.project, 0.25, 30)
    p = build_mix(film, os.path.join(film.dir, "build", "mix.wav"))
    print(p)


def cmd_music_info(a):
    from .audio import analyze_track
    print(json.dumps(analyze_track(a.file), indent=2))


def cmd_doctor(a):
    ok = True
    for mod in ("numpy", "cv2", "PIL", "scipy", "requests"):
        try:
            __import__(mod)
            print(f"  ok   {mod}")
        except Exception:
            ok = False
            print(f"  MISSING {mod}")
    for mod in ("edge_tts", "pyloudnorm"):
        try:
            __import__(mod)
            print(f"  ok   {mod}")
        except Exception:
            print(f"  opt  {mod} (pip install {mod.replace('_', '-')})")
    print(f"  {'ok  ' if shutil.which('ffmpeg') else 'MISSING'} ffmpeg")
    from .brand import find_browser
    b = find_browser()
    print(f"  {'ok  ' if b else 'opt '} browser {b or '(needed for SVG logos + site capture; or pip install cairosvg)'}")
    avail, total = mem_available_gb()
    print(f"  cpus {os.cpu_count()}, RAM {avail:.1f} GB free of {total:.1f} GB")
    auto_workers(1920, 1080)
    sys.exit(0 if ok else 1)


def cmd_capture(a):
    from .capture import capture
    print(json.dumps(capture(a.url, a.out), indent=2)[:4000])


def main(argv=None):
    ap = argparse.ArgumentParser(prog="promo-video")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def proj(p):
        p.add_argument("project")
        p.add_argument("--scale", type=float, default=0.5)
        p.add_argument("--fps", type=int, default=60)
        return p

    p = proj(sub.add_parser("stills"))
    p.add_argument("--times", default="")
    p.add_argument("--cols", type=int, default=4)
    p.add_argument("--thumb", type=int, default=480)
    p = proj(sub.add_parser("sheet"))
    p.add_argument("--every", type=float, default=0.5)
    p.add_argument("--cols", type=int, default=6)
    p.add_argument("--thumb", type=int, default=320)
    for name in ("preview", "final"):
        p = sub.add_parser(name)
        p.add_argument("project")
        p.add_argument("--workers", type=int, default=0, help="0 = automatic (fits free RAM and CPUs)")
        p.add_argument("--seg", type=int, default=60)
        p.add_argument("--only", default="", help="re-render only segments touching this range, e.g. 9-12")
        p.add_argument("--no-audio", action="store_true")
    p = sub.add_parser("check")
    p.add_argument("project")
    p.add_argument("--file", default="")
    p = sub.add_parser("voice")
    p.add_argument("project")
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("audio")
    p.add_argument("project")
    sub.add_parser("doctor")
    p = sub.add_parser("history")
    p.add_argument("-n", type=int, default=10)
    p = sub.add_parser("music-info")
    p.add_argument("file")
    p = sub.add_parser("capture")
    p.add_argument("url")
    p.add_argument("out")
    a = ap.parse_args(argv)
    if a.cmd == "stills":
        cmd_stills(a)
    elif a.cmd == "sheet":
        cmd_sheet(a)
    elif a.cmd == "preview":
        cmd_video(a, "preview", 0.5, 30, 20, "medium")
    elif a.cmd == "final":
        cmd_video(a, "final", 1.0, 60, 16, "slow")
    elif a.cmd == "check":
        cmd_check(a)
    elif a.cmd == "voice":
        cmd_voice(a)
    elif a.cmd == "audio":
        cmd_audio(a)
    elif a.cmd == "doctor":
        cmd_doctor(a)
    elif a.cmd == "history":
        cmd_history(a)
    elif a.cmd == "music-info":
        cmd_music_info(a)
    elif a.cmd == "capture":
        cmd_capture(a)
