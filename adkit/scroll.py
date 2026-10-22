"""Scroll-through capture over the Chrome DevTools protocol.

A plain headless screenshot only sees the first viewport, and sections that animate in on scroll
stay invisible. This drives a real (headless) Chrome tab: it scrolls the page a viewport at a time,
lets each section's reveal animation play, and saves every stop as a 2x PNG. ``hold`` also takes a
time-lapse at one scroll position, for UI that types or streams (a live demo, a chat, a ticker).

It only scrolls and looks: it never clicks, types or submits anything.
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import shutil
import subprocess
import tempfile
import time

from .brand import find_browser


async def _run(url, out_dir, name, w, h, dpr, wait, settle, step, hold, port):
    import aiohttp
    prof = tempfile.mkdtemp(prefix="promo_cdp_")
    br = subprocess.Popen([find_browser(), "--headless=new", f"--remote-debugging-port={port}",
                           f"--user-data-dir={prof}", "--hide-scrollbars", "--mute-audio", "--no-first-run",
                           "--use-angle=swiftshader", "--enable-unsafe-swiftshader", f"--window-size={w},{h}",
                           "about:blank"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    saved = []
    try:
        async with aiohttp.ClientSession() as s:
            page = None
            for _ in range(80):
                try:
                    async with s.get(f"http://127.0.0.1:{port}/json") as r:
                        page = [t for t in await r.json() if t["type"] == "page"][0]
                        break
                except Exception:
                    await asyncio.sleep(0.25)
            if page is None:
                raise RuntimeError("Chrome did not open a debugging port")
            ws = await s.ws_connect(page["webSocketDebuggerUrl"], max_msg_size=0)
            n = 0

            async def cmd(method, **params):
                nonlocal n
                n += 1
                mid = n
                await ws.send_str(json.dumps({"id": mid, "method": method, "params": params}))
                while True:
                    m = json.loads((await ws.receive()).data)
                    if m.get("id") == mid:
                        return m.get("result", {})

            async def js(expr):
                r = await cmd("Runtime.evaluate", expression=expr, returnByValue=True)
                return r.get("result", {}).get("value")

            async def shot(path):
                r = await cmd("Page.captureScreenshot", format="png")
                with open(path, "wb") as f:
                    f.write(base64.b64decode(r["data"]))
                saved.append(path)

            await cmd("Emulation.setDeviceMetricsOverride", width=w, height=h, deviceScaleFactor=dpr, mobile=False)
            await cmd("Page.enable")
            await cmd("Page.navigate", url=url)
            await asyncio.sleep(wait)
            os.makedirs(out_dir, exist_ok=True)
            y, i = 0, 0
            H = int(await js("document.documentElement.scrollHeight") or h)
            while y < H and i < 40:
                await js(f"window.scrollTo(0,{y})")
                await asyncio.sleep(settle)
                await shot(os.path.join(out_dir, f"{name}_{i:02d}.png"))
                y += int(h * step)
                i += 1
                H = int(await js("document.documentElement.scrollHeight") or H)
            if hold:
                hy, hn, every = hold
                await js(f"window.scrollTo(0,{int(hy)})")
                for k in range(int(hn)):
                    await asyncio.sleep(every)
                    await shot(os.path.join(out_dir, f"{name}_hold_{k:02d}.png"))
    finally:
        br.terminate()
        time.sleep(0.5)
        shutil.rmtree(prof, ignore_errors=True)
    return saved


def scroll_capture(url, out_dir, name="page", w=1440, h=900, dpr=2.0, wait=5.0, settle=2.2, step=0.85,
                   hold=None, port=9339):
    """Save viewport shots down the whole page (and an optional time-lapse ``hold=(y, n, every_s)``)."""
    if not find_browser():
        raise RuntimeError("No Chrome/Edge found for the scroll capture")
    return asyncio.run(_run(url, out_dir, name, w, h, dpr, wait, settle, step, hold, port))
