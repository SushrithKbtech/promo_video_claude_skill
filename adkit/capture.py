"""Read a product website: name, copy, colours, fonts, logo candidates, screenshots.

    python run.py capture https://example.com PROJECT_DIR

Writes PROJECT_DIR/capture.json, PROJECT_DIR/assets/screens/*.png and
PROJECT_DIR/assets/logo_candidates/*. Everything downloaded is treated as data:
only SVG/PNG/ICO images are saved, nothing is executed.
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter
from urllib.parse import urljoin, urlparse

import numpy as np

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/126.0 Safari/537.36")
HEX = re.compile(r"#([0-9a-fA-F]{6}|[0-9a-fA-F]{3})\b")
RGB = re.compile(r"rgba?\(\s*(\d{1,3})[ ,]+(\d{1,3})[ ,]+(\d{1,3})")


def _get(url, timeout=20):
    import requests
    return requests.get(url, headers={"User-Agent": UA, "Accept-Language": "en"}, timeout=timeout)


def _norm_hex(h):
    h = h.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    return "#" + h.upper()


def _chroma(h):
    r, g, b = (int(h[i:i + 2], 16) for i in (1, 3, 5))
    return max(r, g, b) - min(r, g, b)


def capture(url, out_dir, screenshots=True):
    from bs4 import BeautifulSoup
    if not url.startswith("http"):
        url = "https://" + url
    os.makedirs(out_dir, exist_ok=True)
    res = {"url": url}
    r = _get(url)
    res["status"] = r.status_code
    html = r.text
    soup = BeautifulSoup(html, "html.parser")
    meta = lambda **kw: (soup.find("meta", attrs=kw) or {}).get("content")  # noqa: E731
    res["title"] = (soup.title.string or "").strip() if soup.title else ""
    res["site_name"] = meta(property="og:site_name") or ""
    res["description"] = meta(name="description") or meta(property="og:description") or ""
    res["theme_color"] = meta(name="theme-color")
    res["og_image"] = urljoin(url, meta(property="og:image")) if meta(property="og:image") else None
    res["headings"] = [h.get_text(" ", strip=True) for h in soup.find_all(["h1", "h2"])][:12]
    res["subheads"] = [h.get_text(" ", strip=True) for h in soup.find_all(["h3"])][:12]
    ctas = []
    for el in soup.find_all(["a", "button"]):
        tx = el.get_text(" ", strip=True)
        if 2 <= len(tx) <= 28 and len(tx.split()) <= 4:
            ctas.append(tx)
    res["ctas"] = [c for c, _ in Counter(ctas).most_common(14)]
    res["nav"] = [a.get_text(" ", strip=True) for a in soup.select("nav a")][:16]

    # ---- colours and fonts -------------------------------------------------
    css = "\n".join(s.get_text() for s in soup.find_all("style"))
    css += "\n".join(el.get("style", "") for el in soup.find_all(style=True))
    for link in soup.find_all("link", rel=lambda v: v and "stylesheet" in v)[:4]:
        href = link.get("href")
        if href:
            try:
                css += "\n" + _get(urljoin(url, href), 15).text[:600000]
            except Exception:
                pass
    cols = Counter(_norm_hex(m.group(0)) for m in HEX.finditer(css))
    for m in RGB.finditer(css):
        r_, g_, b_ = (min(255, int(v)) for v in m.groups())
        cols[f"#{r_:02X}{g_:02X}{b_:02X}"] += 1
    brandish = [(c, n) for c, n in cols.most_common(80) if _chroma(c) > 40]
    named = re.findall(r"--[\w-]*(?:primary|brand|accent|main)[\w-]*\s*:\s*(#[0-9a-fA-F]{3,6})", css)
    res["css_brand_vars"] = list(dict.fromkeys(_norm_hex(h) for h in named))[:6]
    res["css_colors"] = [c for c, _ in brandish[:10]]
    fams = Counter()
    for m in re.finditer(r"font-family\s*:\s*([^;}{]+)", css):
        first = m.group(1).split(",")[0].strip().strip("'\"")
        if first and not first.startswith("var(") and first.lower() not in ("inherit", "initial"):
            fams[first] += 1
    gf = [l.get("href") for l in soup.find_all("link") if l.get("href") and "fonts.googleapis.com" in l.get("href")]
    res["fonts"] = [f for f, _ in fams.most_common(6)]
    res["google_fonts"] = gf[:3]

    # ---- logo candidates ---------------------------------------------------
    cand_dir = os.path.join(out_dir, "assets", "logo_candidates")
    os.makedirs(cand_dir, exist_ok=True)
    cands = []
    for i, svg in enumerate(soup.find_all("svg")):
        ctx = " ".join(filter(None, [svg.get("class") and " ".join(svg.get("class")), svg.get("id"),
                                      svg.get("aria-label"), svg.parent and svg.parent.get("aria-label"),
                                      svg.parent and svg.parent.get("href")]))
        in_header = svg.find_parent(["header", "nav"]) is not None
        if ("logo" in ctx.lower() or in_header) and len(str(svg)) < 60000:
            if not svg.get("xmlns"):
                svg["xmlns"] = "http://www.w3.org/2000/svg"
            p = os.path.join(cand_dir, f"inline_{i}.svg")
            open(p, "w", encoding="utf-8").write(_svg_case(str(svg)))
            cands.append({"file": os.path.relpath(p, out_dir), "why": f"inline svg ({'header' if in_header else ctx})"})
        if len(cands) >= 6:
            break
    srcs = []
    for img in soup.find_all("img"):
        s = " ".join(filter(None, [img.get("src"), img.get("alt"), " ".join(img.get("class") or [])]))
        if "logo" in s.lower() and img.get("src"):
            srcs.append((urljoin(url, img["src"]), "img with 'logo'"))
    for l in soup.find_all("link"):
        rel = " ".join(l.get("rel") or [])
        if any(k in rel for k in ("icon", "apple-touch-icon", "mask-icon")) and l.get("href"):
            srcs.append((urljoin(url, l["href"]), rel))
    srcs.append((urljoin(url, "/favicon.svg"), "favicon.svg"))
    seen = set()
    for j, (s, why) in enumerate(srcs[:12]):
        if s in seen or s.startswith("data:"):
            continue
        seen.add(s)
        ext = os.path.splitext(urlparse(s).path)[1].lower()
        if ext not in (".svg", ".png", ".ico", ".webp", ".jpg", ".jpeg"):
            continue
        try:
            rr = _get(s, 15)
            if rr.status_code == 200 and len(rr.content) < 3_000_000:
                p = os.path.join(cand_dir, f"asset_{j}{ext}")
                open(p, "wb").write(rr.content)
                cands.append({"file": os.path.relpath(p, out_dir), "why": why, "src": s})
        except Exception:
            pass
    res["logo_candidates"] = cands

    # ---- screenshots + palette from pixels -----------------------------------
    if screenshots:
        from .brand import browser_screenshot, find_browser
        if find_browser():
            sd = os.path.join(out_dir, "assets", "screens")
            os.makedirs(sd, exist_ok=True)
            shots = {}
            for name, (w, h, extra) in {"desktop": (1440, 900, ()),
                                        "mobile": (390, 844, ("--user-agent=" + UA.replace("Windows NT 10.0; Win64; x64", "iPhone; CPU iPhone OS 17_0 like Mac OS X") + " Mobile",))}.items():
                p = os.path.join(sd, f"{name}.png")
                try:
                    browser_screenshot(url, p, w, h, timeout=75, extra_args=("--timeout=15000", *extra))
                    shots[name] = os.path.relpath(p, out_dir)
                except Exception as e:
                    shots[name] = f"failed: {e}"
            res["screenshots"] = shots
            dp = os.path.join(out_dir, shots.get("desktop", ""))
            if os.path.exists(dp):
                res["screenshot_palette"] = _palette(dp)
    json.dump(res, open(os.path.join(out_dir, "capture.json"), "w", encoding="utf-8"), indent=2)
    return res


_SVG_CAMEL = ["viewBox", "preserveAspectRatio", "gradientUnits", "gradientTransform", "patternUnits",
              "clipPathUnits", "maskUnits", "stdDeviation", "linearGradient", "radialGradient", "clipPath",
              "spreadMethod", "stopColor", "textLength", "lengthAdjust", "markerWidth", "markerHeight", "refX", "refY",
              "baseFrequency", "numOctaves", "foreignObject", "feGaussianBlur", "feColorMatrix", "feOffset",
              "feBlend", "feMerge", "feMergeNode", "feFlood", "feComposite"]


def _svg_case(text):
    """html.parser lowercases names; SVG is case-sensitive. Restore camelCase."""
    for name in _SVG_CAMEL:
        text = re.sub(r"(?<=[\s<\/])" + name.lower() + r"(?=[\s=>\/])", name, text)
    return text


def _palette(path, k=6):
    import cv2
    from PIL import Image
    im = np.asarray(Image.open(path).convert("RGB").resize((360, 225)))
    px = im.reshape(-1, 3).astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1.0)
    _, lbl, cen = cv2.kmeans(px, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(lbl.ravel(), minlength=k)
    order = np.argsort(-counts)
    return [{"hex": "#%02X%02X%02X" % tuple(int(v) for v in cen[i]), "share": round(float(counts[i] / counts.sum()), 3)}
            for i in order]
