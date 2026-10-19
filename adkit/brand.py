"""Brand assets: the logo (facets, mask, rim contour), fonts and text sprites.

The logo is never redrawn or recoloured. It is rasterised once at high
resolution, split into facet layers whose sum is exactly the original, and then
only moved, scaled, faded or masked.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path as FsPath

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .core import hex_srgb, resample, to_hex
from .gfx import Sprite

# ---------------------------------------------------------------------------
# headless browser (used to rasterise SVG and to capture websites)
# ---------------------------------------------------------------------------

_BROWSER_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
]


def find_browser():
    env = os.environ.get("SPECAD_BROWSER")
    if env and os.path.exists(env):
        return env
    for name in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "chrome",
                 "msedge", "microsoft-edge"):
        p = shutil.which(name)
        if p:
            return p
    for p in _BROWSER_CANDIDATES:
        if os.path.exists(p):
            return p
    return None


def browser_screenshot(url_or_html, out_png, width, height, transparent=False, timeout=90,
                       extra_args=()):
    """Screenshot a URL (or a local HTML file path) with headless Chrome/Edge."""
    br = find_browser()
    if not br:
        raise RuntimeError("No Chrome/Edge/Chromium found. Install one or set SPECAD_BROWSER.")
    target = url_or_html
    if os.path.exists(url_or_html):
        target = FsPath(url_or_html).resolve().as_uri()
    args = [br, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--no-first-run",
            "--no-default-browser-check", "--mute-audio", f"--window-size={width},{height}",
            f"--screenshot={os.path.abspath(out_png)}", *extra_args]
    if transparent:
        args.append("--default-background-color=00000000")
    args.append(target)
    subprocess.run(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=timeout)
    if not os.path.exists(out_png):
        raise RuntimeError(f"browser screenshot failed for {url_or_html}")
    return out_png


def rasterize_svg(svg_path_or_text, height_px=1600, out_png=None):
    """SVG -> RGBA uint8 array at ``height_px`` tall, aspect preserved.

    Tries cairosvg first (if installed), then a headless browser.
    """
    text = svg_path_or_text
    if os.path.exists(str(svg_path_or_text)):
        text = FsPath(svg_path_or_text).read_text(encoding="utf-8", errors="ignore")
    vw, vh = _svg_size(text)
    width_px = max(1, int(round(height_px * vw / vh)))
    try:
        import cairosvg  # noqa: F401
        png = cairosvg.svg2png(bytestring=text.encode("utf-8"), output_width=width_px, output_height=height_px)
        import io
        return np.asarray(Image.open(io.BytesIO(png)).convert("RGBA"))
    except Exception:
        pass
    tmp = tempfile.mkdtemp(prefix="specad_svg_")
    svg_file = os.path.join(tmp, "logo.svg")
    FsPath(svg_file).write_text(text, encoding="utf-8")
    html = os.path.join(tmp, "page.html")
    FsPath(html).write_text(
        "<!doctype html><html><head><style>html,body{margin:0;background:transparent;overflow:hidden}"
        f"img{{display:block;width:{width_px}px;height:{height_px}px}}</style></head>"
        f"<body><img src='logo.svg'></body></html>", encoding="utf-8")
    png = out_png or os.path.join(tmp, "logo.png")
    browser_screenshot(html, png, width_px, height_px, transparent=True)
    arr = np.asarray(Image.open(png).convert("RGBA"))[:height_px, :width_px]
    shutil.rmtree(tmp, ignore_errors=True)
    return arr


def _svg_size(text):
    root = ET.fromstring(text.encode("utf-8"))
    vb = root.get("viewBox") or root.get("viewbox")
    if vb:
        parts = [float(v) for v in re.split(r"[ ,]+", vb.strip())]
        return parts[2], parts[3]
    def num(v):
        return float(re.sub(r"[^0-9.]", "", v)) if v else None
    w, h = num(root.get("width")), num(root.get("height"))
    return (w or 100.0), (h or 100.0)


def svg_drawable_children(text):
    """Split an SVG into one SVG string per top-level drawable element.

    Keeps <defs>/<style> in every copy so gradients still resolve. Used for
    path-accurate facets (one facet per shape).
    """
    ET.register_namespace("", "http://www.w3.org/2000/svg")
    ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
    root = ET.fromstring(text.encode("utf-8"))
    keep_tags = ("defs", "style", "title", "desc", "metadata", "clipPath", "linearGradient",
                 "radialGradient", "mask", "filter", "symbol")
    children = list(root)
    shared = [c for c in children if c.tag.split("}")[-1] in keep_tags]
    drawable = [c for c in children if c.tag.split("}")[-1] not in keep_tags]
    # unwrap a single top-level <g> wrapper
    while len(drawable) == 1 and drawable[0].tag.split("}")[-1] == "g" and len(list(drawable[0])):
        g = drawable[0]
        drawable = list(g)
        if g.attrib:
            # carry the group's transform/fill onto each child via a wrapper
            drawable = [_wrap(g, c) for c in drawable]
    out = []
    for d in drawable:
        r = ET.Element(root.tag, root.attrib)
        for s in shared:
            r.append(s)
        r.append(d)
        out.append(ET.tostring(r, encoding="unicode"))
    return out


def _wrap(g, child):
    w = ET.Element(g.tag, g.attrib)
    w.append(child)
    return w


# ---------------------------------------------------------------------------
# the logo
# ---------------------------------------------------------------------------

class Logo:
    """A logo split into facet sprites, plus its mask, outer contour and colours.

    facets: "paths"  one facet per SVG shape (best when shapes are colour facets)
            "colors" k-means by colour (works for PNGs and gradient logos)
            "none"   a single layer (monochrome marks)
    """

    def __init__(self, src, design_h=380.0, facets="auto", k=None, master_px=1400, cache_dir=None,
                 merge_paths_by_color=True):
        self.src = str(src)
        self.design_h = float(design_h)
        cache_dir = cache_dir or os.path.join(os.path.dirname(os.path.abspath(self.src)), ".cache")
        os.makedirs(cache_dir, exist_ok=True)
        stem = FsPath(self.src).stem
        is_svg = self.src.lower().endswith(".svg")
        full_png = os.path.join(cache_dir, f"{stem}_{master_px}.png")
        if is_svg:
            if not os.path.exists(full_png):
                Image.fromarray(rasterize_svg(self.src, master_px)).save(full_png)
        else:
            im = Image.open(self.src).convert("RGBA")
            im = _trim(im)
            if im.height != master_px:
                im = im.resize((max(1, round(im.width * master_px / im.height)), master_px), Image.LANCZOS)
            im.save(full_png)
        full = np.asarray(Image.open(full_png).convert("RGBA"))
        full, box = _trim_arr(full, pad=4)
        self.rgba8 = full
        self.h_px, self.w_px = full.shape[:2]
        ppd = self.h_px / self.design_h
        self.ppd = ppd
        self.sprite = Sprite.from_rgba8(full, ppd)
        alpha = full[..., 3].astype(np.float32) / 255

        if facets == "auto":
            facets = "paths" if is_svg else "colors"
        layers = None
        if facets == "paths" and is_svg:
            layers = self._facets_from_paths(cache_dir, stem, master_px, box, merge_paths_by_color)
        if layers is None and facets in ("paths", "colors"):
            layers = _facets_by_color(full, k)
        if not layers or facets == "none":
            layers = [alpha]
        # soft weights that sum to the original alpha -> no seams when re-assembled
        tot = np.sum(layers, axis=0)
        tot = np.where(tot > 1e-6, tot, 1)
        self.facets = []
        cy0, cx0 = self.h_px / 2, self.w_px / 2
        for wgt in layers:
            wgt = np.clip(wgt / tot, 0, 1) * alpha
            if wgt.sum() < 50:
                continue
            arr = full.copy()
            arr[..., 3] = (wgt * 255).astype(np.uint8)
            ys, xs = np.nonzero(wgt > 0.5)
            cx, cy = (xs.mean(), ys.mean()) if len(xs) else (cx0, cy0)
            rgb = full[..., :3][wgt > 0.5].astype(np.float32).mean(0) / 255 if len(xs) else np.ones(3)
            self.facets.append(dict(
                sprite=Sprite.from_rgba8(arr, ppd),
                center=np.array([(cx - cx0) / ppd, (cy - cy0) / ppd]),  # design px from logo centre
                color=_vivid(to_hex(rgb)),
                mask=wgt,
            ))
        # sort facets clockwise from the top so lights are assigned predictably
        self.facets.sort(key=lambda f: (np.arctan2(f["center"][0], -f["center"][1]) + 2 * np.pi + 0.05) % (2 * np.pi))
        self.colors = [f["color"] for f in self.facets]
        self.mask = alpha
        self.contour = self._outer_contour(alpha, ppd)
        self.contour_colors = self._contour_colors(full, alpha, ppd)

    # -- facets --------------------------------------------------------------
    def _facets_from_paths(self, cache_dir, stem, master_px, box, merge):
        text = FsPath(self.src).read_text(encoding="utf-8", errors="ignore")
        parts = svg_drawable_children(text)
        if len(parts) < 2:
            return None
        layers, cols = [], []
        for i, p in enumerate(parts):
            fp = os.path.join(cache_dir, f"{stem}_{master_px}_part{i}.png")
            if not os.path.exists(fp):
                Image.fromarray(rasterize_svg(p, master_px)).save(fp)
            arr = np.asarray(Image.open(fp).convert("RGBA"))
            arr = arr[box[1]:box[3], box[0]:box[2]]
            a = arr[..., 3].astype(np.float32) / 255
            if a.sum() < 50:
                continue
            col = arr[..., :3][a > 0.5].astype(np.float32).mean(0) if (a > 0.5).any() else np.zeros(3)
            layers.append(a)
            cols.append(col)
        if len(layers) < 2:
            return None
        if merge:
            # shapes of (nearly) the same colour form one facet
            groups = []
            for a, c in zip(layers, cols):
                for g in groups:
                    if np.linalg.norm(g["c"] - c) < 40:
                        g["a"] = np.maximum(g["a"], a)
                        break
                else:
                    groups.append(dict(a=a, c=c))
            layers = [g["a"] for g in groups]
        # occlusion: later shapes cover earlier ones
        out = []
        cover = np.zeros_like(layers[0])
        for a in reversed(layers):
            out.append(a * (1 - cover))
            cover = np.clip(cover + a, 0, 1)
        out.reverse()
        return out if len(out) >= 2 else None

    # -- geometry ------------------------------------------------------------
    def _outer_contour(self, alpha, ppd, n=900):
        m = (alpha > 0.5).astype(np.uint8) * 255
        k = max(3, int(self.h_px * 0.03) | 1)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cs:
            return np.zeros((n, 2))
        c = max(cs, key=cv2.contourArea)[:, 0, :].astype(np.float64)
        pts = resample(c, n, closed=True)
        # start at the top-most point, run clockwise
        i0 = int(np.argmin(pts[:, 1] + 0.001 * np.abs(pts[:, 0] - self.w_px / 2)))
        pts = np.roll(pts, -i0, axis=0)
        area = 0.5 * np.sum(pts[:, 0] * np.roll(pts[:, 1], -1) - np.roll(pts[:, 0], -1) * pts[:, 1])
        if area < 0:  # image coords: positive area == clockwise on screen
            pts = np.vstack([pts[:1], pts[1:][::-1]])
        return (pts - (self.w_px / 2, self.h_px / 2)) / ppd

    def _contour_colors(self, full, alpha, ppd, inset_design=40):
        cols = []
        c = self.contour
        inset = inset_design / self.design_h * self.design_h  # design px
        for p in c:
            d = -p / (np.linalg.norm(p) + 1e-6)
            col = None
            for step in np.linspace(inset * 0.4, inset * 1.6, 6):
                q = (p + d * step) * ppd + (self.w_px / 2, self.h_px / 2)
                x, y = int(np.clip(q[0], 0, self.w_px - 1)), int(np.clip(q[1], 0, self.h_px - 1))
                if alpha[y, x] > 0.6:
                    col = full[y, x, :3]
                    break
            cols.append(col if col is not None else np.array([95, 99, 104]))
        cols = np.array(cols, np.float32) / 255
        # smooth along the contour so colour hand-overs are soft
        k = 15
        pad = np.concatenate([cols[-k:], cols, cols[:k]])
        ker = np.ones(2 * k + 1) / (2 * k + 1)
        sm = np.stack([np.convolve(pad[:, i], ker, mode="same")[k:-k] for i in range(3)], 1)
        return np.array([hex_srgb(_vivid(to_hex(c))) for c in sm])

    @property
    def aspect(self):
        return self.w_px / self.h_px


def _vivid(h):
    """Light colours read better slightly more saturated than flat fills."""
    from .core import saturate_hex, _hex
    import colorsys
    r, g, b = _hex(h)
    hh, ll, ss = colorsys.rgb_to_hls(r, g, b)
    if ss < 0.12:  # neutral stays neutral (becomes slate ink)
        return h
    return saturate_hex(h, 1.15, light=min(max(ll, 0.42), 0.6))


def _trim(im):
    bb = im.getchannel("A").getbbox()
    return im.crop(bb) if bb else im


def _trim_arr(arr, pad=4):
    a = arr[..., 3]
    ys, xs = np.nonzero(a > 8)
    if len(ys) == 0:
        return arr, (0, 0, arr.shape[1], arr.shape[0])
    x0, x1 = max(0, xs.min() - pad), min(arr.shape[1], xs.max() + pad + 1)
    y0, y1 = max(0, ys.min() - pad), min(arr.shape[0], ys.max() + pad + 1)
    return arr[y0:y1, x0:x1], (x0, y0, x1, y1)


def _facets_by_color(full, k=None):
    a = full[..., 3].astype(np.float32) / 255
    sel = a > 0.6
    if sel.sum() < 100:
        return None
    px = full[..., :3][sel].astype(np.float32)
    lab = cv2.cvtColor(px.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    chroma = np.sqrt((lab[:, 1] - 128) ** 2 + (lab[:, 2] - 128) ** 2)
    if np.percentile(chroma, 90) < 12:
        return None  # monochrome mark: one facet
    feats = np.stack([lab[:, 0] * 0.35, lab[:, 1], lab[:, 2]], 1)
    sample = feats[np.random.default_rng(0).choice(len(feats), min(len(feats), 40000), replace=False)]
    best = None
    ks = [k] if k else [2, 3, 4, 5, 6]
    for kk in ks:
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.5)
        comp, lbl, cen = cv2.kmeans(sample, kk, None, crit, 3, cv2.KMEANS_PP_CENTERS)
        score = comp / len(sample)
        if best is None or (k is None and score < best[0] * 0.55):
            best = (score, kk, cen)
    _, kk, cen = best
    allf = cv2.cvtColor(full[..., :3], cv2.COLOR_RGB2LAB).astype(np.float32)
    allf = np.stack([allf[..., 0] * 0.35, allf[..., 1], allf[..., 2]], -1)
    d = np.stack([np.linalg.norm(allf - c, axis=-1) for c in cen], 0)
    wts = np.exp(-(d - d.min(0)) / 6.0)  # soft assignment -> clean internal seams
    wts /= wts.sum(0)
    return [w.astype(np.float32) for w in wts]


# ---------------------------------------------------------------------------
# fonts and text
# ---------------------------------------------------------------------------

_SYSTEM_FONTS = {
    400: ["Inter-Regular.ttf", "segoeui.ttf", "SFNS.ttf", "Helvetica.ttc", "DejaVuSans.ttf", "arial.ttf"],
    500: ["Inter-Medium.ttf", "seguisb.ttf", "segoeui.ttf", "DejaVuSans.ttf", "arial.ttf"],
    700: ["Inter-Bold.ttf", "segoeuib.ttf", "DejaVuSans-Bold.ttf", "arialbd.ttf"],
}
_FONT_DIRS = [r"C:\Windows\Fonts", os.path.expanduser(r"~\AppData\Local\Microsoft\Windows\Fonts"),
              "/System/Library/Fonts", "/Library/Fonts", os.path.expanduser("~/Library/Fonts"),
              "/usr/share/fonts/truetype/dejavu", "/usr/share/fonts", os.path.expanduser("~/.fonts")]


def fetch_google_font(family, out_dir, weights=(400, 500, 700)):
    """Download static TTFs for a Google Fonts family. Returns {weight: path}."""
    import requests
    os.makedirs(out_dir, exist_ok=True)
    fam = family.replace(" ", "+")
    out = {}
    for w in weights:
        dest = os.path.join(out_dir, f"{family.replace(' ', '')}-{w}.ttf")
        if os.path.exists(dest):
            out[w] = dest
            continue
        css = requests.get(f"https://fonts.googleapis.com/css2?family={fam}:wght@{w}",
                           headers={"User-Agent": "Mozilla/5.0 (Windows NT 6.1) Gecko/20100101"},
                           timeout=30)
        if css.status_code != 200:
            continue
        m = re.search(r"url\((https://[^)]+\.ttf)\)", css.text)
        if not m:
            continue
        data = requests.get(m.group(1), timeout=60).content
        FsPath(dest).write_bytes(data)
        out[w] = dest
    return out


def resolve_fonts(spec, project_dir):
    """spec: {"400": path, ...} or {"family": "Inter"}; falls back to system fonts."""
    out = {}
    spec = spec or {}
    if spec.get("family"):
        try:
            out.update(fetch_google_font(spec["family"], os.path.join(project_dir, "assets", "fonts")))
        except Exception as e:  # offline: fall back quietly
            print(f"[fonts] could not fetch {spec['family']}: {e}", file=sys.stderr)
    for k, v in spec.items():
        if k.isdigit() and v:
            p = v if os.path.isabs(v) else os.path.join(project_dir, v)
            if os.path.exists(p):
                out[int(k)] = p
    for w, names in _SYSTEM_FONTS.items():
        if w in out:
            continue
        for d in _FONT_DIRS:
            hit = next((os.path.join(d, n) for n in names if os.path.exists(os.path.join(d, n))), None)
            if hit:
                out[w] = hit
                break
    if 400 not in out:
        raise RuntimeError("No usable font found. Set brief.font.family to a Google Fonts family.")
    for w in (500, 700):
        out.setdefault(w, out[400])
    return out


@lru_cache(maxsize=256)
def _font(path, size):
    return ImageFont.truetype(path, size)


class Typesetter:
    """Renders text sprites at 3x and downsamples (clean edges at any S)."""

    def __init__(self, fonts, S):
        self.fonts = fonts
        self.S = S
        self._cache = {}

    def font_path(self, weight):
        w = min(self.fonts, key=lambda k: abs(k - weight))
        return self.fonts[w]

    def text(self, s, size, weight=400, color="#1F1F1F", tracking=0.0, max_scale=1.0):
        """A sprite of ``s`` at ``size`` design px. Anchor (0, baseline) is the
        text origin: sprite.baseline is the baseline's distance from the top."""
        key = (s, size, weight, color, tracking, max_scale)
        if key in self._cache:
            return self._cache[key]
        ss = 3
        ppd = self.S * max_scale
        px = max(4, int(round(size * ppd * ss)))
        f = _font(self.font_path(weight), px)
        asc, desc = f.getmetrics()
        tr = tracking * size * ppd * ss
        widths = [f.getlength(ch) for ch in s]
        width = int(sum(widths) + tr * max(0, len(s) - 1) + px * 0.3) + 4
        hgt = asc + desc + 4
        rgb = tuple(int(c * 255) for c in hex_srgb(color))
        im = Image.new("RGBA", (width, hgt), rgb + (0,))
        d = ImageDraw.Draw(im)
        if tracking == 0:
            d.text((2, 2), s, font=f, fill=rgb + (255,))
        else:
            x = 2.0
            for ch, wch in zip(s, widths):
                d.text((x, 2), ch, font=f, fill=rgb + (255,))
                x += wch + tr
        bb = im.getchannel("A").getbbox()
        right = (bb[2] + 2) if bb else 4
        im = im.crop((0, 0, right + (ss - right % ss) % ss, hgt + (ss - hgt % ss) % ss))
        sp = Sprite.from_pil(im, ppd, supersample=ss, baseline=(asc + 2) / ss / ppd)
        # x advance of each prefix, for typing reveals (design px)
        adv = np.concatenate([[0], np.cumsum(np.array(widths) + tr)]) / ss / ppd + 2 / ss / ppd
        sp.meta["advance"] = adv
        self._cache[key] = sp
        return sp

    def width(self, s, size, weight=400, tracking=0.0):
        f = _font(self.font_path(weight), max(4, int(size * 4)))
        return (f.getlength(s) + tracking * size * 4 * max(0, len(s) - 1)) / 4
