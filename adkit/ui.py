"""UI kit, rebuilt by eye: clean light-UI pieces drawn with Pillow at 3x.

Every function returns a ``Sprite`` sized in design pixels. Content (names,
files, numbers, people) is fictional and comes from the brief. Shapes are drawn
without anti-aliasing at 3x, then area-downsampled in premultiplied linear
light, which gives clean edges without fringes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field as dc_field

import numpy as np
from PIL import Image, ImageDraw

from .brand import _font
from .core import hex_srgb, mix_hex, rng
from .gfx import Sprite

SS = 3  # supersample


@dataclass
class Theme:
    text: str = "#1F1F1F"
    text2: str = "#5F6368"
    card: str = "#FFFFFF"
    border: str = "#DADCE0"
    track: str = "#E8EAED"
    field: str = "#E9EEF6"
    accent_bg: str = "#C2E7FF"      # e.g. Share button
    accent_fg: str = "#062E6F"
    bezel: str = "#202124"
    hairline: str = "#5F6368"
    paper_frame: str = "#F9FBFD"
    bars: str = "#DADCE0"
    folder: tuple = ("#80868B", "#9AA0A6", "#B0B5BA")
    kinds: dict = dc_field(default_factory=lambda: {
        "doc": "#4285F4", "sheet": "#0F9D58", "slides": "#F4B400", "pdf": "#EA4335",
        "photo": "#D93025", "video": "#EA4335", "code": "#7C4DFF", "audio": "#00897B",
        "design": "#A259FF", "link": "#1A73E8", "task": "#188038", "chat": "#5F6368",
    })

    @classmethod
    def from_brand(cls, primary=None, overrides=None, dark=False):
        t = cls()
        if dark:
            t.text, t.text2, t.card, t.border = "#F1F3F4", "#9AA0A6", "#17191E", "#2C3038"
            t.track, t.field, t.bezel, t.hairline = "#2C3038", "#1D2026", "#0B0C0F", "#3C4049"
            t.paper_frame, t.bars = "#121418", "#30343C"
            t.folder = ("#4A4F58", "#5B616B", "#6E747F")
        if primary:
            if dark:
                t.accent_bg = mix_hex(primary, "#17191E", 0.55)
                t.accent_fg = mix_hex(primary, "#FFFFFF", 0.75)
                t.field = mix_hex(primary, "#1D2026", 0.9)
            else:
                t.accent_bg = mix_hex(primary, "#FFFFFF", 0.78)
                t.accent_fg = mix_hex(primary, "#000000", 0.55)
                t.field = mix_hex(primary, "#EDF0F4", 0.93)
        for k, v in (overrides or {}).items():
            if hasattr(t, k):
                setattr(t, k, v)
        return t


def _rgba(h, a=255):
    r, g, b = (int(round(c * 255)) for c in hex_srgb(h))
    return (r, g, b, a)


class Painter:
    """Draw in design px onto a 3x supersampled RGBA image, then ``.sprite()``."""

    def __init__(self, w, h, ppd, fonts):
        self.w, self.h, self.ppd, self.fonts = w, h, ppd, fonts
        self.k = ppd * SS
        self.im = Image.new("RGBA", (max(1, int(math.ceil(w * self.k))), max(1, int(math.ceil(h * self.k)))),
                            (0, 0, 0, 0))
        self.d = ImageDraw.Draw(self.im)

    def _b(self, x, y, w, h):
        k = self.k
        x0, y0 = x * k, y * k
        return [x0, y0, max(x0, (x + w) * k - 1), max(y0, (y + h) * k - 1)]

    def rrect(self, x, y, w, h, r, fill=None, outline=None, width=1.0, alpha=255):
        k = self.k
        if outline:
            self.d.rounded_rectangle(self._b(x, y, w, h), radius=r * k, fill=_rgba(outline, alpha))
            bw = width
            self.d.rounded_rectangle(self._b(x + bw, y + bw, w - 2 * bw, h - 2 * bw),
                                     radius=max(0, (r - bw)) * k, fill=_rgba(fill, alpha) if fill else (0, 0, 0, 0))
        elif fill:
            self.d.rounded_rectangle(self._b(x, y, w, h), radius=r * k, fill=_rgba(fill, alpha))

    def hole(self, x, y, w, h, r):
        self.d.rounded_rectangle(self._b(x, y, w, h), radius=r * self.k, fill=(0, 0, 0, 0))

    def circle(self, cx, cy, r, fill, alpha=255):
        k = self.k
        x0, y0 = (cx - r) * k, (cy - r) * k
        self.d.ellipse([x0, y0, max(x0, (cx + r) * k - 1), max(y0, (cy + r) * k - 1)], fill=_rgba(fill, alpha))

    def ring(self, cx, cy, r, width, color):
        k = self.k
        self.d.ellipse([(cx - r) * k, (cy - r) * k, (cx + r) * k, (cy + r) * k], outline=_rgba(color),
                       width=max(1, int(width * k)))

    def line(self, pts, color, width, alpha=255):
        k = self.k
        p = [(x * k, y * k) for x, y in pts]
        self.d.line(p, fill=_rgba(color, alpha), width=max(1, int(round(width * k))), joint="curve")
        r = width * k / 2
        for x, y in (p[0], p[-1]):
            self.d.ellipse([x - r, y - r, x + r, y + r], fill=_rgba(color, alpha))

    def poly(self, pts, fill, alpha=255):
        k = self.k
        self.d.polygon([(x * k, y * k) for x, y in pts], fill=_rgba(fill, alpha))

    def text(self, x, y, s, size, weight=400, color="#1F1F1F", anchor="ls"):
        """anchor uses Pillow codes: 'ls' left-baseline, 'ms' middle-baseline, 'lm'..."""
        path = self.fonts[min(self.fonts, key=lambda w: abs(w - weight))]
        f = _font(path, max(4, int(round(size * self.k))))
        self.d.text((x * self.k, y * self.k), s, font=f, fill=_rgba(color), anchor=anchor)

    def sprite(self, **kw):
        return Sprite.from_pil(self.im, self.ppd, supersample=SS, **kw)


# ---------------------------------------------------------------------------
# icons
# ---------------------------------------------------------------------------

def draw_kind_icon(p: Painter, kind, x, y, s, theme: Theme):
    """A small file-type tile at (x, y) of size s."""
    c = theme.kinds.get(kind, theme.text2)
    p.rrect(x, y, s, s, s * 0.22, fill=c)
    w = "#FFFFFF"
    if kind in ("doc", "pdf", "chat"):
        for i in range(3):
            p.rrect(x + s * 0.24, y + s * (0.3 + i * 0.16), s * (0.52 if i < 2 else 0.34), s * 0.08, s * 0.04, fill=w)
    elif kind == "sheet":
        p.rrect(x + s * 0.22, y + s * 0.24, s * 0.56, s * 0.52, s * 0.04, fill=w)
        for i in range(1, 3):
            p.rrect(x + s * 0.22, y + s * (0.24 + i * 0.17), s * 0.56, s * 0.04, 0, fill=c)
        p.rrect(x + s * 0.48, y + s * 0.24, s * 0.04, s * 0.52, 0, fill=c)
    elif kind in ("slides", "design"):
        p.rrect(x + s * 0.2, y + s * 0.3, s * 0.6, s * 0.4, s * 0.05, fill=w)
        p.rrect(x + s * 0.27, y + s * 0.37, s * 0.46, s * 0.26, s * 0.03, fill=c)
    elif kind in ("photo",):
        p.circle(x + s * 0.66, y + s * 0.34, s * 0.09, w)
        p.poly([(x + s * 0.18, y + s * 0.76), (x + s * 0.42, y + s * 0.42), (x + s * 0.62, y + s * 0.76)], w)
    elif kind == "video":
        p.poly([(x + s * 0.36, y + s * 0.28), (x + s * 0.74, y + s * 0.5), (x + s * 0.36, y + s * 0.72)], w)
    elif kind == "code":
        p.line([(x + s * 0.4, y + s * 0.3), (x + s * 0.22, y + s * 0.5), (x + s * 0.4, y + s * 0.7)], w, s * 0.08)
        p.line([(x + s * 0.6, y + s * 0.3), (x + s * 0.78, y + s * 0.5), (x + s * 0.6, y + s * 0.7)], w, s * 0.08)
    else:
        p.circle(x + s / 2, y + s / 2, s * 0.16, w)


def draw_check(p: Painter, cx, cy, r, color, fg="#FFFFFF"):
    p.circle(cx, cy, r, color)
    p.line([(cx - r * 0.42, cy + r * 0.02), (cx - r * 0.1, cy + r * 0.34), (cx + r * 0.46, cy - r * 0.3)], fg, r * 0.24)


def draw_magnifier(p: Painter, cx, cy, s, color):
    p.ring(cx - s * 0.1, cy - s * 0.1, s * 0.32, s * 0.1, color)
    p.line([(cx + s * 0.14, cy + s * 0.14), (cx + s * 0.4, cy + s * 0.4)], color, s * 0.11)


def photo_thumb(p: Painter, x, y, w, h, seed, video=False):
    """A generated gradient landscape (never a real photo)."""
    r = rng(seed)
    skies = [("#8EC5FC", "#E0C3FC"), ("#FAD0C4", "#FFD1FF"), ("#A1C4FD", "#C2E9FB"),
             ("#FBC2EB", "#A6C1EE"), ("#FDCBF1", "#E6DEE9"), ("#84FAB0", "#8FD3F4")]
    top, bot = skies[int(r.integers(len(skies)))]
    steps = 24
    for i in range(steps):
        p.rrect(x, y + h * i / steps, w, h / steps + 0.6, 0, fill=mix_hex(top, bot, i / (steps - 1)))
    p.circle(x + w * r.uniform(0.6, 0.8), y + h * r.uniform(0.25, 0.4), h * 0.11, "#FFF6E0")
    for j, col in enumerate(("#6C8EAD", "#3E5C76")):
        base = y + h * (0.72 + 0.1 * j)
        pts = [(x, y + h)]
        n = 5
        for k in range(n + 1):
            pts.append((x + w * k / n, base - h * r.uniform(0.08, 0.3) * (1 - 0.3 * j)))
        pts.append((x + w, y + h))
        p.poly(pts, col)
    if video:
        p.circle(x + w / 2, y + h / 2, h * 0.17, "#000000", alpha=110)
        s = h * 0.12
        p.poly([(x + w / 2 - s * 0.45, y + h / 2 - s * 0.7), (x + w / 2 + s * 0.75, y + h / 2),
                (x + w / 2 - s * 0.45, y + h / 2 + s * 0.7)], "#FFFFFF")


# ---------------------------------------------------------------------------
# components
# ---------------------------------------------------------------------------

def card(w, h, r, ppd, fonts, theme: Theme, fill=None, border=True):
    p = Painter(w, h, ppd, fonts)
    p.rrect(0, 0, w, h, r, fill=fill or theme.card, outline=theme.border if border else None, width=1.2)
    return p.sprite()


def pill(w, h, ppd, fonts, fill, outline=None):
    p = Painter(w, h, ppd, fonts)
    p.rrect(0, 0, w, h, h / 2, fill=fill, outline=outline, width=1.2)
    return p.sprite()


def folder(w, h, ppd, fonts, theme: Theme):
    """Material-style folder: back + tab, then the front panel."""
    back, mid, front = theme.folder
    p = Painter(w, h, ppd, fonts)
    p.rrect(0, h * 0.04, w * 0.42, h * 0.2, h * 0.06, fill=back)
    p.rrect(0, h * 0.12, w, h * 0.86, h * 0.07, fill=back)
    p.rrect(w * 0.02, h * 0.2, w * 0.96, h * 0.3, h * 0.04, fill="#FFFFFF")
    p.rrect(0, h * 0.3, w, h * 0.7, h * 0.07, fill=mid)
    p.rrect(0, h * 0.36, w, h * 0.64, h * 0.07, fill=front)
    return p.sprite()


def file_card(kind, seed, ppd, fonts, theme: Theme, w=150, h=186):
    p = Painter(w, h, ppd, fonts)
    p.rrect(0, 0, w, h, 14, fill=theme.card, outline=theme.border, width=1.2)
    th = h * 0.62
    if kind in ("photo", "video"):
        sub = Painter(w - 2.4, th - 1.2, ppd, fonts)
        photo_thumb(sub, 0, 0, w - 2.4, th - 1.2, seed, video=(kind == "video"))
        mask = Image.new("L", sub.im.size, 0)
        ImageDraw.Draw(mask).rounded_rectangle([0, 0, sub.im.size[0] - 1, sub.im.size[1] + 40], radius=13 * p.k, fill=255)
        p.im.paste(sub.im, (int(1.2 * p.k), int(1.2 * p.k)), mask)
    else:
        c = theme.kinds.get(kind, theme.text2)
        p.rrect(1.2, 1.2, w - 2.4, th, 13, fill=mix_hex(c, "#FFFFFF", 0.9))
        p.rrect(1.2, th - 6, w - 2.4, 7, 0, fill=mix_hex(c, "#FFFFFF", 0.9))
        pw, ph = w * 0.62, th * 0.82
        px0, py0 = (w - pw) / 2, th - ph
        p.rrect(px0, py0, pw, ph + 2, 6, fill="#FFFFFF")
        r = rng(seed)
        if kind == "sheet":
            for i in range(5):
                for j in range(3):
                    p.rrect(px0 + 8 + j * (pw - 16) / 3, py0 + 10 + i * 13, (pw - 16) / 3 - 3, 9, 1,
                            fill=mix_hex(c, "#FFFFFF", 0.55 if i == 0 else 0.8))
        elif kind in ("slides", "design"):
            p.rrect(px0 + 8, py0 + 10, pw - 16, ph * 0.42, 3, fill=mix_hex(c, "#FFFFFF", 0.35))
            p.rrect(px0 + 8, py0 + 16 + ph * 0.42, (pw - 16) * 0.7, 7, 3, fill=theme.bars)
        else:
            for i in range(6):
                ln = r.uniform(0.5, 1.0) if i else 0.55
                p.rrect(px0 + 9, py0 + 11 + i * 12, (pw - 18) * ln, 6, 3,
                        fill=c if (i == 0 and kind == "pdf") else theme.bars)
    draw_kind_icon(p, kind, 12, th + 12, 22, theme)
    p.rrect(42, th + 15, w * 0.48, 8, 4, fill="#BDC1C6")
    p.rrect(12, th + 46, w * 0.62, 7, 3.5, fill=theme.track)
    return p.sprite()


def result_row(kind, title, meta, ppd, fonts, theme: Theme, w=1000, h=78, selected=False):
    p = Painter(w, h, ppd, fonts)
    p.rrect(0, 0, w, h, 16, fill=theme.card, outline=theme.border, width=1.2)
    draw_kind_icon(p, kind, 24, (h - 34) / 2, 34, theme)
    p.text(80, h / 2 + 10, title, 28, 500, theme.text)
    if meta:
        p.text(w - 28, h / 2 + 8, meta, 22, 400, theme.text2, anchor="rs")
    return p.sprite()


def doc_page(title, ppd, fonts, theme: Theme, share_label="Share", w=1240, h=720, avatars=(),
             header_kind="doc", file_name=None, bars=9, content="doc", seed=3):
    """A document/app page: frame, header (icon, name, avatars, Share pill), paper.

    content: 'doc' text bars, 'canvas' a design frame, 'board' task columns.
    Returns (sprite, layout dict in design px relative to the page's top-left).
    """
    p = Painter(w, h, ppd, fonts)
    p.rrect(0, 0, w, h, 22, fill=theme.paper_frame, outline=theme.border, width=1.2)
    draw_kind_icon(p, header_kind, 28, 24, 34, theme)
    p.text(76, 50, file_name or title, 22, 500, theme.text)
    p.rrect(76, 60, 180, 7, 3.5, fill=theme.track)
    sw, sh = 130, 48
    sx, sy = w - sw - 28, 18
    p.rrect(sx, sy, sw, sh, sh / 2, fill=theme.accent_bg)
    p.text(sx + sw / 2, sy + sh / 2 + 8, share_label, 22, 500, theme.accent_fg, anchor="ms")
    lay = dict(share=(sx, sy, sw, sh), avatars=[])
    for i, col in enumerate(avatars):
        cx = sx - 30 - (len(avatars) - 1 - i) * 30
        lay["avatars"].append((cx, sy + sh / 2, 15, col))
    inset = 150
    px0, py0, pw = inset, 96, w - 2 * inset
    p.rrect(px0, py0, pw, h - py0 + 30, 10, fill="#FFFFFF", outline="#E3E6EA", width=1.0)
    lay["paper"] = (px0, py0, pw, h - py0)
    lay["bars"] = []
    if content == "doc":
        p.text(px0 + 70, py0 + 92, title, 44, 500, theme.text)
        r = rng(seed)
        y = py0 + 140
        for i in range(bars):
            ln = (pw - 140) * (r.uniform(0.55, 0.98) if i % 4 != 3 else r.uniform(0.3, 0.5))
            lay["bars"].append((px0 + 70, y, ln, 10))
            y += 34 if i % 4 != 3 else 52
        for i, b in enumerate(lay["bars"]):
            p.rrect(*b, 5, fill=theme.bars)
    elif content == "canvas":
        p.text(px0 + 40, py0 + 50, title, 26, 500, theme.text2)
        fx, fy, fw, fh = px0 + 40, py0 + 72, pw - 80, h - py0 - 110
        p.rrect(fx, fy, fw, fh, 8, fill="#FFFFFF", outline="#D0D4DA", width=1.0)
        p.rrect(fx + 24, fy + 22, fw - 48, 34, 6, fill="#F1F3F4")
        p.rrect(fx + 24, fy + 74, fw * 0.48, fh - 150, 10, fill=mix_hex(theme.accent_fg, "#FFFFFF", 0.82))
        for i in range(3):
            lay["bars"].append((fx + fw * 0.56, fy + 84 + i * 46, fw * 0.36, 14))
        for b in lay["bars"]:
            p.rrect(*b, 7, fill=theme.bars)
        p.rrect(fx + fw * 0.56, fy + fh - 96, 150, 44, 22, fill=theme.accent_fg)
    else:  # board
        cols = 3
        cw = (pw - 80 - (cols - 1) * 24) / cols
        for c in range(cols):
            cx = px0 + 40 + c * (cw + 24)
            p.rrect(cx, py0 + 40, cw, h - py0 - 70, 14, fill="#F1F3F4")
            for k in range(3 - (c == 2)):
                b = (cx + 14, py0 + 70 + k * 92, cw - 28, 76)
                p.rrect(*b, 10, fill="#FFFFFF", outline="#E3E6EA", width=1)
                lay["bars"].append((b[0] + 16, b[1] + 22, b[2] * 0.6, 10))
        for b in lay["bars"]:
            p.rrect(*b, 5, fill=theme.bars)
    return p.sprite(), lay


def avatar(color, letter, ppd, fonts, r=15):
    p = Painter(2 * r + 4, 2 * r + 4, ppd, fonts)
    p.circle(r + 2, r + 2, r + 2, "#FFFFFF")
    p.circle(r + 2, r + 2, r, color)
    if letter:
        p.text(r + 2, r + 2 + r * 0.36, letter, r * 1.0, 700, "#FFFFFF", anchor="ms")
    return p.sprite()


def cursor(color, ppd, fonts, s=30):
    p = Painter(s, s * 1.3, ppd, fonts)
    pts = [(2, 2), (2, s * 1.05), (s * 0.3, s * 0.8), (s * 0.5, s * 1.22), (s * 0.66, s * 1.15),
           (s * 0.47, s * 0.74), (s * 0.85, s * 0.74)]
    # white outline then colour fill
    p.poly([(x - 0.0, y) for x, y in pts], "#FFFFFF")
    inner = [(2 + 2.2, 2 + 4.5), (2 + 2.2, s * 1.05 - 5), (s * 0.31, s * 0.75), (s * 0.51, s * 1.15),
             (s * 0.61, s * 1.11), (s * 0.43, s * 0.7), (s * 0.78, s * 0.7)]
    p.poly(inner, color)
    return p.sprite()


def name_flag(name, color, ppd, fonts, size=20):
    from .brand import _font as F
    f = F(fonts[500], 100)
    tw = f.getlength(name) / 100 * size
    w, h = tw + 22, size + 14
    p = Painter(w, h, ppd, fonts)
    p.rrect(0, 0, w, h, 7, fill=color)
    p.text(11, h / 2 + size * 0.36, name, size, 500, "#FFFFFF")
    return p.sprite()


def chip(label, ppd, fonts, theme: Theme, icon="check", color="#0EBC5F", w=None, h=46, size=22):
    f = _font(fonts[500], 100)
    tw = f.getlength(label) / 100 * size
    w = w or tw + h + 30
    p = Painter(w, h, ppd, fonts)
    p.rrect(0, 0, w, h, h / 2, fill=theme.card, outline=theme.border, width=1.2)
    if icon == "check":
        draw_check(p, h / 2 + 2, h / 2, h * 0.27, color)
    elif icon == "dot":
        p.circle(h / 2 + 2, h / 2, h * 0.14, color)
    p.text(h - 2 + 6, h / 2 + size * 0.36, label, size, 500, theme.text)
    return p.sprite()


def laptop(screen_w, screen_h, ppd, fonts, theme: Theme):
    """Laptop lid with a real hole for the screen, plus the base. Returns
    (lid_sprite, base_sprite, screen_rect relative to lid top-left)."""
    b = 16
    w, h = screen_w + 2 * b, screen_h + 2 * b
    p = Painter(w, h, ppd, fonts)
    p.rrect(0, 0, w, h, 22, fill=theme.hairline)
    p.rrect(1.2, 1.2, w - 2.4, h - 2.4, 21, fill=theme.bezel)
    p.hole(b, b, screen_w, screen_h, 6)
    lid = p.sprite()
    bw, bh = w * 1.14, 22
    q = Painter(bw, bh, ppd, fonts)
    q.rrect(0, 0, bw, bh * 0.62, 6, fill="#E3E5E8")
    q.rrect(bw * 0.06, bh * 0.5, bw * 0.88, bh * 0.5, 8, fill="#C9CCD1")
    q.rrect(bw / 2 - 60, 0, 120, 6, 3, fill="#C9CCD1")
    return lid, q.sprite(), (b, b, screen_w, screen_h)


def handheld(screen_w, screen_h, ppd, fonts, theme: Theme, phone=False):
    b = 12 if phone else 16
    w, h = screen_w + 2 * b, screen_h + 2 * b
    r = 40 if phone else 30
    p = Painter(w, h, ppd, fonts)
    p.rrect(0, 0, w, h, r, fill=theme.hairline)
    p.rrect(1.2, 1.2, w - 2.4, h - 2.4, r - 1, fill=theme.bezel)
    p.hole(b, b, screen_w, screen_h, r - b)
    if phone:
        p.rrect(w / 2 - 30, b + 10, 60, 18, 9, fill=theme.bezel)
    return p.sprite(), (b, b, screen_w, screen_h)


def search_pill(placeholder, ppd, fonts, theme: Theme, w=1000, h=92):
    p = Painter(w, h, ppd, fonts)
    p.rrect(0, 0, w, h, h / 2, fill=theme.field)
    draw_magnifier(p, 52, h / 2, 34, theme.text2)
    return p.sprite()


def bar_track(w, ppd, fonts, theme: Theme, h=5):
    return pill(w, h, ppd, fonts, theme.track)


def node_card(label, kind, ppd, fonts, theme: Theme, w=260, h=96):
    p = Painter(w, h, ppd, fonts)
    p.rrect(0, 0, w, h, 20, fill=theme.card, outline=theme.border, width=1.2)
    draw_kind_icon(p, kind, 22, (h - 44) / 2, 44, theme)
    p.text(82, h / 2 + 9, label, 26, 500, theme.text)
    return p.sprite()


def chart_card(w, h, ppd, fonts, theme: Theme, title="", grid=4):
    p = Painter(w, h, ppd, fonts)
    p.rrect(0, 0, w, h, 24, fill=theme.card, outline=theme.border, width=1.2)
    if title:
        p.text(36, 56, title, 24, 500, theme.text2)
    x0, y0, x1, y1 = 36, 120, w - 36, h - 50
    for i in range(grid + 1):
        y = y0 + (y1 - y0) * i / grid
        p.rrect(x0, y, x1 - x0, 1.6, 0, fill=theme.track)
    return p.sprite(), (x0, y0, x1, y1)


def browser_frame(shot: Image.Image, ppd, fonts, theme: Theme, w=1240, url=""):
    """A real product screenshot inside a clean browser window."""
    bar = 52
    sh = shot.height * (w / shot.width)
    h = sh + bar
    p = Painter(w, h, ppd, fonts)
    p.rrect(0, 0, w, h, 18, fill="#F1F3F4", outline=theme.border, width=1.2)
    for i, c in enumerate(("#FF5F57", "#FEBC2E", "#28C840")):
        p.circle(28 + i * 22, bar / 2, 6.5, c)
    if url:
        p.rrect(110, 12, w - 220, bar - 24, (bar - 24) / 2, fill="#FFFFFF")
        p.text(w / 2, bar / 2 + 7, url, 18, 400, theme.text2, anchor="ms")
    k = p.k
    tgt = shot.convert("RGBA").resize((int(w * k - 2.4 * k), int(sh * k)), Image.LANCZOS)
    mask = Image.new("L", tgt.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, -40, tgt.size[0] - 1, tgt.size[1] - 1], radius=17 * k, fill=255)
    p.im.paste(tgt, (int(1.2 * k), int(bar * k)), mask)
    return p.sprite(), dict(content=(1.2, bar, w - 2.4, sh), scale=w / shot.width)


def chat_bubble(w, h, ppd, fonts, theme: Theme, user=False, fill=None):
    p = Painter(w, h, ppd, fonts)
    p.rrect(0, 0, w, h, min(28, h / 2), fill=fill or (theme.field if user else theme.card),
            outline=None if user else theme.border, width=1.2)
    return p.sprite()


def bar(w, h, color, ppd, fonts):
    p = Painter(w, h, ppd, fonts)
    p.rrect(0, 0, w, h, h / 2, fill=color)
    return p.sprite()
