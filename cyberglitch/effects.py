"""Per-frame effect stages. Images are float32 RGB arrays in 0-1, shape (H, W, 3)."""

import math
import threading

import numpy as np
from numba import njit, prange
from scipy.ndimage import gaussian_filter

from . import fx
from .config import Config, Dither, Drip, Glow, Post, Tone

LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
# numba's default (workqueue) threading layer aborts the process if parallel kernels run from two
# threads at once, e.g. a web preview during a render job. Each frame already uses every core.
_kernels = threading.Lock()


def hex_to_rgb(colours: list[str]) -> np.ndarray:
    return np.array(
        [[int(c.lstrip("#")[i : i + 2], 16) / 255 for i in (0, 2, 4)] for c in colours], dtype=np.float32
    )


def luma(img: np.ndarray) -> np.ndarray:
    return img @ LUMA


def luma_matte(img: np.ndarray, low: float, high: float) -> np.ndarray:
    t = np.clip((luma(img) - low) / max(high - low, 1e-6), 0, 1)
    return t * t * (3 - 2 * t)


def tone(img: np.ndarray, t: Tone) -> np.ndarray:
    out = (img - 0.5) * t.contrast + 0.5 + t.brightness
    if t.saturation != 1.0:
        grey = luma(out)[..., None]
        out = grey + (out - grey) * t.saturation
    out = np.clip(out, 0, 1)
    if t.gamma != 1.0:
        out = out ** (1 / t.gamma)
    if t.colourise > 0:
        ramp = hex_to_rgb(t.ramp)
        pos = luma(out) * (len(ramp) - 1)
        i = np.clip(pos.astype(np.int32), 0, len(ramp) - 2)
        f = (pos - i)[..., None]
        mapped = ramp[i] * (1 - f) + ramp[i + 1] * f
        out = out * (1 - t.colourise) + mapped * t.colourise
    return out.astype(np.float32)


@njit(cache=True, parallel=True)
def _drip(img, lum, threshold, decay, limit, mix):
    h, w, _ = img.shape
    out = img.copy()
    for x in prange(w):  # columns are independent
        c0 = c1 = c2 = 0.0
        age = np.inf  # pixels since the column was last fed
        for y in range(h):
            feed = lum[y, x] >= threshold
            if feed:
                c0, c1, c2 = img[y, x, 0], img[y, x, 1], img[y, x, 2]
                age = 0.0
            else:
                age += 1
            c0 *= decay
            c1 *= decay
            c2 *= decay
            if not feed and age <= limit[x]:
                out[y, x, 0] = max(img[y, x, 0], c0 * mix)
                out[y, x, 1] = max(img[y, x, 1], c1 * mix)
                out[y, x, 2] = max(img[y, x, 2], c2 * mix)
    return out


def drip(
    img: np.ndarray, d: Drip, rng: np.random.Generator | None = None, subject: np.ndarray | None = None
) -> np.ndarray:
    """Smear the colour of each column's last bright pixel down over the darker pixels below it.

    `subject` (bool, H x W) limits where streaks can start, so a lit background doesn't drip.
    """
    h, w, _ = img.shape
    lum = luma(img)
    if subject is not None:
        lum = np.where(subject, lum, 0)
    limit = np.full(w, d.length if d.length > 0 else np.inf, np.float64)
    if d.jitter > 0:
        rng = rng or np.random.default_rng(0)
        limit = limit if np.isfinite(limit).all() else np.full(w, float(h))
        limit *= 1 - d.jitter * rng.random(w)
    return _drip(np.ascontiguousarray(img, np.float32), lum, d.threshold, d.decay, limit, d.mix)


@njit(cache=True)
def _dither_row(
    img, y, palette, below, next_below, out, diffuse_x, diffuse_y, amp, fx, fy, fl, phase, run_bias, floor
):
    w = img.shape[1]
    n = palette.shape[0]
    two_pi = 2 * math.pi
    r_err = g_err = b_err = 0.0
    prev = -1
    for x in range(w):
        lum = 0.2126 * img[y, x, 0] + 0.7152 * img[y, x, 1] + 0.0722 * img[y, x, 2]
        m = amp * math.sin(two_pi * (x * fx + y * fy + lum * fl) + phase)
        # Fade out on the background (at or below the darkest palette colour) so it stays clean;
        # otherwise specks there get smeared into full-height lines by the drip
        m *= min(1.0, max(0.0, (lum - floor) * 10))
        r = img[y, x, 0] + r_err + below[x, 0] + m
        g = img[y, x, 1] + g_err + below[x, 1] + m
        b = img[y, x, 2] + b_err + below[x, 2] + m
        best = 0
        best_d = 1e9
        for k in range(n):
            dr = r - palette[k, 0]
            dg = g - palette[k, 1]
            db = b - palette[k, 2]
            dist = 0.3 * dr * dr + 0.59 * dg * dg + 0.11 * db * db
            if k == prev:
                dist -= run_bias  # hysteresis: holding the previous colour makes horizontal runs
            if dist < best_d:
                best_d = dist
                best = k
        out[y, x] = best
        prev = best
        # The modulation is excluded from the error so it shapes the pattern without shifting tone.
        # Clamping stops error piling up across large regions darker or brighter than the palette.
        er = min(max(r - m, -0.25), 1.25) - palette[best, 0]
        eg = min(max(g - m, -0.25), 1.25) - palette[best, 1]
        eb = min(max(b - m, -0.25), 1.25) - palette[best, 2]
        r_err, g_err, b_err = er * diffuse_x, eg * diffuse_x, eb * diffuse_x
        next_below[x, 0] = er * diffuse_y
        next_below[x, 1] = eg * diffuse_y
        next_below[x, 2] = eb * diffuse_y


@njit(cache=True)
def _dither_serial(img, palette, *args):
    h, w, _ = img.shape
    out = np.empty((h, w), np.uint8)
    below = np.zeros((w, 3), np.float32)
    next_below = np.zeros((w, 3), np.float32)
    for y in range(h):
        _dither_row(img, y, palette, below, next_below, out, *args)
        below, next_below = next_below, below
    return out


@njit(cache=True, parallel=True)
def _dither_rows(img, palette, *args):
    # Without downward diffusion every row is independent, so rows run in parallel
    h, w, _ = img.shape
    out = np.empty((h, w), np.uint8)
    for y in prange(h):
        zeros = np.zeros((w, 3), np.float32)
        _dither_row(img, y, palette, zeros, np.empty((w, 3), np.float32), out, *args)
    return out


def dither(img: np.ndarray, d: Dither, frame: int = 0) -> np.ndarray:
    palette = hex_to_rgb(d.palette)
    kernel = _dither_rows if d.diffuse_y == 0 else _dither_serial
    idx = kernel(
        np.ascontiguousarray(img, dtype=np.float32),
        palette,
        d.diffuse_x,
        d.diffuse_y,
        d.mod_amp,
        1 / d.mod_period_x if d.mod_period_x else 0.0,
        1 / d.mod_period_y if d.mod_period_y else 0.0,
        d.mod_luma,
        d.mod_speed * frame,
        d.run_bias,
        float((palette @ LUMA).min()),
    )
    return palette[idx]


def upscale(img: np.ndarray, factor: int) -> np.ndarray:
    return img if factor == 1 else img.repeat(factor, axis=0).repeat(factor, axis=1)


def glow_halo(img: np.ndarray, g: Glow, pixel: int) -> np.ndarray | None:
    """Blurred bright-pass at work resolution; `compose` upsamples it smoothly."""
    if g.intensity <= 0:
        return None
    bright = img * np.clip((luma(img) - g.threshold) / max(1 - g.threshold, 1e-6), 0, 1)[..., None]
    return gaussian_filter(bright, sigma=(g.radius / pixel, g.radius / pixel, 0), mode="constant")


@njit(inline="always")
def _hash01(x, y, s):
    h = (x * 374761393 + y * 668265263 + s * 2246822519) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    h ^= h >> 16
    return (h + 0.5) / 4294967296.0


@njit(cache=True, parallel=True)
def _compose(base, halo, p, gain, chroma, scan, period, vignette, flicker, grain, seed):
    """Upscale + glow screen-blend + post effects + 8-bit conversion, fused into one pass."""
    hw, ww, _ = base.shape
    H, W = hw * p, ww * p
    out = np.empty((H, W, 3), np.uint8)
    has_halo = halo.shape[0] > 0
    for y in prange(H):
        # Bilinear halo sample positions, relative to dither-pixel centres
        fy = min(max((y + 0.5) / p - 0.5, 0.0), hw - 1.0)
        y0 = int(fy)
        y1 = min(y0 + 1, hw - 1)
        ty = fy - y0
        row = flicker
        if scan > 0 and (y % period) >= period / 2:
            row *= 1 - scan
        vy = 2 * y / max(H - 1, 1) - 1
        for x in range(W):
            mul = row
            if vignette > 0:
                vx = 2 * x / max(W - 1, 1) - 1
                mul *= 1 - vignette * (vx * vx + vy * vy) / 2
            noise = 0.0
            if grain > 0:
                u1 = _hash01(x, y, seed)
                u2 = _hash01(x, y, seed + 7919)
                noise = grain * math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
            for c in range(3):
                xs = x - chroma if c == 0 else x + chroma if c == 2 else x
                xs = min(max(xs, 0), W - 1)
                v = base[y // p, xs // p, c]
                if has_halo:
                    fx = min(max((xs + 0.5) / p - 0.5, 0.0), ww - 1.0)
                    x0 = int(fx)
                    x1 = min(x0 + 1, ww - 1)
                    tx = fx - x0
                    top = halo[y0, x0, c] * (1 - tx) + halo[y0, x1, c] * tx
                    bot = halo[y1, x0, c] * (1 - tx) + halo[y1, x1, c] * tx
                    hv = min(max((top * (1 - ty) + bot * ty) * gain, 0.0), 1.0)
                    v = 1 - (1 - v) * (1 - hv)  # screen blend
                v = v * mul + noise
                out[y, x, c] = np.uint8(min(max(v, 0.0), 1.0) * 255 + 0.5)
    return out


def compose(
    img: np.ndarray, halo: np.ndarray | None, pixel: int, gain: float, post: Post, frame: int = 0
) -> np.ndarray:
    """Work-resolution float RGB to output-resolution uint8, with glow and post effects."""
    flicker = 1.0
    if post.flicker > 0:
        flicker = 1 - post.flicker * np.random.default_rng(frame).random()
    return _compose(
        np.ascontiguousarray(img, np.float32),
        np.zeros((0, 0, 3), np.float32) if halo is None else np.ascontiguousarray(halo, np.float32),
        pixel,
        gain,
        post.chroma,
        post.scanlines,
        post.scanline_period,
        post.vignette,
        flicker,
        post.grain,
        frame,
    )


def render(
    work: np.ndarray,
    cfg: Config,
    frame: int = 0,
    matte: np.ndarray | None = None,
    state: dict | None = None,
) -> np.ndarray:
    """Work-resolution float RGB in, output-resolution uint8 RGB out.

    `state` carries data between frames of one clip (echo trails); pass the same dict every frame.
    """
    with _kernels:
        return _render(work, cfg, frame, matte, state)


def _render(
    work: np.ndarray, cfg: Config, frame: int, matte: np.ndarray | None, state: dict | None
) -> np.ndarray:
    img = work if matte is None else work * matte[..., None]
    img = tone(img, cfg.tone)
    subject = None
    if cfg.haze.strength > 0:
        subject = fx.background(img, matte) < 0.5
        img = fx.haze(img, matte, cfg.haze)
    if cfg.drip.enabled and cfg.drip.stage == "before":
        img = drip(img, cfg.drip, subject=subject)
    img = fx.echo(img, cfg.echo, state if state is not None else {})
    img = dither(img, cfg.dither, frame)
    if cfg.drip.enabled and cfg.drip.stage == "after":
        img = drip(img, cfg.drip, subject=subject)
    img = fx.rain(img, cfg.rain, frame)
    img = fx.glitch(img, cfg.glitch, frame)
    p = cfg.output.pixel
    return compose(img, glow_halo(img, cfg.glow, p), p, cfg.glow.intensity, cfg.post, frame)
