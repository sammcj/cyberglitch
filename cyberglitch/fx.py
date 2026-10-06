"""Optional extra stages. Each is a no-op at its default settings so the base look is unchanged."""

import numpy as np

from .config import Echo, Glitch, Haze, Rain


def _rgb(colour: str) -> np.ndarray:
    c = colour.lstrip("#")
    return np.array([int(c[i : i + 2], 16) / 255 for i in (0, 2, 4)], np.float32)


def background(img: np.ndarray, matte: np.ndarray | None) -> np.ndarray:
    """0-1 background weight: the inverse matte, or darkness when there is no matte."""
    if matte is not None:
        return 1 - matte
    return np.clip(1 - (img @ np.array([0.2126, 0.7152, 0.0722], np.float32)) * 5, 0, 1)


def haze(img: np.ndarray, matte: np.ndarray | None, h: Haze) -> np.ndarray:
    """Vertical smog gradient behind the subject; gets dithered with everything else."""
    if h.strength <= 0:
        return img
    rows = img.shape[0]
    t = np.linspace(0, 1, rows, dtype=np.float32)[:, None, None] ** h.falloff
    grad = _rgb(h.top) * (1 - t) + _rgb(h.bottom) * t
    return np.clip(img + grad * background(img, matte)[..., None] * h.strength, 0, 1)


def echo(img: np.ndarray, e: Echo, state: dict) -> np.ndarray:
    """Ghost trails: each frame keeps a fading copy of the brightest recent pixels."""
    if e.decay <= 0:
        return img
    prev = state.get("echo")
    out = img if prev is None or prev.shape != img.shape else np.maximum(img, prev * e.decay)
    state["echo"] = out
    return out


def rain(img: np.ndarray, r: Rain, frame: int) -> np.ndarray:
    if r.density <= 0:
        return img
    h, w, _ = img.shape
    n = max(1, int(w * r.density))
    rng = np.random.default_rng(r.seed)  # fixed per clip so drops move smoothly between frames
    x0 = rng.random(n) * (w + h * abs(r.slant))
    y0 = rng.random(n) * (h + r.length)
    speed = r.speed * (0.6 + 0.8 * rng.random(n))
    length = np.maximum(1, (r.length * (0.5 + rng.random(n))).astype(int))
    head = (y0 + frame * speed) % (h + r.length)
    steps = np.arange(r.length * 2)
    ys = head[:, None] - steps[None, :]
    xs = x0[:, None] - h * max(r.slant, 0) + ys * r.slant
    keep = (steps[None, :] < length[:, None]) & (ys >= 0) & (ys < h) & (xs >= 0) & (xs < w)
    yi, xi = ys[keep].astype(int), xs[keep].astype(int)
    out = img.copy()
    out[yi, xi] = out[yi, xi] * (1 - r.opacity) + _rgb(r.colour) * r.opacity
    return out


def glitch(img: np.ndarray, g: Glitch, frame: int) -> np.ndarray:
    """Horizontal slice tears, re-rolled every frame."""
    if g.chance <= 0:
        return img
    rng = np.random.default_rng((g.seed, frame))
    if rng.random() >= g.chance:
        return img
    out = img.copy()
    h = img.shape[0]
    for _ in range(rng.integers(1, g.slices + 1)):
        y = rng.integers(0, h)
        band = slice(y, min(h, y + rng.integers(1, g.max_height + 1)))
        shift = int(rng.integers(-g.max_shift, g.max_shift + 1))
        out[band] = np.roll(out[band], shift, axis=1)
        if g.channel_swap and rng.random() < 0.5:
            out[band] = out[band][..., ::-1]
    return out
