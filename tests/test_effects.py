import subprocess
import sys

import numpy as np
import pytest

from cyberglitch import config, effects


@pytest.fixture
def cfg():
    return config.Config()


def gradient(h=24, w=32):
    x = np.linspace(0, 1, w, dtype=np.float32)
    img = np.repeat(x[None, :, None], h, axis=0).repeat(3, axis=2)
    return np.ascontiguousarray(img)


def test_dither_output_uses_only_palette_colours(cfg):
    out = effects.dither(gradient(), cfg.dither)
    palette = {tuple(c) for c in effects.hex_to_rgb(cfg.dither.palette)}
    assert {tuple(c) for c in out.reshape(-1, 3)} <= palette


def test_dither_preserves_average_tone(cfg):
    cfg.dither.palette = ["#000000", "#ffffff"]
    cfg.dither.mod_amp = 0.0
    cfg.dither.run_bias = 0.0
    img = np.full((8, 200, 3), 0.3, np.float32)
    assert effects.dither(img, cfg.dither).mean() == pytest.approx(0.3, abs=0.02)


def test_dither_is_deterministic(cfg):
    a = effects.dither(gradient(), cfg.dither, frame=5)
    b = effects.dither(gradient(), cfg.dither, frame=5)
    assert np.array_equal(a, b)


def test_black_background_stays_darkest_palette_colour(cfg):
    # Modulation used to push row starts on black to the next colour up, which the drip then smeared down
    darkest = effects.hex_to_rgb(cfg.dither.palette)[0]
    for bg in (np.zeros(3, np.float32), darkest):  # raw black, and black after colourise
        out = effects.dither(np.tile(bg, (60, 40, 1)), cfg.dither)
        assert np.all(out == darkest)


def test_run_bias_lengthens_horizontal_runs(cfg):
    cfg.dither.mod_amp = 0.0
    img = np.full((4, 300, 3), 0.45, np.float32)

    def changes(bias):
        cfg.dither.run_bias = bias
        out = effects.dither(img, cfg.dither)
        return np.any(out[:, 1:] != out[:, :-1], axis=2).sum()

    assert changes(0.1) < changes(0.0)


def test_drip_fills_below_bright_pixel_only(cfg):
    img = np.zeros((10, 3, 3), np.float32)
    img[3, 1] = [1.0, 0.2, 0.4]
    out = effects.drip(img, cfg.drip)
    assert np.all(out[:3] == 0), "nothing above the source pixel"
    assert np.allclose(out[4:, 1], [1.0, 0.2, 0.4]), "streak reaches the bottom"
    assert np.all(out[:, [0, 2]] == 0), "neighbouring columns untouched"


def test_drip_length_limits_streak(cfg):
    cfg.drip.length = 2
    img = np.zeros((10, 1, 3), np.float32)
    img[0, 0] = 1.0
    out = effects.drip(img, cfg.drip)
    assert out[2, 0, 0] == 1.0 and out[3, 0, 0] == 0.0


def test_drip_jitter_shortens_fixed_length_streaks(cfg):
    cfg.drip.length, cfg.drip.jitter = 8, 1.0
    img = np.zeros((10, 50, 3), np.float32)
    img[0] = 1.0
    out = effects.drip(img, cfg.drip, np.random.default_rng(0))
    assert (out[1:, :, 0] > 0).sum(axis=0).max() <= 8 and out[8, :, 0].min() == 0


def test_drip_new_bright_pixel_replaces_carry(cfg):
    img = np.zeros((6, 1, 3), np.float32)
    img[0, 0] = [1, 0, 0]
    img[3, 0] = [0, 1, 0]
    out = effects.drip(img, cfg.drip)
    assert np.allclose(out[2, 0], [1, 0, 0]) and np.allclose(out[5, 0], [0, 1, 0])


def test_luma_matte_keys_out_dark_background():
    img = np.zeros((1, 3, 3), np.float32)
    img[0, 1] = 0.08
    img[0, 2] = 1.0
    m = effects.luma_matte(img, 0.05, 0.15)
    assert m[0, 0] == 0 and 0 < m[0, 1] < 1 and m[0, 2] == 1


def test_colourise_maps_luma_onto_ramp(cfg):
    cfg.tone = config.Tone(colourise=1.0, ramp=["#000000", "#ff0000"])
    out = effects.tone(np.ones((1, 1, 3), np.float32), cfg.tone)
    assert np.allclose(out[0, 0], [1, 0, 0])


def glow(img, g, pixel):
    return effects.compose(img, effects.glow_halo(img, g, pixel), pixel, g.intensity, config.Post()) / 255


def test_glow_brightens_around_highlights(cfg):
    img = np.zeros((61, 61, 3), np.float32)
    img[28:33, 28:33] = 1.0  # a single pixel's glow rounds away in 8-bit output
    out = glow(img, cfg.glow, 1)
    assert out[30, 38].sum() > 0 and out[0, 0].sum() == pytest.approx(0, abs=1e-3)


def test_glow_on_dither_grid_matches_full_resolution_glow(cfg):
    img = np.zeros((30, 30, 3), np.float32)
    img[14:16, 14:16] = 1.0
    fast = glow(img, cfg.glow, 3)
    full = glow(effects.upscale(img, 3), cfg.glow, 1)
    assert fast.shape == full.shape == (90, 90, 3)
    assert np.abs(fast - full).max() < 0.1


def test_compose_without_glow_or_post_is_plain_upscale(cfg):
    img = np.random.default_rng(1).random((5, 7, 3)).astype(np.float32)
    out = effects.compose(img, None, 3, 0.0, config.Post())
    assert np.array_equal(out, (effects.upscale(img, 3) * 255 + 0.5).astype(np.uint8))


def test_render_returns_output_resolution(cfg):
    out = effects.render(gradient(12, 16), cfg)
    assert out.shape == (24, 32, 3) and out.dtype == np.uint8


CONCURRENT = """
import threading, numpy as np
from cyberglitch import config, effects
img = np.random.default_rng(0).random((180, 320, 3)).astype(np.float32)
def go():
    for n in range(20):
        effects.render(img, config.Config(), n)
ts = [threading.Thread(target=go) for _ in range(3)]
[t.start() for t in ts]
[t.join() for t in ts]
"""


def test_concurrent_renders_do_not_abort_the_process():
    # The web UI renders previews while a render job runs; numba's workqueue layer aborts on
    # concurrent parallel kernels, killing the server. A subprocess keeps an abort out of pytest.
    proc = subprocess.run([sys.executable, "-c", CONCURRENT], capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr[-500:]
