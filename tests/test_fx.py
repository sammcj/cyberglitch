import numpy as np

from cyberglitch import config, effects, fx


def frame(h=40, w=60, value=0.0):
    return np.full((h, w, 3), value, np.float32)


def test_defaults_leave_render_unchanged():
    # Extras are opt-in: the base look must not shift when they are added to the pipeline
    cfg = config.Config()
    img = frame(value=0.5)
    assert np.array_equal(fx.haze(img, None, cfg.haze), img)
    assert np.array_equal(fx.rain(img, cfg.rain, 3), img)
    assert np.array_equal(fx.glitch(img, cfg.glitch, 3), img)
    assert np.array_equal(fx.echo(img, cfg.echo, {}), img)
    assert np.all(effects.compose(img, None, 1, 0.0, cfg.post, 3) == 128)


def test_haze_fills_background_not_subject():
    h = config.Haze(strength=1.0, top="#ff0000", bottom="#ff0000")
    img = frame()
    matte = np.zeros((40, 60), np.float32)
    matte[:, 30:] = 1
    out = fx.haze(img, matte, h)
    assert np.allclose(out[:, :30, 0], 1) and np.allclose(out[:, 30:], 0)


def test_echo_leaves_fading_trail():
    e = config.Echo(decay=0.5)
    state: dict = {}
    lit = frame()
    lit[10, 10] = 1
    fx.echo(lit, e, state)
    out = fx.echo(frame(), e, state)
    assert out[10, 10, 0] == 0.5


def test_rain_moves_between_frames_and_is_deterministic():
    r = config.Rain(density=0.3, colour="#ffffff", opacity=1.0)
    a, b = fx.rain(frame(), r, 0), fx.rain(frame(), r, 1)
    assert a.sum() > 0 and not np.array_equal(a, b)
    assert np.array_equal(a, fx.rain(frame(), r, 0))


def test_glitch_shifts_rows_without_losing_pixels():
    g = config.Glitch(chance=1.0, slices=3, max_shift=10)
    img = np.random.default_rng(0).random((40, 60, 3)).astype(np.float32)
    out = fx.glitch(img, g, 5)
    assert not np.array_equal(out, img)
    assert np.allclose(np.sort(out, axis=1), np.sort(img, axis=1))  # rows are rotated, not altered


def post(img, p, frame_no=0):
    return effects.compose(img, None, 1, 0.0, p, frame_no).astype(np.float32) / 255


def test_post_scanlines_darken_alternate_bands():
    out = post(frame(value=1.0), config.Post(scanlines=0.5, scanline_period=2))
    assert np.allclose(out[0], 1) and np.allclose(out[1], 0.5, atol=0.01)


def test_post_chroma_splits_channels():
    img = frame()
    img[:, 30] = 1
    out = post(img, config.Post(chroma=2))
    assert out[0, 32, 0] == 1 and out[0, 28, 2] == 1 and out[0, 30, 1] == 1


def test_post_grain_is_deterministic_per_frame_and_centred():
    p = config.Post(grain=0.05)
    a, b = post(frame(value=0.5), p, 3), post(frame(value=0.5), p, 4)
    assert np.array_equal(a, post(frame(value=0.5), p, 3)) and not np.array_equal(a, b)
    assert abs(a.mean() - 0.5) < 0.01 and a.std() > 0.02


def test_post_vignette_darkens_corners_not_centre():
    out = post(frame(41, 41, 1.0), config.Post(vignette=0.5))
    assert out[20, 20, 0] == 1 and out[0, 0, 0] < 0.6


def test_render_with_all_extras_on():
    cfg = config.load(
        overrides=[
            "haze.strength=0.5",
            "rain.density=0.2",
            "echo.decay=0.8",
            "glitch.chance=1.0",
            "post.grain=0.05",
        ]
    )
    state: dict = {}
    for n in range(3):
        out = effects.render(frame(12, 16, 0.4), cfg, n, None, state)
    assert out.shape == (24, 32, 3) and "echo" in state


def test_haze_background_does_not_feed_drips():
    cfg = config.load(overrides=["haze.strength=1.0", 'haze.top="#ffffff"', 'haze.bottom="#ffffff"'])
    matte = np.zeros((12, 16), np.float32)
    out = effects.render(frame(12, 16, 0.6), cfg, 0, matte)
    hazed = effects.render(
        frame(12, 16, 0.6),
        config.load(
            overrides=[
                "drip.enabled=false",
                *["haze.strength=1.0", 'haze.top="#ffffff"', 'haze.bottom="#ffffff"'],
            ]
        ),
        0,
        matte,
    )
    assert np.array_equal(out, hazed)
