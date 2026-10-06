import tomllib

import pytest

from cyberglitch import config


def test_override_parses_toml_values():
    cfg = config.load(overrides=["drip.threshold=0.3", "matte.mode=rembg", 'tone.ramp=["#000000","#ffffff"]'])
    assert cfg.drip.threshold == 0.3
    assert cfg.matte.mode == "rembg"
    assert cfg.tone.ramp == ["#000000", "#ffffff"]


def test_int_accepted_for_float_setting():
    assert config.load(overrides=["glow.radius=20"]).glow.radius == 20.0


def test_unknown_key_rejected():
    with pytest.raises(KeyError):
        config.load(overrides=["drip.thresold=0.3"])


def test_wrong_type_rejected():
    with pytest.raises(TypeError):
        config.load(overrides=["output.pixel=2.5"])


def test_preset_file_and_overrides_layer(tmp_path):
    preset = tmp_path / "p.toml"
    preset.write_text("[glow]\nintensity = 0.2\nradius = 5.0\n")
    cfg = config.load([preset], ["glow.radius=9.0"])
    assert (cfg.glow.intensity, cfg.glow.radius) == (0.2, 9.0)


def test_dump_round_trips():
    cfg = config.Config()
    assert config.load(overrides=[]) == cfg
    dumped = tomllib.loads(config.to_toml(cfg))
    reloaded = config.Config()
    config.apply(reloaded, dumped)
    assert reloaded == cfg


def test_shipped_presets_load_with_ramps_ordered_dark_to_light():
    from pathlib import Path

    import numpy as np

    from cyberglitch.effects import LUMA, hex_to_rgb

    presets = sorted(Path(__file__).parent.parent.joinpath("presets").glob("*.toml"))
    assert len(presets) >= 5
    for p in presets:
        cfg = config.load([p])
        lum = hex_to_rgb(cfg.tone.ramp) @ LUMA
        # Small tolerance: the reference ramp puts red just before an equally bright purple on purpose
        assert np.all(np.diff(lum) > -0.02), f"{p.name}: tone.ramp not ordered by brightness"


@pytest.mark.parametrize(
    "override",
    [
        "output.pixel=0",
        "drip.stage=befor",
        "matte.mode=magic",
        'tone.ramp=["#000000"]',
        'dither.palette=["#fff","#000000"]',
        "rain.colour=red",
        "post.scanline_period=1",
    ],
)
def test_invalid_values_rejected_at_load(override):
    with pytest.raises(ValueError):
        config.load(overrides=[override])
