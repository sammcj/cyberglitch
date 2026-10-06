"""Effect settings. Dataclass defaults reproduce the reference flower video."""

import re
import tomllib
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path

# Palette decoded from the author's Dither EYE preset (ARGB ints -> hex)
FLOWER_PALETTE = [
    "#15161a",  # near black (background)
    "#69293d",  # wine
    "#875099",  # purple
    "#f93951",  # hot red
    "#fc4bf9",  # magenta
    "#fe9bfd",  # light pink
    "#b3e060",  # lime (highlight speckle)
    "#f9f0fa",  # near white
]


@dataclass
class Output:
    width: int = 1280  # 0 keeps the source size
    height: int = 720
    pixel: int = 2  # size of one dither pixel in output pixels; 2 at 720p matches the reference grid
    fps: float = 0  # 0 keeps the source frame rate
    crf: int = 16
    audio: bool = False  # copy the source's first audio track


@dataclass
class Matte:
    # none: use frame as-is. luma: key out dark background. rembg: ML subject segmentation
    mode: str = "luma"
    low: float = 0.05  # luma at or below this becomes background
    high: float = 0.15  # luma at or above this is fully subject
    model: str = "u2net"  # rembg model name
    smooth: float = 0.5  # temporal smoothing of the matte, 0 = none, reduces flicker
    accelerate: bool = True  # run rembg on CoreML (GPU/Neural Engine) when available


@dataclass
class Tone:
    brightness: float = 0.04
    contrast: float = 1.15
    saturation: float = 1.0
    gamma: float = 1.0
    # 0 keeps source colours, 1 replaces them with a gradient map through `ramp`
    colourise: float = 1.0
    ramp: list[str] = field(
        default_factory=lambda: ["#15161a", "#69293d", "#f93951", "#875099", "#fe9bfd", "#f9f0fa"]
    )


@dataclass
class Drip:
    enabled: bool = True
    # before: smear the source, then dither over it (textured streaks)
    # after: smear the dithered palette colours (solid, crisp streaks)
    stage: str = "after"
    threshold: float = 0.2  # pixels brighter than this feed the drip below them
    decay: float = 1.0  # per-pixel fade down the streak, 1 = never fades
    length: int = 0  # max streak length in dither pixels, 0 = to the bottom edge
    mix: float = 1.0  # streak opacity over the background
    jitter: float = 0.0  # 0-1, randomly shortens streaks per column


@dataclass
class Dither:
    palette: list[str] = field(default_factory=lambda: list(FLOWER_PALETTE))
    diffuse_x: float = 1.0  # error carried to the next pixel in the row
    diffuse_y: float = 0.0  # error carried to the pixel below
    # Sine wave added to each pixel before quantising; produces the wavy banding texture
    mod_amp: float = 0.25
    mod_period_x: float = 0.0  # in dither pixels, 0 disables
    mod_period_y: float = 6.0
    mod_luma: float = 8.0  # wave cycles across black-to-white; bands then follow the image's shading
    mod_speed: float = 0.0  # phase change per frame (radians), animates the texture
    run_bias: float = 0.04  # favour repeating the previous colour; longer horizontal runs as it rises


@dataclass
class Glow:
    threshold: float = 0.5
    radius: float = 9.5  # gaussian sigma in output pixels
    intensity: float = 1.6


# Optional extras below are all off by default


@dataclass
class Haze:
    strength: float = 0.0  # smog gradient behind the subject
    top: str = "#0e3a44"
    bottom: str = "#7a2e1a"
    falloff: float = 1.0  # above 1 pushes the bottom colour lower


@dataclass
class Rain:
    density: float = 0.0  # drops per dither-grid column
    colour: str = "#19c3c9"
    opacity: float = 0.7
    speed: float = 8.0  # dither pixels per frame
    length: int = 6
    slant: float = 0.15  # horizontal drift per pixel fallen
    seed: int = 1


@dataclass
class Echo:
    decay: float = 0.0  # ghost trails; 0 off, 0.9 long


@dataclass
class Glitch:
    chance: float = 0.0  # probability a frame gets torn
    slices: int = 3
    max_height: int = 8  # dither pixels
    max_shift: int = 40
    channel_swap: bool = False
    seed: int = 7


@dataclass
class Post:
    chroma: int = 0  # red/blue channel offset in output pixels
    scanlines: float = 0.0  # darkening of alternate line bands
    scanline_period: int = 2  # output pixels; matches the 2 px dither grid at 720p
    grain: float = 0.0
    vignette: float = 0.0
    flicker: float = 0.0  # random per-frame brightness dip


@dataclass
class Config:
    output: Output = field(default_factory=Output)
    matte: Matte = field(default_factory=Matte)
    tone: Tone = field(default_factory=Tone)
    drip: Drip = field(default_factory=Drip)
    dither: Dither = field(default_factory=Dither)
    glow: Glow = field(default_factory=Glow)
    haze: Haze = field(default_factory=Haze)
    rain: Rain = field(default_factory=Rain)
    echo: Echo = field(default_factory=Echo)
    glitch: Glitch = field(default_factory=Glitch)
    post: Post = field(default_factory=Post)


HEX = re.compile(r"^#[0-9a-fA-F]{6}$")
CHOICES = {"matte.mode": ("none", "luma", "rembg"), "drip.stage": ("before", "after")}


def validate(cfg: Config) -> None:
    def need(ok: bool, msg: str) -> None:
        if not ok:
            raise ValueError(msg)

    need(cfg.output.pixel >= 1, "output.pixel must be >= 1")
    need(cfg.output.width >= 0 and cfg.output.height >= 0, "output size must be >= 0")
    for key, options in CHOICES.items():
        section, name = key.split(".")
        need(getattr(getattr(cfg, section), name) in options, f"{key} must be one of {', '.join(options)}")
    need(len(cfg.tone.ramp) >= 2, "tone.ramp needs at least 2 colours")
    need(len(cfg.dither.palette) >= 2, "dither.palette needs at least 2 colours")
    need(cfg.post.scanline_period >= 2, "post.scanline_period must be >= 2")
    need(cfg.glitch.slices >= 1 and cfg.glitch.max_height >= 1, "glitch.slices/max_height must be >= 1")
    need(cfg.glitch.max_shift >= 0 and cfg.rain.length >= 1, "glitch.max_shift >= 0, rain.length >= 1")
    for name, c in [
        *(("tone.ramp", c) for c in cfg.tone.ramp),
        *(("dither.palette", c) for c in cfg.dither.palette),
        ("haze.top", cfg.haze.top),
        ("haze.bottom", cfg.haze.bottom),
        ("rain.colour", cfg.rain.colour),
    ]:
        need(bool(HEX.match(c)), f"{name}: {c!r} is not a #rrggbb colour")


def apply(obj, values: dict, path: str = "") -> None:
    known = {f.name: f for f in fields(obj)}
    for key, value in values.items():
        if key not in known:
            raise KeyError(f"unknown setting {path}{key}")
        current = getattr(obj, key)
        if is_dataclass(current):
            if not isinstance(value, dict):
                raise TypeError(f"{path}{key} must be a table")
            apply(current, value, f"{path}{key}.")
        else:
            if isinstance(current, float) and isinstance(value, int) and not isinstance(value, bool):
                value = float(value)
            if type(current) is not type(value):
                raise TypeError(f"{path}{key} expects {type(current).__name__}, got {type(value).__name__}")
            setattr(obj, key, value)


def parse_override(text: str) -> dict:
    """Turn `drip.threshold=0.3` into {"drip": {"threshold": 0.3}} using TOML value syntax."""
    key, sep, raw = text.partition("=")
    if not sep:
        raise ValueError(f"override must be key=value: {text}")
    try:
        value = tomllib.loads(f"v = {raw}")["v"]
    except tomllib.TOMLDecodeError:
        value = raw  # bare strings such as matte.mode=rembg
    out: dict = {}
    node = out
    *parents, leaf = key.strip().split(".")
    for p in parents:
        node = node.setdefault(p, {})
    node[leaf] = value
    return out


def load(presets: Sequence[Path] = (), overrides: Sequence[str] = ()) -> Config:
    cfg = Config()
    for preset in presets:  # later presets win
        apply(cfg, tomllib.loads(Path(preset).read_text()))
    for o in overrides:
        apply(cfg, parse_override(o))
    validate(cfg)
    return cfg


def to_toml(cfg: Config) -> str:
    def fmt(v):
        if isinstance(v, bool):
            return str(v).lower()
        if isinstance(v, str):
            return f'"{v}"'
        if isinstance(v, list):
            return "[" + ", ".join(fmt(x) for x in v) + "]"
        return repr(v)

    lines = []
    for section, values in asdict(cfg).items():
        lines.append(f"[{section}]")
        lines += [f"{k} = {fmt(v)}" for k, v in values.items()]
        lines.append("")
    return "\n".join(lines)
