import argparse
import sys
from pathlib import Path

from . import config


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="cyberglitch", description="Palette dither + drip + glow effect")
    ap.add_argument("input", type=Path, nargs="?", help="video or image")
    ap.add_argument("output", type=Path, nargs="?", help=".mp4/.mov for video, .png/.jpg for a single frame")
    ap.add_argument("-p", "--preset", type=Path, action="append", default=[], help="TOML preset, repeatable")
    ap.add_argument(
        "-s", "--set", action="append", default=[], metavar="KEY=VALUE", help="e.g. drip.threshold=0.3"
    )
    ap.add_argument("--start", type=float, default=0, help="seconds into the input")
    ap.add_argument("--duration", type=float, default=0, help="seconds to render, 0 = all")
    ap.add_argument(
        "--dump-config", action="store_true", help="print the effective settings as TOML and exit"
    )
    args = ap.parse_args(argv)

    try:
        cfg = config.load(args.preset, args.set)
    except (KeyError, TypeError, ValueError) as e:
        sys.exit(f"config error: {e}")
    if args.dump_config:
        print(config.to_toml(cfg))
        return
    if not (args.input and args.output):
        ap.error("input and output are required")
    from .video import RenderError, process  # deferred so --dump-config skips the numba import

    def progress(n: int) -> None:
        if n % 30 == 0:
            print(f"\r{n} frames", end="", file=sys.stderr, flush=True)

    try:
        n = process(args.input, args.output, cfg, args.start, args.duration, progress)
    except RenderError as e:
        sys.exit(f"\nerror: {e}")
    print(f"\r{n} frames -> {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
