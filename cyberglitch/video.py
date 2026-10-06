"""ffmpeg decode/encode and the frame loop."""

import contextlib
import functools
import json
import math
import os
import queue
import subprocess
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np

from . import effects
from .config import Config

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
QUEUE_DEPTH = 4  # frames buffered between stages; enough to absorb jitter without much memory


class RenderError(Exception):
    pass


@dataclass
class Source:
    width: int  # as displayed, after rotation metadata is applied
    height: int
    fps: str  # rational string, passed verbatim to ffmpeg so decode and encode agree
    has_audio: bool
    duration: float = 0.0  # seconds, 0 for still images


def _rotation(stream: dict) -> int:
    for sd in stream.get("side_data_list", []):
        if "rotation" in sd:
            return int(sd["rotation"])
    return int(stream.get("tags", {}).get("rotate", 0))


def probe(path: Path) -> Source:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        capture_output=True,
        text=True,
    )
    if proc.returncode:
        raise RenderError(f"ffprobe failed: {proc.stderr.strip()}")
    info = json.loads(proc.stdout)
    streams = info["streams"]
    v = next((s for s in streams if s["codec_type"] == "video"), None)
    if v is None:
        raise RenderError(f"no video stream in {path}")
    w, h = v["width"], v["height"]
    if abs(_rotation(v)) % 180 == 90:
        w, h = h, w
    fps = v.get("r_frame_rate", "0/0")
    # r_frame_rate can be a timebase-like value (e.g. 1000/1) for VFR; fall back to the average
    if fps.endswith("/0") or Fraction(fps) > 120:
        fps = v.get("avg_frame_rate", "30/1")
    if fps.endswith("/0") or Fraction(fps) == 0:
        fps = "30/1"
    try:
        duration = float(info.get("format", {}).get("duration", 0))
    except ValueError:
        duration = 0.0
    return Source(w, h, fps, any(s["codec_type"] == "audio" for s in streams), duration)


def frame_size(cfg: Config, src: Source) -> tuple[int, int, int, int]:
    """Output and work (dither grid) sizes. Output is a multiple of the pixel size and even (yuv420p)."""
    p = cfg.output.pixel
    cw, ch = cfg.output.width, cfg.output.height
    w = cw or (round(ch * src.width / src.height) if ch else src.width)
    h = ch or (round(cw * src.height / src.width) if cw else src.height)
    unit = p * 2 // math.gcd(p, 2)
    out_w, out_h = max(unit, w // unit * unit), max(unit, h // unit * unit)
    return out_w, out_h, out_w // p, out_h // p


@functools.cache
def _rembg_session(model: str, accelerate: bool):
    # Loading the model takes ~1 s; the web UI renders many previews per process
    try:
        import onnxruntime as ort
        from rembg import new_session
    except ImportError as e:
        raise RenderError("matte.mode=rembg needs the extra: uv sync --extra matte") from e
    if accelerate and "CoreMLExecutionProvider" in ort.get_available_providers():
        home = os.environ.get("REMBG_HOME") or os.environ.get("U2NET_HOME") or "~/.rembg"
        cache = Path(home).expanduser() / "coreml"
        cache.mkdir(parents=True, exist_ok=True)
        # NeuralNetwork format measured ~2.7x faster than CPU for u2net on Apple silicon; the
        # compiled model is cached so only the first run pays the compile cost
        opts = {"ModelCacheDirectory": str(cache), "ModelFormat": "NeuralNetwork"}
        coreml = ("CoreMLExecutionProvider", opts)
        try:
            return new_session(model, providers=[coreml, "CPUExecutionProvider"])
        except Exception as e:  # e.g. CoreML can't compile in a sandbox; the CPU path still works
            print(f"CoreML unavailable ({str(e)[:80]}), using CPU", file=sys.stderr)
    return new_session(model, providers=["CPUExecutionProvider"])


class Matter:
    """Produces a 0-1 subject matte per work-resolution frame.

    `raw` is stateless so it can run ahead on another thread; `smooth` carries the temporal state.
    """

    def __init__(self, cfg: Config):
        self.m = cfg.matte
        self.prev: np.ndarray | None = None
        self.session = _rembg_session(self.m.model, self.m.accelerate) if self.m.mode == "rembg" else None

    def raw(self, work: np.ndarray) -> np.ndarray | None:
        if self.m.mode == "none":
            return None
        if self.m.mode == "luma":
            return effects.luma_matte(work, self.m.low, self.m.high)
        from rembg import remove

        u8 = (work * 255).astype(np.uint8)
        matte = np.asarray(remove(u8, session=self.session, only_mask=True), np.float32) / 255
        return matte[..., 0] if matte.ndim == 3 else matte

    def smooth(self, matte: np.ndarray | None) -> np.ndarray | None:
        if matte is not None and self.prev is not None and self.m.smooth > 0:
            matte = self.prev * self.m.smooth + matte * (1 - self.m.smooth)
        self.prev = matte
        return matte


def _decode_cmd(
    path: Path, cfg: Config, src: Source, start: float, duration: float, single: bool
) -> list[str]:
    out_w, out_h, work_w, work_h = frame_size(cfg, src)
    fps = str(cfg.output.fps) if cfg.output.fps else src.fps
    vf = f"fps={fps},scale={out_w}:{out_h}:force_original_aspect_ratio=increase,crop={out_w}:{out_h}"
    vf += f",scale={work_w}:{work_h}:flags=area"
    cmd = ["ffmpeg", "-nostdin", "-v", "error"]
    if start:
        cmd += ["-ss", str(start)]
    cmd += ["-i", str(path)]
    if duration:
        cmd += ["-t", str(duration)]
    if single:
        cmd += ["-frames:v", "1"]
    return cmd + ["-vf", vf, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]


def _encode_cmd(path: Path, out: Path, cfg: Config, src: Source, start: float, duration: float) -> list[str]:
    out_w, out_h, _, _ = frame_size(cfg, src)
    fps = str(cfg.output.fps) if cfg.output.fps else src.fps
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24"]
    cmd += ["-s", f"{out_w}x{out_h}", "-r", fps, "-i", "-"]
    if out.suffix.lower() in IMAGE_EXTS:
        return cmd + ["-frames:v", "1", str(out)]
    if cfg.output.audio and src.has_audio:
        if start:
            cmd += ["-ss", str(start)]
        cmd += ["-i", str(path)]
        if duration:
            cmd += ["-t", str(duration)]
        cmd += ["-map", "0:v", "-map", "1:a:0", "-c:a", "aac", "-b:a", "192k", "-shortest"]
    cmd += ["-c:v", "libx264", "-crf", str(cfg.output.crf), "-preset", "slow", "-pix_fmt", "yuv420p"]
    return cmd + ["-movflags", "+faststart", str(out)]


def process(
    src_path: Path,
    out_path: Path,
    cfg: Config,
    start: float = 0,
    duration: float = 0,
    progress: Callable[[int], None] | None = None,
) -> int:
    """Render src to out. Returns the frame count; raises RenderError on failure.

    Three stages overlap: a reader thread decodes and computes raw mattes (ONNX releases the GIL and
    can run on the GPU/Neural Engine), the calling thread renders, and a writer thread feeds the encoder.
    """
    src = probe(src_path)
    _, _, work_w, work_h = frame_size(cfg, src)
    single = out_path.suffix.lower() in IMAGE_EXTS
    frame_bytes = work_w * work_h * 3
    matter = Matter(cfg)
    state: dict = {}
    dec = subprocess.Popen(_decode_cmd(src_path, cfg, src, start, duration, single), stdout=subprocess.PIPE)
    enc = subprocess.Popen(_encode_cmd(src_path, out_path, cfg, src, start, duration), stdin=subprocess.PIPE)
    assert dec.stdout and enc.stdin
    dec_out, enc_in = dec.stdout, enc.stdin
    frames: queue.Queue = queue.Queue(maxsize=QUEUE_DEPTH)
    encoded: queue.Queue = queue.Queue(maxsize=QUEUE_DEPTH)
    stop = threading.Event()
    failures: list[BaseException] = []

    def put(q: queue.Queue, item) -> None:
        while not stop.is_set():
            with contextlib.suppress(queue.Full):
                q.put(item, timeout=0.1)
                return

    def read() -> None:
        try:
            while not stop.is_set() and len(buf := dec_out.read(frame_bytes)) == frame_bytes:
                work = np.frombuffer(buf, np.uint8).reshape(work_h, work_w, 3).astype(np.float32) / 255
                put(frames, (work, matter.raw(work)))
        except BaseException as e:
            failures.append(e)
        finally:
            put(frames, None)

    broken = threading.Event()  # encoder gone: stop rendering frames nobody will receive

    def write() -> None:
        while (data := encoded.get()) is not None:
            if broken.is_set():
                continue  # keep draining so the renderer never blocks; the encoder's exit code is reported
            try:
                enc_in.write(data)
            except OSError:  # BrokenPipe or similar
                broken.set()

    reader = threading.Thread(target=read, daemon=True)
    writer = threading.Thread(target=write, daemon=True)
    reader.start()
    writer.start()
    n = 0
    ok = False
    try:
        while not broken.is_set() and (item := frames.get()) is not None:
            work, raw = item
            encoded.put(effects.render(work, cfg, n, matter.smooth(raw), state).tobytes())
            n += 1
            if progress:
                progress(n)
        ok = not failures and not broken.is_set()
    finally:
        stop.set()  # unblocks the reader if rendering stopped part way
        encoded.put(None)
        writer.join()
        if not ok:
            # Killing rather than closing stdin stops ffmpeg finalising a truncated but playable file,
            # and unblocks a reader stuck on a decoder read
            enc.kill()
            dec.kill()
        with contextlib.suppress(BrokenPipeError):
            enc_in.close()
        reader.join()
        dec_out.close()
        dec.wait()
        enc.wait()
        if not ok:
            out_path.unlink(missing_ok=True)
    if failures:
        raise RenderError(f"matte failed: {failures[0]}") from failures[0]
    if enc.returncode:
        raise RenderError(f"ffmpeg encode failed ({enc.returncode})")
    if dec.returncode:
        raise RenderError(f"ffmpeg decode failed ({dec.returncode})")
    if n == 0:
        raise RenderError("no frames decoded (check --start/--duration)")
    return n
