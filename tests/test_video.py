import shutil
import subprocess

import pytest

from cyberglitch import config, video
from cyberglitch.video import Source

needs_ffmpeg = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")


SRC = Source(1366, 768, "30/1", False)


def sizes(**out):
    return video.frame_size(config.load(overrides=[f"output.{k}={v}" for k, v in out.items()]), SRC)


def test_frame_size_is_even_and_pixel_aligned():
    for w, h, p in [(1001, 1001, 3), (1365, 767, 3), (640, 360, 4), (101, 77, 1)]:
        out_w, out_h, work_w, work_h = sizes(width=w, height=h, pixel=p)
        assert out_w % 2 == 0 and out_h % 2 == 0
        assert (out_w, out_h) == (work_w * p, work_h * p)
        assert out_w <= w and out_h <= h


def test_frame_size_derives_missing_axis_from_source_aspect():
    assert sizes(width=0, height=0, pixel=3)[:2] == (1362, 768)
    out_w, out_h, *_ = sizes(width=0, height=540)
    assert out_h == 540 and abs(out_w - 960) <= 6


def test_encode_cmd_image_output_writes_one_frame(tmp_path):
    cmd = video._encode_cmd(
        tmp_path / "in.mp4", tmp_path / "x.png", config.Config(), Source(64, 64, "25/1", True), 0, 0
    )
    assert cmd[-3:] == ["-frames:v", "1", str(tmp_path / "x.png")]
    assert "-map" not in cmd


def test_audio_is_off_by_default(tmp_path):
    cmd = video._encode_cmd(
        tmp_path / "a.mp4", tmp_path / "o.mp4", config.Config(), Source(64, 64, "25/1", True), 0, 0
    )
    assert "-map" not in cmd


def test_encode_cmd_maps_first_audio_track_only_when_present(tmp_path):
    cfg = config.load(overrides=["output.audio=true"])
    with_audio = video._encode_cmd(
        tmp_path / "a.mp4", tmp_path / "o.mp4", cfg, Source(64, 64, "25/1", True), 2, 3
    )
    assert "1:a:0" in with_audio and "-shortest" in with_audio
    silent = video._encode_cmd(
        tmp_path / "a.mp4", tmp_path / "o.mp4", cfg, Source(64, 64, "25/1", False), 0, 0
    )
    assert "-map" not in silent


@pytest.fixture
def clip(tmp_path):
    path = tmp_path / "src.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=96x64:rate=10:duration=0.5", str(path)],
        check=True,
    )
    return path


@needs_ffmpeg
def test_process_end_to_end(clip, tmp_path):
    out = tmp_path / "out.mp4"
    cfg = config.load(overrides=["output.width=96", "output.height=64", "output.pixel=2", "rain.density=0.2"])
    assert video.process(clip, out, cfg) == 5
    assert video.probe(out).width == 96


@needs_ffmpeg
def test_process_with_no_frames_raises(clip, tmp_path):
    cfg = config.load(overrides=["output.width=96", "output.height=64"])
    with pytest.raises(video.RenderError, match="no frames"):
        video.process(clip, tmp_path / "out.png", cfg, start=99)


@needs_ffmpeg
def test_failed_render_leaves_no_truncated_output(clip, tmp_path, monkeypatch):
    real = video.effects.render

    def fail_on_third(work, cfg, n, *args):
        if n == 2:
            raise RuntimeError("boom")
        return real(work, cfg, n, *args)

    monkeypatch.setattr(video.effects, "render", fail_on_third)
    out = tmp_path / "out.mp4"
    with pytest.raises(RuntimeError, match="boom"):
        video.process(clip, out, config.load(overrides=["output.width=96", "output.height=64"]))
    assert not out.exists()


@needs_ffmpeg
def test_dead_encoder_stops_rendering_early(clip, tmp_path, monkeypatch):
    # Encoder exits at once (unwritable output); the loop should not render the whole clip first
    rendered = []
    real = video.effects.render
    monkeypatch.setattr(
        video.effects, "render", lambda w, c, n, *a: (rendered.append(n), real(w, c, n, *a))[1]
    )
    long_clip = tmp_path / "long.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=96x64:rate=30:duration=10",
            str(long_clip),
        ],
        check=True,
    )
    cfg = config.load(overrides=["output.width=96", "output.height=64"])
    with pytest.raises(video.RenderError, match="encode failed"):
        video.process(long_clip, tmp_path / "missing" / "out.mp4", cfg)
    assert len(rendered) < 300
