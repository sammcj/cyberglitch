import shutil
import subprocess
import threading
import time

import pytest

pytest.importorskip("starlette")
from starlette.testclient import TestClient  # noqa: E402

from cyberglitch import config, web  # noqa: E402

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")


@pytest.fixture
def client(tmp_path, monkeypatch):
    samples, presets = tmp_path / "samples", tmp_path / "presets"
    samples.mkdir()
    presets.mkdir()
    (presets / "pinkish.toml").write_text('[glow]\nintensity = 0.4\n[tone]\nramp = ["#000000", "#ff00ff"]\n')
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=96x64:rate=10:duration=0.5",
            str(samples / "clip.mp4"),
        ],
        check=True,
    )
    monkeypatch.setattr(web, "SAMPLES", samples)
    monkeypatch.setattr(web, "PRESETS", presets)
    monkeypatch.setattr(web, "WORK", tmp_path / "work")
    return TestClient(web.create_app())


def base_form(**extra):
    form = {"source": "samples/clip.mp4", "output.width": "96", "output.height": "64", "output.pixel": "2"}
    return {**form, **extra}


def test_every_setting_has_an_input_with_help(client):
    page = client.get("/").text
    for sec in config.fields(config.Config):
        for f in config.fields(getattr(config.Config(), sec.name)):
            assert f'name="{sec.name}.{f.name}"' in page
    assert web.HELP["drip.threshold"].startswith("pixels brighter")
    assert "samples/clip.mp4" in page and "pinkish" in page


def test_form_round_trips_to_config():
    cfg = config.load(overrides=["drip.enabled=false", "glow.radius=3.0", "post.chroma=2"])
    form = {}
    for sec in config.fields(cfg):
        obj = getattr(cfg, sec.name)
        for f in config.fields(obj):
            v = getattr(obj, f.name)
            if isinstance(v, bool):
                if v:
                    form[f"{sec.name}.{f.name}"] = "true"
            else:
                form[f"{sec.name}.{f.name}"] = ", ".join(v) if isinstance(v, list) else str(v)
    assert web.config_from_form(form) == cfg


def test_presets_populate_settings(client):
    r = client.post("/fields", data={"preset": "pinkish"})
    assert 'name="glow.intensity" data-default="0.4" value="0.4"' in r.text
    assert r.headers["HX-Trigger-After-Settle"] == "refresh"


def test_preview_renders_png(client):
    r = client.post("/preview", data=base_form(t="0.2", tmax="0"))
    assert '<img src="/files/preview-' in r.text
    name = r.text.split('src="/files/')[1].split('"')[0]
    assert client.get(f"/files/{name}").status_code == 200
    assert 'id="scrub" hx-swap-oob' in r.text  # slider resized to the clip length


def test_preview_reports_bad_values(client):
    r = client.post("/preview", data=base_form(**{"drip.stage": "sideways"}))
    assert 'class="err"' in r.text


def test_unknown_source_rejected(client):
    r = client.post("/preview", data=base_form(source="../../etc/passwd"))
    assert "pick a source" in r.text


def test_render_job_completes(client):
    r = client.post("/render", data=base_form())
    jid = r.text.split('hx-get="/jobs/')[1].split('"')[0]
    for _ in range(100):
        status = client.get(f"/jobs/{jid}").text
        if "<video" in status or 'class="err"' in status:
            break
        time.sleep(0.1)
    assert "<video" in status, status


def test_upload_adds_source(client, tmp_path):
    clip = web.SAMPLES / "clip.mp4"
    with clip.open("rb") as fh:
        r = client.post("/upload", data=base_form(), files={"upload": ("my clip!.mp4", fh, "video/mp4")})
    assert 'value="uploads/my_clip_.mp4" selected' in r.text


def test_save_and_export_preset(client):
    r = client.post("/save", data=base_form(preset_name="My Look", **{"glow.radius": "5"}))
    assert "my-look" in r.text
    saved = config.load([web.PRESETS / "my-look.toml"])
    assert saved.glow.radius == 5.0
    assert "already exists" in client.post("/save", data=base_form(preset_name="my look")).text
    exported = client.get("/export", params=base_form(**{"glow.radius": "7"}))
    assert exported.status_code == 200 and "radius = 7.0" in exported.text


def test_page_requests_first_preview_on_load_and_has_icon(client):
    # An inline htmx.trigger() ran before htmx bound its listeners, so the first preview never loaded
    page = client.get("/").text
    assert 'hx-trigger="load,' in page and "htmx.trigger(" not in page
    assert 'rel="icon"' in page


def test_every_setting_carries_its_loaded_value_for_reset(client):
    page = client.get("/").text
    assert 'name="glow.intensity" data-default="1.6"' in page
    assert 'name="drip.enabled" data-default="true"' in page
    assert 'name="tone.ramp" data-default="#15161a, #69293d' in page
    assert page.count('class="rst"') == page.count("data-default=")
    presets = client.post("/fields", data={"preset": "pinkish"}).text
    assert 'name="glow.intensity" data-default="0.4"' in presets


def test_no_preset_gives_the_reference_look(client):
    assert web.settings_html(config.Config()) in client.post("/fields").text


def test_page_starts_on_the_default_look(client, monkeypatch):
    monkeypatch.setattr(web, "DEFAULT_LOOK", "pinkish")
    page = client.get("/").text
    assert 'value="pinkish" checked' in page and 'name="glow.intensity" data-default="0.4"' in page
    assert ">reference</label>" in page
    assert 'id="reset-all" data-look="pinkish"' in page  # app.js re-selects it, which reloads the fields


def test_child_requests_do_not_inherit_preview_sync(client):
    # Inherited hx-sync="this:replace" let a pending preview abort preset loads and render requests
    page = client.get("/").text
    assert 'hx-disinherit="hx-sync' in page


TYPED = {"output.width", "output.height", "rain.seed", "glitch.seed"}  # exact values, no slider


def test_numeric_settings_are_sliders(client):
    page = client.get("/").text
    for sec in config.fields(config.Config):
        for f in config.fields(getattr(config.Config(), sec.name)):
            key, v = f"{sec.name}.{f.name}", getattr(getattr(config.Config(), sec.name), f.name)
            if isinstance(v, (int, float)) and not isinstance(v, bool) and key not in TYPED:
                assert key in web.RANGES, f"{key} needs a slider range"
    assert page.count('type="range"') == len(web.RANGES) + 1  # plus the clip scrubber
    # A preset value past the usual range widens the slider rather than being clamped
    assert 'max="9.0"' in web.field_input("glow.intensity", 9.0)


def wait_for(client, jid):
    for _ in range(100):
        status = client.get(f"/jobs/{jid}").text
        if "<video" in status or 'class="err"' in status:
            return status
        time.sleep(0.1)
    return status


def test_finished_render_shows_and_reloads_its_settings(client):
    r = client.post(
        "/render", data=base_form(preset="pinkish", **{"glow.radius": "5", "tone.ramp": "#000000, #ff00ff"})
    )
    jid = r.text.split('hx-get="/jobs/')[1].split('"')[0]
    status = wait_for(client, jid)
    assert "pinkish" in status and "glow.radius" in status and "5.0" in status
    assert "background:#ff00ff" in status  # the preset ramp, as swatches
    toml = status.split('href="/files/')[2].split('"')[0]
    assert toml.endswith(".toml") and "radius = 5.0" in client.get(f"/files/{toml}").text
    loaded = client.post(f"/jobs/{jid}/load")
    assert 'name="glow.radius" data-default="5.0"' in loaded.text
    assert 'value="pinkish" checked' in loaded.text and loaded.headers["HX-Trigger-After-Settle"] == "refresh"


def test_renders_queue_and_can_be_stopped(client, monkeypatch):
    started = threading.Event()

    def slow(src, out, cfg, start, length, progress):
        started.set()
        for n in range(1, 500):
            progress(n)
            time.sleep(0.01)

    monkeypatch.setattr(web, "process", slow)
    first = client.post("/render", data=base_form()).text.split('hx-get="/jobs/')[1].split('"')[0]
    assert started.wait(5)
    second = client.post("/render", data=base_form()).text
    assert "queued, 1 ahead" in second and "stop</button>" in second
    second = second.split('hx-get="/jobs/')[1].split('"')[0]
    assert "stopped" in client.post(f"/jobs/{second}/stop").text
    assert not (web.WORK / web.jobs[second]["toml"]).exists()
    client.post(f"/jobs/{first}/stop")
    for _ in range(100):
        if web.jobs[first]["state"] == "stopped":
            break
        time.sleep(0.05)
    assert "stopped at" in client.get(f"/jobs/{first}").text
    assert web.jobs[second]["n"] == 0, "a job stopped while queued never runs"
