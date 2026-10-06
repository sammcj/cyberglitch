"""Local web UI: pick a clip, stack presets, tweak any setting with a live preview, render.

Every input is generated from the Config dataclasses, so new settings appear here automatically.
"""

import argparse
import html
import re
import shutil
import time
import tomllib
import uuid
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields
from fractions import Fraction
from pathlib import Path

from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile
from starlette.requests import Request
from starlette.responses import HTMLResponse, Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from . import config
from .video import IMAGE_EXTS, RenderError, probe, process

ROOT = Path(__file__).resolve().parent.parent
STATIC = Path(__file__).resolve().parent / "static"
PRESETS = ROOT / "presets"
SAMPLES = ROOT / "samples"
WORK = ROOT / "out" / "web"
MEDIA_EXTS = IMAGE_EXTS | {".mp4", ".mov", ".mkv", ".webm", ".ogv", ".avi", ".m4v", ".gif"}
DEFAULT_CLIP = "flower-black.webm"
DEFAULT_LOOK = "hologram"  # the UI starts here; no preset ("reference") is the built-in look
OPEN_SECTIONS = {"tone", "drip", "dither", "glow"}
FORM_ERRORS = (RenderError, ValueError, KeyError, TypeError)
# After settle, so the preview it triggers posts the newly swapped-in settings (HX-Trigger fires before the swap)
REFRESH = {"HX-Trigger-After-Settle": "refresh"}
# Slider (min, max, step) per numeric setting. A loaded value outside the range widens it.
RANGES: dict[str, tuple[float, float, float]] = {
    "output.pixel": (1, 8, 1),
    "output.fps": (0, 60, 1),
    "output.crf": (0, 40, 1),
    "matte.low": (0, 1, 0.01),
    "matte.high": (0, 1, 0.01),
    "matte.smooth": (0, 0.95, 0.01),
    "tone.brightness": (-0.5, 0.5, 0.01),
    "tone.contrast": (0, 3, 0.01),
    "tone.saturation": (0, 3, 0.01),
    "tone.gamma": (0.2, 3, 0.01),
    "tone.colourise": (0, 1, 0.01),
    "drip.threshold": (0, 1, 0.01),
    "drip.decay": (0.8, 1, 0.001),
    "drip.length": (0, 300, 1),
    "drip.mix": (0, 1, 0.01),
    "drip.jitter": (0, 1, 0.01),
    "dither.diffuse_x": (0, 1, 0.01),
    "dither.diffuse_y": (0, 1, 0.01),
    "dither.mod_amp": (0, 1, 0.01),
    "dither.mod_period_x": (0, 64, 0.5),
    "dither.mod_period_y": (0, 64, 0.5),
    "dither.mod_luma": (0, 32, 0.5),
    "dither.mod_speed": (0, 1, 0.01),
    "dither.run_bias": (0, 0.3, 0.005),
    "glow.threshold": (0, 1, 0.01),
    "glow.radius": (0, 40, 0.5),
    "glow.intensity": (0, 4, 0.05),
    "haze.strength": (0, 1, 0.01),
    "haze.falloff": (0.2, 4, 0.05),
    "rain.density": (0, 1, 0.01),
    "rain.opacity": (0, 1, 0.01),
    "rain.speed": (0, 40, 0.5),
    "rain.length": (1, 40, 1),
    "rain.slant": (-1, 1, 0.01),
    "echo.decay": (0, 0.98, 0.01),
    "glitch.chance": (0, 1, 0.01),
    "glitch.slices": (1, 12, 1),
    "glitch.max_height": (1, 60, 1),
    "glitch.max_shift": (0, 200, 1),
    "post.chroma": (0, 12, 1),
    "post.scanlines": (0, 1, 0.01),
    "post.scanline_period": (2, 12, 1),
    "post.grain": (0, 0.3, 0.005),
    "post.vignette": (0, 1, 0.01),
    "post.flicker": (0, 0.5, 0.01),
}

jobs: dict[str, dict] = {}
runner = ThreadPoolExecutor(max_workers=1)  # renders queue; each one already uses every core


def field_help() -> dict[str, str]:
    """Comments beside each setting in config.py, so UI tooltips can't drift from the code."""
    sections = {type(getattr(config.Config(), f.name)).__name__: f.name for f in fields(config.Config)}
    found: dict[str, str] = {}
    section, pending = None, []
    for line in Path(config.__file__).read_text().splitlines():
        if m := re.match(r"class (\w+)", line):
            section, pending = sections.get(m[1]), []
        elif section and (m := re.match(r"\s+#\s*(.*)", line)):
            pending.append(m[1])
        elif section and (m := re.match(r"\s+(\w+): .*?(?:  #\s(.*))?$", line)):
            found[f"{section}.{m[1]}"] = " ".join([*pending, m[2] or ""]).strip()
            pending = []
        else:
            pending = []
    return found


HELP = field_help()


def uploads() -> Path:
    return WORK / "uploads"


def sources() -> list[Path]:
    found = []
    for d in (uploads(), SAMPLES):
        if d.is_dir():
            found += sorted(p for p in d.iterdir() if p.suffix.lower() in MEDIA_EXTS)
    return found


def source_id(p: Path) -> str:
    return f"{p.parent.name}/{p.name}"


def resolve_source(value: str | None) -> Path:
    # Only files we listed ourselves, never a client-supplied path
    for p in sources():
        if source_id(p) == value:
            return p
    raise RenderError("upload or pick a source clip")


def preset_names() -> list[str]:
    return sorted(p.stem for p in PRESETS.glob("*.toml"))


def config_from_form(form) -> config.Config:
    cfg = config.Config()
    values: dict = {}
    for sec in fields(cfg):
        obj = getattr(cfg, sec.name)
        for f in fields(obj):
            key, default = f"{sec.name}.{f.name}", getattr(obj, f.name)
            raw = form.get(key)
            if isinstance(default, bool):
                value = raw == "true"
            elif raw is None:
                continue
            elif isinstance(default, list):
                value = [c.strip() for c in str(raw).split(",") if c.strip()]
            elif isinstance(default, int):
                value = int(float(raw))
            elif isinstance(default, float):
                value = float(raw)
            else:
                value = str(raw)
            values.setdefault(sec.name, {})[f.name] = value
    config.apply(cfg, values)
    config.validate(cfg)
    return cfg


def fv(form, key: str, default: str = "") -> str:
    """String form value; ignores file uploads."""
    v = form.get(key)
    return v if isinstance(v, str) else default


def esc(v) -> str:
    return html.escape(str(v), quote=True)


def field_input(key: str, value) -> str:
    """One labelled control. `data-default` holds the value it was loaded with, for the reset button."""
    shown = (
        str(value).lower()
        if isinstance(value, bool)
        else ", ".join(value)
        if isinstance(value, list)
        else value
    )
    attrs = f'name="{key}" data-default="{esc(shown)}"'
    swatches = ""
    if key in config.CHOICES:
        opts = "".join(
            f"<option{' selected' if o == value else ''}>{o}</option>" for o in config.CHOICES[key]
        )
        widget = f"<select {attrs}>{opts}</select>"
    elif isinstance(value, bool):
        widget = f'<input type="checkbox" {attrs} value="true"{" checked" if value else ""}>'
    elif isinstance(value, list):
        widget = f'<input type="text" {attrs} value="{esc(shown)}">'
        swatches = (
            '<span class="sw">' + "".join(f'<i style="background:{esc(c)}"></i>' for c in value) + "</span>"
        )
    elif isinstance(value, str) and config.HEX.match(value):
        widget = f'<input type="color" {attrs} value="{value}">'
    elif isinstance(value, str):
        widget = f'<input type="text" {attrs} value="{esc(value)}">'
    elif key in RANGES:
        lo, hi, step = RANGES[key]
        # The number box is the submitted value so typed values aren't snapped to the slider step
        widget = (
            f'<span class="sl"><input type="range" min="{min(lo, value)}" max="{max(hi, value)}"'
            f' step="{step}" value="{value}" aria-label="{key}">'
            f'<input type="number" {attrs} value="{value}" step="any"></span>'
        )
    else:
        widget = f'<input type="number" {attrs} value="{value}" step="1">'
    label = key.split(".", 1)[1].replace("_", " ")
    reset = '<button type="button" class="rst" title="reset">&#8634;</button>'
    return f'<label title="{esc(HELP.get(key, ""))}"><span>{label}</span>{reset}{widget}{swatches}</label>'


def settings_html(cfg: config.Config) -> str:
    parts = []
    for sec in fields(cfg):
        obj = getattr(cfg, sec.name)
        inputs = "".join(field_input(f"{sec.name}.{f.name}", getattr(obj, f.name)) for f in fields(obj))
        is_open = " open" if sec.name in OPEN_SECTIONS else ""
        parts.append(f"<details{is_open}><summary>{sec.name}</summary>{inputs}</details>")
    return f'<div id="settings">{"".join(parts)}</div>'


def default_source() -> Path | None:
    # A lit subject on black, like the reference, so the built-in look shows as intended
    items = sources()
    return next((p for p in items if p.name == DEFAULT_CLIP), items[0] if items else None)


def sources_html(selected: str | None = None) -> str:
    items = sources()
    first = default_source()
    selected = selected or (source_id(first) if first else None)
    opts = "".join(
        f'<option value="{esc(source_id(p))}"{" selected" if source_id(p) == selected else ""}>{esc(source_id(p))}</option>'
        for p in items
    )
    return f'<select id="sources" name="source">{opts or "<option value=>no clips yet</option>"}</select>'


def is_modifier(name: str) -> bool:
    """Presets that only change the matte (e.g. any-background) stack on top of a look."""
    try:
        data = tomllib.loads((PRESETS / f"{name}.toml").read_text())
    except (OSError, tomllib.TOMLDecodeError):
        return False
    return bool(data) and set(data) <= {"matte"}


def presets_html(checked: Sequence[str] = ()) -> str:
    """One look (radio) plus any matte modifiers (checkboxes); /fields stacks them in that order."""
    names = preset_names()
    looks = [n for n in names if not is_modifier(n)]

    def chip(kind: str, value: str, label: str, on: bool) -> str:
        return (
            f'<label class="chip"><input type="{kind}" class="np" name="preset" value="{esc(value)}"'
            f'{" checked" if on else ""} hx-post="/fields" hx-trigger="change" hx-target="#settings"'
            f' hx-swap="outerHTML" hx-include="#ctl">{esc(label)}</label>'
        )

    radios = chip("radio", "", "reference", not any(n in looks for n in checked))
    radios += "".join(chip("radio", n, n, n in checked) for n in looks)
    mods = "".join(chip("checkbox", n, n, n in checked) for n in names if n not in looks)
    return f'<div id="presets">{radios}{f"<hr>{mods}" if mods else ""}</div>'


def changed_settings(cfg: config.Config) -> list[tuple[str, object]]:
    base = config.Config()
    out = []
    for sec in fields(cfg):
        obj, ref = getattr(cfg, sec.name), getattr(base, sec.name)
        out += [
            (f"{sec.name}.{f.name}", getattr(obj, f.name))
            for f in fields(obj)
            if getattr(obj, f.name) != getattr(ref, f.name)
        ]
    return out


def value_html(v) -> str:
    """Colours as swatches; long hex lists are unreadable as text."""
    colours = v if isinstance(v, list) else [v] if isinstance(v, str) and config.HEX.match(v) else None
    if colours:
        cells = "".join(f'<i style="background:{esc(c)}"></i>' for c in colours)
        return f'<b class="swl" title="{esc(", ".join(colours))}">{cells}</b>'
    return esc(str(v).lower() if isinstance(v, bool) else v)


def oob(fragment: str, element_id: str) -> str:
    return fragment.replace(f'id="{element_id}"', f'id="{element_id}" hx-swap-oob="true"', 1)


def scrub_html(duration: float, t: float = 0) -> str:
    # The hidden max lets /preview spot a source change and resize the slider
    return (
        f'<span id="scrub"><input type="range" name="t" min="0" max="{duration:.2f}" step="0.05" value="{t:.2f}"'
        f' oninput="this.nextElementSibling.value=(+this.value).toFixed(2)+&quot;s&quot;"><output>{t:.2f}s</output>'
        f'<input type="hidden" name="tmax" value="{duration:.2f}"></span>'
    )


def error_html(e: Exception | str) -> str:
    return f'<p class="err">{esc(e)}</p>'


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>cyberglitch</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'%3E%3Crect width='16' height='16' fill='%2315161a'/%3E%3Crect x='4' y='2' width='3' height='12' fill='%23fc4bf9'/%3E%3Crect x='9' y='2' width='3' height='8' fill='%2319c3c9'/%3E%3C/svg%3E">
<link rel="stylesheet" href="/static/style.css"><script src="/static/htmx.min.js"></script></head>
<body>
<form id="ctl" onsubmit="return false" hx-post="/preview" hx-target="#preview" hx-sync="this:replace"
  hx-disinherit="hx-sync hx-indicator hx-target" hx-indicator="#preview" hx-trigger="load, input[!target.closest('.np')] delay:350ms, refresh from:body">
<aside>
  <h1>cyberglitch</h1>
  <section><h2>source</h2>{sources}
    <input type="file" name="upload" class="np" accept="video/*,image/*" hx-post="/upload"
      hx-encoding="multipart/form-data" hx-trigger="change" hx-target="#sources" hx-swap="outerHTML">
  </section>
  <section><h2>presets <button type="button" id="reset-all" data-look="{look}">reset all</button></h2>
    {presets}</section>
  {settings}
</aside>
<main>
  <div id="preview"><p class="hint">loading preview</p></div>
  <div class="bar">{scrub}</div>
  <div class="bar np">
    <label>from <input type="number" name="rstart" value="0" min="0" step="0.5"></label>
    <label>length <input type="number" name="rlen" value="0" min="0" step="0.5" title="0 = to the end"></label>
    <button type="button" hx-post="/render" hx-include="#ctl" hx-target="#jobs" hx-swap="afterbegin">render</button>
    <input type="text" name="preset_name" placeholder="preset name" size="12">
    <button type="button" hx-post="/save" hx-include="#ctl" hx-target="#presets" hx-swap="outerHTML">save preset</button>
    <button type="button" onclick="location='/export?'+new URLSearchParams(new FormData(this.form))">toml</button>
  </div>
  <div id="jobs"></div>
</main>
</form>
<script src="/static/app.js"></script>
</body></html>"""


async def index(request: Request) -> HTMLResponse:
    first = default_source()
    look = DEFAULT_LOOK if DEFAULT_LOOK in preset_names() else ""
    duration = 0.0
    if first:
        try:
            duration = (await run_in_threadpool(probe, first)).duration
        except RenderError:
            pass
    page = PAGE.format(
        sources=sources_html(),
        look=esc(look),
        presets=presets_html([look] if look else []),
        settings=settings_html(config.load([PRESETS / f"{look}.toml"] if look else [])),
        scrub=scrub_html(duration),
    )
    return HTMLResponse(page)


async def fields_for_presets(request: Request) -> HTMLResponse:
    form = await request.form()
    chosen = [n for n in form.getlist("preset") if n in preset_names()]
    try:
        cfg = config.load([PRESETS / f"{n}.toml" for n in chosen])
    except FORM_ERRORS as e:
        return HTMLResponse(f'<div id="settings">{error_html(e)}</div>')
    return HTMLResponse(settings_html(cfg), headers=REFRESH)


def _clean_previews(max_age: float = 60) -> None:
    cutoff = time.time() - max_age
    for p in WORK.glob("preview-*.png"):
        if p.stat().st_mtime < cutoff:
            p.unlink(missing_ok=True)


def _preview(src: Path, cfg: config.Config, t: float) -> tuple[str, float]:
    info = probe(src)
    t = min(t, max(0.0, info.duration - 0.1))
    name = f"preview-{uuid.uuid4().hex[:10]}.png"
    process(src, WORK / name, cfg, start=t)
    _clean_previews()
    return name, info.duration


async def preview(request: Request) -> HTMLResponse:
    form = await request.form()
    try:
        src = resolve_source(fv(form, "source"))
        cfg = config_from_form(form)
        t = float(fv(form, "t", "0") or 0)
        name, duration = await run_in_threadpool(_preview, src, cfg, t)
    except FORM_ERRORS as e:
        return HTMLResponse(error_html(e))
    body = f'<img src="/files/{name}" alt="preview">'
    if abs(duration - float(fv(form, "tmax", "0") or 0)) > 0.01:
        body += oob(scrub_html(duration, min(t, duration)), "scrub")
    return HTMLResponse(body)


async def upload(request: Request) -> HTMLResponse:
    form = await request.form()
    file = form.get("upload")
    if not isinstance(file, UploadFile) or not file.filename:
        return HTMLResponse(sources_html(fv(form, "source")))
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(file.filename).name)
    if Path(name).suffix.lower() not in MEDIA_EXTS:
        return HTMLResponse(sources_html(fv(form, "source")) + error_html(f"unsupported file type: {name}"))
    uploads().mkdir(parents=True, exist_ok=True)
    dest = uploads() / name
    with dest.open("wb") as out:
        await run_in_threadpool(shutil.copyfileobj, file.file, out)
    return HTMLResponse(sources_html(source_id(dest)), headers=REFRESH)


class Stopped(Exception):
    pass


def _run_job(jid: str, src: Path, cfg: config.Config, start: float, length: float) -> None:
    job = jobs[jid]
    if job["stop"]:  # stopped while queued
        return
    job["state"] = "running"

    def progress(n: int) -> None:
        job["n"] = n
        if job["stop"]:
            raise Stopped  # process() then kills ffmpeg and removes the partial output

    try:
        process(src, WORK / job["out"], cfg, start, length, progress)
        job["state"] = "done"
    except Stopped:
        job["state"] = "stopped"
    except Exception as e:  # surfaced in the UI; the worker thread must not die silently
        job["state"], job["error"] = "error", str(e)
    if job["state"] != "done":
        (WORK / job["toml"]).unlink(missing_ok=True)


async def render(request: Request) -> HTMLResponse:
    form = await request.form()
    try:
        src = resolve_source(fv(form, "source"))
        cfg = config_from_form(form)
        start, length = float(fv(form, "rstart") or 0), float(fv(form, "rlen") or 0)
        info = await run_in_threadpool(probe, src)
    except FORM_ERRORS as e:
        return HTMLResponse(error_html(e))
    jid = uuid.uuid4().hex[:10]
    single = src.suffix.lower() in IMAGE_EXTS
    seconds = length or max(0.0, info.duration - start)
    presets = [n for n in form.getlist("preset") if isinstance(n, str) and n in preset_names()]
    stem = f"{src.stem}-{jid}"
    # Sidecar so the settings behind a render survive a server restart
    header = f"# {source_id(src)} from {start}s, presets: {', '.join(presets) or 'none'}\n"
    (WORK / f"{stem}.toml").write_text(header + config.to_toml(cfg))
    jobs[jid] = {
        "state": "queued",
        "n": 0,
        "total": 1 if single else max(1, round(seconds * float(Fraction(info.fps)))),
        "out": f"{stem}{'.png' if single else '.mp4'}",
        "toml": f"{stem}.toml",
        "error": "",
        "stop": False,
        "cfg": cfg,
        "presets": presets,
        "source": source_id(src),
        "span": f"{start:g}s+{length:g}s" if length else f"{start:g}s-end",
    }
    runner.submit(_run_job, jid, src, cfg, start, length)
    return HTMLResponse(job_html(jid))


def job_summary(job: dict) -> str:
    changed = changed_settings(job["cfg"])
    rows = "".join(f"<li><span>{esc(k)}</span>{value_html(v)}</li>" for k, v in changed)
    looks = " + ".join(job["presets"]) or "reference"
    return (
        f"<details><summary><b>{esc(looks)}</b> {esc(job['source'])} {job['span']},"
        f' {len(changed)} changed</summary><ul class="kv">{rows}</ul></details>'
    )


def job_html(jid: str) -> str:
    job = jobs[jid]
    name = esc(job["out"])
    dismiss = (
        '<button type="button" class="x" onclick="this.closest(&quot;.job&quot;).remove()">dismiss</button>'
    )
    if job["state"] in ("queued", "running"):
        pct = min(100, 100 * job["n"] / job["total"])
        if job["state"] == "queued":
            ahead = sum(
                j["state"] in ("queued", "running") for j in list(jobs.values())[: list(jobs).index(jid)]
            )
            status = f"queued, {ahead} ahead"
        else:
            status = "stopping" if job["stop"] else f"rendering {job['n']}/{job['total']}"
        return (
            f'<div class="job" hx-get="/jobs/{jid}" hx-trigger="every 1s" hx-swap="outerHTML">'
            f'<progress max="100" value="{pct:.0f}"></progress><div class="meta"><span>{status}</span>'
            f'<button type="button" hx-post="/jobs/{jid}/stop" hx-params="none" hx-target="closest .job"'
            f' hx-swap="outerHTML">stop</button></div>{job_summary(job)}</div>'
        )
    if job["state"] == "error":
        return f'<div class="job">{error_html(job["error"])}<div class="meta">{dismiss}</div></div>'
    if job["state"] == "stopped":
        return (
            f'<div class="job"><div class="meta"><span class="hint">stopped at {job["n"]}/{job["total"]}'
            f"</span>{dismiss}</div>{job_summary(job)}</div>"
        )
    media = (
        f'<img src="/files/{name}" alt="">'
        if name.endswith(".png")
        else f'<video src="/files/{name}" controls loop></video>'
    )
    return (
        f'<div class="job">{media}<div class="meta"><a href="/files/{name}" download>{name}</a>'
        f'<a href="/files/{esc(job["toml"])}" download>toml</a>'
        f'<button type="button" hx-post="/jobs/{jid}/load" hx-params="none" hx-target="#settings"'
        f' hx-swap="outerHTML" title="load these settings into the editor">load</button></div>'
        f"{job_summary(job)}</div>"
    )


async def job_status(request: Request) -> HTMLResponse:
    jid = request.path_params["jid"]
    if jid not in jobs:
        return HTMLResponse("")
    return HTMLResponse(job_html(jid))


async def stop_job(request: Request) -> HTMLResponse:
    jid = request.path_params["jid"]
    if jid not in jobs:
        return HTMLResponse("")
    job = jobs[jid]
    job["stop"] = True
    if job["state"] == "queued":
        job["state"] = "stopped"  # the worker skips it when its turn comes
        (WORK / job["toml"]).unlink(missing_ok=True)
    return HTMLResponse(job_html(jid))


async def load_job(request: Request) -> HTMLResponse:
    job = jobs.get(request.path_params["jid"])
    if job is None:
        return HTMLResponse(
            f'<div id="settings">{error_html("render not found, was the server restarted?")}</div>'
        )
    extra = oob(presets_html(job["presets"]), "presets") + oob(sources_html(job["source"]), "sources")
    return HTMLResponse(settings_html(job["cfg"]) + extra, headers=REFRESH)


async def save_preset(request: Request) -> HTMLResponse:
    form = await request.form()
    chosen = [c for c in form.getlist("preset") if isinstance(c, str)]
    name = re.sub(r"[^a-z0-9-]+", "-", fv(form, "preset_name").lower()).strip("-")
    try:
        if not name:
            raise ValueError("give the preset a name")
        path = PRESETS / f"{name}.toml"
        if path.exists():
            raise ValueError(f"preset {name} already exists")
        path.write_text(f"# Saved from the web UI\n{config.to_toml(config_from_form(form))}")
    except FORM_ERRORS as e:
        return HTMLResponse(presets_html(chosen) + error_html(e))
    return HTMLResponse(presets_html([name]))


async def export(request: Request) -> Response:
    try:
        toml = config.to_toml(config_from_form(request.query_params))
    except FORM_ERRORS as e:
        return Response(str(e), status_code=400)
    return Response(
        toml,
        media_type="application/toml",
        headers={"Content-Disposition": 'attachment; filename="cyberglitch.toml"'},
    )


def create_app() -> Starlette:
    WORK.mkdir(parents=True, exist_ok=True)
    return Starlette(
        routes=[
            Route("/", index),
            Route("/fields", fields_for_presets, methods=["POST"]),
            Route("/preview", preview, methods=["POST"]),
            Route("/upload", upload, methods=["POST"]),
            Route("/render", render, methods=["POST"]),
            Route("/jobs/{jid}", job_status),
            Route("/jobs/{jid}/load", load_job, methods=["POST"]),
            Route("/jobs/{jid}/stop", stop_job, methods=["POST"]),
            Route("/save", save_preset, methods=["POST"]),
            Route("/export", export),
            Mount("/static", StaticFiles(directory=STATIC), name="static"),
            Mount("/files", StaticFiles(directory=WORK), name="files"),
        ]
    )


def main() -> None:
    ap = argparse.ArgumentParser(prog="cyberglitch-web")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    import uvicorn

    uvicorn.run(create_app(), host=args.host, port=args.port)
