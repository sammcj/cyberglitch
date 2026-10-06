.PHONY: install lint test check render preview presets web

IN ?= samples/waterlily.ogv
OUT ?= out/render.mp4
PRESET ?=
ARGS ?=
START ?= 0
PRESET_FLAG = $(foreach p,$(PRESET),--preset $(p))

install:
	uv sync --extra matte --extra web

lint:
	uv run ruff check .
	uv run ruff format --check .

test:
	uv run pytest -q

check: lint test

render:
	@mkdir -p "$(dir $(OUT))"
	uv run cyberglitch "$(IN)" "$(OUT)" $(PRESET_FLAG) $(ARGS)

# One frame per preset + contact sheet, e.g. make presets IN=clip.mp4 ARGS="-p presets/any-background.toml"
presets:
	bash scripts/preview-presets.sh "$(IN)" "$(START)" $(ARGS)

# Single frame for fast tweaking, e.g. make preview ARGS="--start 4 -s glow.intensity=2"
preview:
	@mkdir -p out
	uv run cyberglitch "$(IN)" out/preview.png $(PRESET_FLAG) $(ARGS)

# Local web UI on http://127.0.0.1:8000
web:
	uv run cyberglitch-web
