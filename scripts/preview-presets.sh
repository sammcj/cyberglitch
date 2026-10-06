#!/usr/bin/env bash
# Render one small frame per preset plus a contact sheet: out/presets/sheet.jpg
# usage: scripts/preview-presets.sh <input> [start-seconds] [extra cyberglitch args...]
set -euo pipefail

input=${1:?usage: preview-presets.sh <input> [start] [args...]}
start=${2:-0}
shift $(( $# >= 2 ? 2 : 1 ))
dir=out/presets
mkdir -p "$dir"

small=(-s output.width=960 -s output.height=540 -s output.pixel=2 --start "$start")

uv run cyberglitch "$input" "$dir/0-default.png" "${small[@]}" "$@"
for preset in presets/*.toml; do
  name=$(basename "$preset" .toml)
  [ "$name" = any-background ] && continue
  uv run cyberglitch "$input" "$dir/$name.png" -p "$preset" "${small[@]}" "$@"
done

ffmpeg -v error -y -pattern_type glob -i "$dir/*.png" -vf "scale=480:-1,tile=4x2" -frames:v 1 "$dir/sheet.jpg"
echo "$dir/sheet.jpg"
