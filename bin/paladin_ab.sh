#!/bin/bash
# A/B roll of the game's paladin-idle shape (Run A / Run B configs from
# PIPELINE-TARGETS.md) against the optimized build.
# usage: paladin_ab.sh <frames> <outname> [FP16=1]
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FRAMES="${1:?frames}"
NAME="${2:?outname}"
FP16="${3:-1}"
ANCHOR="$ROOT/anchors-pipeline-paladin.png"
OUT="$ROOT/outputs/$NAME.mp4"
FRAMESDIR="$ROOT/outputs/$NAME-frames"
mkdir -p "$FRAMESDIR"
PROMPT='The character from the image holds EXACTLY the pose, stance, camera angle and attitude it has in the image — if the stance is battle-ready, coiled or menacing, it STAYS battle-ready, coiled and menacing for the whole video. A living combat idle inside that stance: the chest rises and falls with breathing, fur, mane, hair and cloth sway, claws or weapon-hand flex slightly, the tail moves if there is one, weight shifts subtly. Do not relax the pose, do not look around, do not re-pose or rotate the character. The feet NEVER move and stay planted on the same spot for the entire video. The camera is completely locked: no zoom, no pan, no cuts. The character stays centered and fully inside the frame. The background is a flat, solid, uniform bright green (#00FF00) with no texture, no shadow, no floor line. The art style, outfit, colors, proportions and weapon of the character remain exactly as in the image in every frame. No other objects or characters.'
cd "$ROOT/src/h3.c"
ENVVARS=(H3_METAL_DEVICE_NAME="Radeon RX 6900 XT" H3_DEBUG_STREAM_SPLIT=1)
if [ "$FP16" = "1" ]; then ENVVARS+=(H3_FP16_GEMM=1); fi
exec env "${ENVVARS[@]}" /usr/bin/time -l ./h3 \
  -d "$ROOT/models/MiniMax-H3" -p "$PROMPT" -o "$OUT" \
  --width 640 --height 832 --frames "$FRAMES" --steps 4 --seed 42 \
  --layers 50 --reuse 1 --ssd-streaming \
  --first-frame "$ANCHOR" --last-frame "$ANCHOR" \
  --frames-dir "$FRAMESDIR" --profile
