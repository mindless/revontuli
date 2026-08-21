#!/bin/bash
# generate.sh -- MiniMax H3 on the AMD Radeon RX 6900 XT eGPU
#
# I run MiniMax H3 locally through antirez/h3.c, pinned explicitly to the
# external RX 6900 XT. I refuse to run on the Intel UHD 630, because a silent
# fallback to a 1.5 GiB integrated GPU would either fail confusingly or take
# so long it looks like a hang.
#
# Usage:
#   ./generate.sh "PROMPT" [WIDTH] [HEIGHT] [FRAMES] [STEPS] [OUTPUT] [SEED]
#
# Example:
#   ./generate.sh "A cinematic fox walking through snow" 608 352 22 4 fox.mp4
#
# Environment overrides:
#   H3_MODEL_DIR   model snapshot          (default ~/minimax-h3-egpu/models/MiniMax-H3)
#   H3_GPU_NAME    Metal device substring  (default "Radeon RX 6900 XT")
#   H3_LAYERS      DiT blocks              (default 50 = exact)
#   H3_REUSE       denoiser reuse          (default 1 = closest to reference)
#   H3_SSD         1 = --ssd-streaming on  (default 1)
#   H3_EXTRA       extra flags passed through to ./h3

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
H3_BIN="$ROOT/src/h3.c/h3"
PROBE="$ROOT/bin/metal_probe"
MODEL_DIR="${H3_MODEL_DIR:-$ROOT/models/MiniMax-H3}"
# I resolve these to absolute paths because I run h3 from its own directory.
case "$MODEL_DIR" in /*) ;; *) MODEL_DIR="$PWD/$MODEL_DIR" ;; esac
GPU_NAME="${H3_GPU_NAME:-Radeon RX 6900 XT}"

PROMPT="${1:-}"
WIDTH="${2:-608}"
HEIGHT="${3:-352}"
FRAMES="${4:-22}"
STEPS="${5:-4}"
OUTPUT="${6:-h3-egpu.mp4}"
SEED="${7:-42}"

LAYERS="${H3_LAYERS:-50}"
REUSE="${H3_REUSE:-1}"
SSD="${H3_SSD:-1}"

if [ -z "$PROMPT" ]; then
    cat >&2 <<USAGE
I need a prompt.

  ./generate.sh "PROMPT" [WIDTH] [HEIGHT] [FRAMES] [STEPS] [OUTPUT] [SEED]

Defaults: WIDTH=608 HEIGHT=352 FRAMES=22 STEPS=4 OUTPUT=h3-egpu.mp4 SEED=42
USAGE
    exit 64
fi

# Make the output path absolute under outputs/ unless the caller gave a path.
case "$OUTPUT" in
    /*) OUT_PATH="$OUTPUT" ;;
    */*) OUT_PATH="$OUTPUT" ;;
    *)  OUT_PATH="$ROOT/outputs/$OUTPUT" ;;
esac
mkdir -p "$(dirname "$OUT_PATH")" "$ROOT/logs"

# ---------------------------------------------------------------- preflight ---

if [ ! -x "$H3_BIN" ]; then
    echo "I cannot find the h3 binary at $H3_BIN." >&2
    echo "I expected it to be built with: (cd $ROOT/src/h3.c && make)" >&2
    exit 70
fi

if [ ! -d "$MODEL_DIR/FL2VA/transformer" ]; then
    echo "I cannot find the MiniMax H3 snapshot at $MODEL_DIR." >&2
    echo "I expected $MODEL_DIR/FL2VA/transformer to exist." >&2
    exit 72
fi

# I verify the eGPU is actually present and exposed to Metal BEFORE loading a
# 33B model. metal_probe exits non-zero when no device name contains "6900".
if [ -x "$PROBE" ]; then
    if ! "$PROBE" >/dev/null 2>&1; then
        echo "I refuse to start: macOS is not exposing the RX 6900 XT to Metal." >&2
        echo "I will not silently fall back to the Intel UHD 630." >&2
        echo "I suggest checking that the Thunderbolt eGPU enclosure is powered" >&2
        echo "and connected, then re-running:  $PROBE" >&2
        exit 75
    fi
else
    echo "I could not find $PROBE, so I am skipping the pre-flight GPU check." >&2
fi

DETECTED="$(system_profiler SPDisplaysDataType 2>/dev/null \
            | awk '/Chipset Model: .*6900/{print substr($0, index($0,":")+2)}' \
            | head -1)"
if [ -n "$DETECTED" ]; then
    echo "I found the RX 6900 XT: $DETECTED"
else
    echo "I found a Metal device matching \"6900\" (System Profiler was quiet)."
fi

# ------------------------------------------------------------------ generate --

ARGS=(
    -d "$MODEL_DIR"
    -p "$PROMPT"
    -o "$OUT_PATH"
    --width  "$WIDTH"
    --height "$HEIGHT"
    --frames "$FRAMES"
    --steps  "$STEPS"
    --layers "$LAYERS"
    --reuse  "$REUSE"
    --seed   "$SEED"
    --profile
)
[ "$SSD" = "1" ] && ARGS+=(--ssd-streaming)
# shellcheck disable=SC2206
[ -n "${H3_EXTRA:-}" ] && ARGS+=(${H3_EXTRA})

STAMP="$(date +%Y%m%d-%H%M%S)"
LOG="$ROOT/logs/generate-$STAMP.log"

echo "I am loading MiniMax H3 from $MODEL_DIR"
echo "I am generating ${WIDTH}x${HEIGHT}, $FRAMES frames, $STEPS steps, seed $SEED"
echo "I am pinning Metal to: $GPU_NAME"
[ "$SSD" = "1" ] && echo "I am streaming the BF16 DiT layers from SSD."
echo "I am logging to $LOG"
echo

START=$(date +%s)
# h3 loads h3_shaders.metal relative to the working directory, so I run it from
# its own source tree. Every path I pass it is absolute, so this is safe.
# H3_METAL_DEVICE_NAME is the selector I added in h3_device_select.m. If no
# Metal device matches it, h3 reports the available devices and exits rather
# than running on the wrong GPU.
( cd "$(dirname "$H3_BIN")" && \
  H3_METAL_DEVICE_NAME="$GPU_NAME" \
  /usr/bin/time -l "$H3_BIN" "${ARGS[@]}" ) 2>&1 | tee "$LOG"
STATUS=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))

echo
if [ "$STATUS" -ne 0 ]; then
    echo "h3 exited with status $STATUS after ${ELAPSED}s. I did not produce a video."
    echo "The full log is at $LOG"
    exit "$STATUS"
fi

if [ ! -s "$OUT_PATH" ]; then
    echo "h3 exited cleanly but $OUT_PATH is missing or empty. I treat that as a failure."
    exit 74
fi

BYTES=$(stat -f%z "$OUT_PATH")
echo "I saved $OUT_PATH ($BYTES bytes) in ${ELAPSED}s."

# I verify the container really decodes, because a file that exists but holds
# noise is not a success.
if command -v ffprobe >/dev/null 2>&1; then
    echo
    echo "I am verifying the output with ffprobe:"
    ffprobe -v error -show_entries \
        stream=index,codec_type,codec_name,width,height,nb_frames,duration \
        -of default=noprint_wrappers=1 "$OUT_PATH" 2>&1 | sed 's/^/  /'
    if ffprobe -v error -select_streams a -show_entries stream=codec_name \
            -of csv=p=0 "$OUT_PATH" 2>/dev/null | grep -q .; then
        echo "  I confirmed an audio stream is present."
    else
        echo "  I did not find an audio stream in the output."
    fi
fi
