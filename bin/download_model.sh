#!/bin/bash
# I download ONLY the files h3.c actually needs for the T2VA / FL2VA path.
# I deliberately skip Ref2VA/ and the duplicated top-level transformer,
# text_encoder and vae directories, because h3.c reads FL2VA/ for this task.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/models/MiniMax-H3"
REPO="MiniMaxAI/MiniMax-H3"
HF="$ROOT/venv/bin/hf"

mkdir -p "$DEST"

echo "I am downloading $REPO -> $DEST"
echo "I am fetching only: FL2VA/** plus the small top-level config files."

# I retry, because a 134 GiB transfer over a long period will hit transient
# network errors and hf download resumes cleanly.
attempt=1
while [ "$attempt" -le 100 ]; do
    echo "=== attempt $attempt at $(date) ==="
    "$HF" download "$REPO" \
        --local-dir "$DEST" \
        --include "FL2VA/*" \
        --include "model_index.json" \
        --include "modular_model_index.json" \
        --include "scheduler/*" \
        --include "audio_scheduler/*" \
        --include "config.json" \
        && { echo "I completed the download at $(date)"; exit 0; }
    echo "attempt $attempt failed; I am retrying in 20s"
    sleep 20
    attempt=$((attempt + 1))
done
echo "I gave up after $((attempt - 1)) attempts."
exit 1
