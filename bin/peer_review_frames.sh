#!/bin/bash
# I fan the frame-length corruption brief out to three models over OpenRouter.
#
# This is a curl/jq rewrite of bin/peer_review.sh. The original drives the API
# through python3, and python3 on this box is an xcrun shim that currently
# refuses to run: Xcode was installed but its licence was never accepted, so
# every xcrun-shimmed tool (clang, make, python3) fails until someone runs
# `sudo xcodebuild -license accept` in a real terminal. curl and jq are real
# binaries and are unaffected.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROMPT_FILE="${1:-$ROOT/logs/peer-review-prompt-frames.md}"
OUT="$ROOT/logs/review-frames"
mkdir -p "$OUT"

if [ -z "${OPENROUTER_API_KEY:-}" ]; then
    echo "peer_review_frames: OPENROUTER_API_KEY is not set in this shell." >&2
    echo "  It lives in ~/.zshrc; start a login shell or export it." >&2
    exit 1
fi
[ -r "$PROMPT_FILE" ] || { echo "cannot read $PROMPT_FILE" >&2; exit 1; }

SYSTEM='You are a senior engineer with deep expertise in diffusion video models,
Metal/MPSGraph on AMD RDNA2 (gfx1030), and numerical debugging of transformer
inference in C. The user is debugging CORRECTNESS, not performance. Give ranked,
concrete, falsifiable hypotheses, and for each one state the cheapest decisive
check. Prioritise checks that read weights or dump intermediates over checks that
require regenerating video (each generation costs 5-40 minutes). Correct the user
where their reasoning is wrong, and say plainly when you are uncertain. No filler,
no restating the brief.'

ask() {
    local model="$1" slug="$2"
    jq -n --arg model "$model" --arg sys "$SYSTEM" --rawfile prompt "$PROMPT_FILE" \
      '{model:$model, max_tokens:8000,
        messages:[{role:"system",content:$sys},{role:"user",content:$prompt}]}' \
      > "$OUT/$slug.req.json"
    curl -sS --max-time 1800 \
        -H "Authorization: Bearer $OPENROUTER_API_KEY" \
        -H "Content-Type: application/json" \
        -d @"$OUT/$slug.req.json" \
        https://openrouter.ai/api/v1/chat/completions \
        > "$OUT/$slug.json" 2> "$OUT/$slug.err"
    # Surface the prose, and the usage block so the spend is recorded honestly.
    jq -r '.choices[0].message.content // ("ERROR: " + (.error.message // tostring))' \
        "$OUT/$slug.json" > "$OUT/$slug.md" 2>/dev/null
    printf "done: %-12s %6s chars  usage=%s\n" "$slug" \
        "$(wc -c < "$OUT/$slug.md" | tr -d ' ')" \
        "$(jq -c '.usage // "none"' "$OUT/$slug.json" 2>/dev/null)"
}

ask "moonshotai/kimi-k3"            kimi-k3   &
ask "openai/gpt-5"                  gpt-5     &
ask "google/gemini-3.1-pro-preview" gemini-31 &
wait
echo "all three returned -> $OUT"
