#!/bin/bash
# I fan the same review brief out to three models in parallel over OpenRouter.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROMPT_FILE="$ROOT/logs/peer-review-prompt.md"
OUT="$ROOT/logs/review"
mkdir -p "$OUT"

ask() {
    local model="$1" slug="$2"
    python3 - "$model" "$PROMPT_FILE" > "$OUT/$slug.json" 2>"$OUT/$slug.err" <<'PY'
import json, os, sys, urllib.request
model, prompt_path = sys.argv[1], sys.argv[2]
body = json.dumps({
    "model": model,
    "messages": [
        {"role": "system", "content":
         "You are a senior GPU performance engineer. You know Metal, MSL, "
         "MPSGraph, RDNA2/gfx1030 microarchitecture, and PCIe/Thunderbolt "
         "transfer behaviour. Give ranked, concrete, implementable advice with "
         "quantitative reasoning. Correct the user where they are wrong. No "
         "filler, no restating the brief back."},
        {"role": "user", "content": open(prompt_path).read()},
    ],
    "max_tokens": 8000,
}).encode()
req = urllib.request.Request(
    "https://openrouter.ai/api/v1/chat/completions", data=body,
    headers={"Authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"],
             "Content-Type": "application/json"})
try:
    with urllib.request.urlopen(req, timeout=1800) as r:
        print(r.read().decode())
except urllib.error.HTTPError as e:
    print(json.dumps({"error": e.code, "body": e.read().decode()[:2000]}))
PY
    echo "done: $slug"
}

ask "moonshotai/kimi-k3"             kimi-k3   &
ask "openai/gpt-5"                   gpt-5     &
ask "google/gemini-3.1-pro-preview"  gemini-31 &
wait
echo "all three returned"
