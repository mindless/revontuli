#!/bin/bash
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; OUT="$ROOT/logs/review"; mkdir -p "$OUT"
ask() {
  local model="$1" slug="$2"
  python3 - "$model" "$ROOT/logs/peer-review-prompt.md" > "$OUT/$slug.json" 2>"$OUT/$slug.err" <<'PY'
import json, os, sys, urllib.request
model, prompt_path = sys.argv[1], sys.argv[2]
body = json.dumps({
    "model": model,
    "messages": [
        {"role": "system", "content":
         "You are a senior GPU performance engineer expert in Metal, MSL, MPSGraph, "
         "RDNA2/gfx1030 microarchitecture and PCIe/Thunderbolt transfer behaviour. "
         "Answer with a RANKED list of concrete optimizations. Keep internal "
         "deliberation minimal and spend your output budget on the final answer. "
         "Be quantitative. Correct the user where wrong."},
        {"role": "user", "content": open(prompt_path).read()},
    ],
    "max_tokens": 25000,
    "reasoning": {"effort": "low"},
}).encode()
req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=body,
    headers={"Authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"],
             "Content-Type": "application/json"})
try:
    with urllib.request.urlopen(req, timeout=2400) as r: print(r.read().decode())
except urllib.error.HTTPError as e:
    print(json.dumps({"error": e.code, "body": e.read().decode()[:2000]}))
PY
  echo "done: $slug"
}
ask "moonshotai/kimi-k3" kimi-k3-v2 &
ask "openai/gpt-5"       gpt-5-v2   &
wait; echo "both returned"
