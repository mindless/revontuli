#!/bin/bash
# Fan the GEMM/pipelining optimization plan out to three models over OpenRouter.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/logs/review-gemm"
mkdir -p "$OUT"
ask() {
  local model="$1" slug="$2"
  python3 - "$model" "$ROOT/logs/peer-review-prompt-gemm.md" > "$OUT/$slug.json" 2>"$OUT/$slug.err" <<'PY'
import json, os, sys, urllib.request
model, prompt_path = sys.argv[1], sys.argv[2]
body = json.dumps({
    "model": model,
    "messages": [
        {"role": "system", "content":
         "You are a senior GPU performance engineer expert in Metal, MSL, MPSGraph, "
         "RDNA2/gfx1030 microarchitecture, Thunderbolt eGPU behaviour, and "
         "diffusion-model numerics. Answer the numbered questions directly, ranked "
         "and quantitative. Keep internal deliberation minimal and spend the output "
         "budget on the final answer. Correct the user where wrong."},
        {"role": "user", "content": open(prompt_path).read()},
    ],
    "max_tokens": 20000,
    "reasoning": {"effort": "medium"},
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
ask "moonshotai/kimi-k3"            kimi-k3   &
ask "openai/gpt-5"                  gpt-5     &
ask "google/gemini-3.1-pro-preview" gemini-31 &
wait
echo "all returned"
