#!/bin/bash
# trace-denoise.sh -- Metal System Trace over an H3 generation, to attribute the
# denoise phase's command_wait time to actual GPU work.
#
# WHY THIS EXISTS
#   The denoise wall is fully accounted for (logs/42-reconciliation.md):
#     wall = encode + command_wait + stream_unhidden_wait + ~0.019s
#   and command_wait owns 140.7s of the 141.0s. But command_wait is TURNAROUND
#   (commit -> completed), and root-gpu -- the only GPU-side counter this repo
#   has -- sees just 10.1s of it, because gpu_seconds sums root
#   MTLCommandBuffer timestamps and MPSGraph schedules child buffers it never
#   times. So real GPU busy time is bounded [10.1s, 140.7s] and nothing in-tree
#   can narrow it.
#
#   Metal System Trace sees child command buffers. That is the whole point of
#   using it here.
#
#   Do NOT quote root-gpu as a total. That mistake retired the matmul rewrite on
#   a sum of floors (17s of 240s = "7%") when command_wait puts real turnaround
#   at 162s of 240s = 67%.
#
# Usage:
#   ./bin/trace-denoise.sh [OUTPUT_TRACE]
#
# Environment overrides:
#   H3_TRACE_LIMIT   xctrace time limit           (default 300s)
#   H3_TRACE_ARGS    args passed to generate.sh   (default: the reference clip)
#
# NOTE: the export/parse step below is written against xctrace's documented
# interface but has NOT been executed on this box -- Xcode was not installed
# when this was written. Treat the recording step as the load-bearing part and
# verify the table names against the TOC this script dumps.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="${1:-$ROOT/logs/trace-denoise-$STAMP.trace}"
LIMIT="${H3_TRACE_LIMIT:-300}"
LOG="$ROOT/logs/trace-denoise-$STAMP.log"

if ! xcrun -f xctrace >/dev/null 2>&1; then
    echo "trace-denoise: xctrace is not available." >&2
    echo "  Command Line Tools do not ship it; a full Xcode is required." >&2
    echo "  Install Xcode, then:" >&2
    echo "    sudo xcode-select -s /Applications/Xcode.app/Contents/Developer" >&2
    exit 1
fi

if ! xcrun xctrace list templates 2>/dev/null | grep -q "Metal System Trace"; then
    echo "trace-denoise: the 'Metal System Trace' template is not present." >&2
    echo "  Templates found:" >&2
    xcrun xctrace list templates >&2
    exit 1
fi

# I profile the shipping configuration, because the regime matters: the stream is
# hidden only when compute is long. In the gate-skip smoke config compute is
# 12.9s against a 58.1s stream and 44.2s of stream IS exposed, which is a
# different bottleneck and not what we are measuring.
set -- ${H3_TRACE_ARGS:-"A cinematic fox walking through snow" 608 352 22 4 trace-fox.mp4 42}

echo "trace-denoise: launching generation, logging to $LOG"
( cd "$ROOT" && H3_PROFILE=1 H3_DEBUG_STREAM_SPLIT=1 \
    ./generate.sh "$@" ) >"$LOG" 2>&1 &
GEN_PID=$!

# I attach rather than --launch because generate.sh resolves paths and runs h3
# from its own source directory (the shader path is relative and hardcoded
# upstream). Tracing the wrapper would trace the shell, not the GPU process.
# Attaching a few seconds late is harmless: the denoise phase does not begin
# until the text encoder and DiT load are done, ~81s in.
H3_PID=""
for _ in $(seq 1 120); do
    H3_PID="$(pgrep -n -x h3 2>/dev/null)"
    [ -n "$H3_PID" ] && break
    if ! kill -0 "$GEN_PID" 2>/dev/null; then
        echo "trace-denoise: generation exited before h3 started; see $LOG" >&2
        exit 1
    fi
    sleep 1
done

if [ -z "$H3_PID" ]; then
    echo "trace-denoise: never saw an h3 process; see $LOG" >&2
    kill "$GEN_PID" 2>/dev/null
    exit 1
fi

echo "trace-denoise: attaching to h3 pid $H3_PID, limit ${LIMIT}s"
xcrun xctrace record \
    --template "Metal System Trace" \
    --attach "$H3_PID" \
    --time-limit "${LIMIT}s" \
    --output "$OUT"
RC=$?

wait "$GEN_PID" 2>/dev/null

if [ $RC -ne 0 ]; then
    echo "trace-denoise: xctrace exited $RC" >&2
    exit $RC
fi

echo
echo "trace-denoise: trace written to $OUT"
echo "trace-denoise: profile lines from this run --"
grep -E "h3 profile|SSD stream|h3upload" "$LOG" || true

echo
echo "trace-denoise: table of contents (verify table names before parsing) --"
xcrun xctrace export --input "$OUT" --toc

cat <<'NEXT'

NEXT STEPS
  1. Read the TOC above and pick the GPU-activity table. Export it, e.g.:
       xcrun xctrace export --input <trace> \
         --xpath '/trace-toc/run[@number="1"]/data/table[@schema="metal-gpu-intervals"]'
     Schema names vary by Xcode version -- trust the TOC, not this example.
  2. Sum GPU interval durations that fall inside the denoise window. Bound the
     window using the log's own phase marks: the "H3 DiT / Euler denoise"
     profile line reports the phase wall, and it is emitted at its end.
  3. Compare that sum against BOTH bounds for the same run:
       lower  root-gpu       (~10.1s)
       upper  command_wait   (~140.7s)
     Where it lands is the answer:
       near the upper bound -> the denoise is GPU-compute bound, and the
         matmul/flash-attention work retired on the "7%" figure should be
         reopened (remaining task 11).
       near the lower bound -> the GPU is idle inside command_wait, and the
         real target is submission/queue structure, not kernels.
  4. Whatever it says, do not project a win before the terms sum. That rule has
     already been violated twice on this workload.
NEXT
