#!/bin/bash
# Quality screen for an optimized roll against a reference roll of the same
# seed/config: YAVG luma screen, PSNR/SSIM between the two clips, and a
# 6-frame visual grid for the eyeball pass.
# usage: compare_clips.sh <candidate.mp4> <reference.mp4> <gridname>
set -uo pipefail
CAND="${1:?candidate}"
REF="${2:?reference}"
GRID="${3:-grid}"
yavg() {
  ffprobe -v error -f lavfi "movie=$1,signalstats" \
    -show_entries frame_tags=lavfi.signalstats.YAVG -of csv=p=0 2>/dev/null |
    awk '{s+=$1;n++} END {if (n) printf "%.1f (%d frames)\n", s/n, n}'
}
echo "YAVG candidate: $(yavg "$CAND")"
echo "YAVG reference: $(yavg "$REF")"
echo "-- PSNR (candidate vs reference) --"
ffmpeg -v error -i "$CAND" -i "$REF" -lavfi psnr -f null - 2>&1 | grep -o "average:[0-9.inf]*"
echo "-- SSIM --"
ffmpeg -v error -i "$CAND" -i "$REF" -lavfi ssim -f null - 2>&1 | grep -o "All:[0-9.]*"
FRAMES=$(ffprobe -v error -select_streams v -count_frames \
  -show_entries stream=nb_read_frames -of csv=p=0 "$CAND" 2>/dev/null)
STEP=$(( (FRAMES > 6 ? FRAMES : 6) / 6 ))
ffmpeg -v error -y -i "$CAND" -vf "select='not(mod(n\,$STEP))',scale=320:416,tile=3x2" \
  -frames:v 1 "outputs/$GRID-candidate.png"
ffmpeg -v error -y -i "$REF" -vf "select='not(mod(n\,$STEP))',scale=320:416,tile=3x2" \
  -frames:v 1 "outputs/$GRID-reference.png"
echo "grids: outputs/$GRID-candidate.png outputs/$GRID-reference.png"
