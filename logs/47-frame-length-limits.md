# Frame-length limits: >39 frames produces corrupt output

**Date:** 2026-08-21
**Trigger:** owner asked for a 5-second version of the reference fox clip.

## Valid frame counts

`h3_align_frame_count` (`h3_host.c:15`) snaps to **5 + 17k**, and `h3.c:514` bounds
the aligned value to 5..362. So the ladder is 5, 22, 39, 56, 73, 90, 107, 124, ...
At `H3_FPS` 24 (`h3_host.h:9`), 5 seconds wants 120 -> aligns up to **124**.

## Results

| frames | duration | verdict | total wall | denoise encode | mean YAVG |
|---|---|---|---|---|---|
| 22 | 0.92 s | **clean** (reference) | 4.0 min | 0.245 s | 113.9 |
| 39 | 1.63 s | **clean** | 5.5 min | 0.241 s | 107.6 |
| 107 | 4.46 s | **CORRUPT** — flat brown mush, faint tile seams | 13.2 min | 0.283 s | 64.5 |
| 124 | 5.17 s | **CORRUPT** — dark noise, tile artifacts | 39.2 min | 1582.687 s | 36.6 |

Breaking point is between **39 and 107**. Untested: 56, 73, 90.

## Two findings that matter more than the limit itself

**1. Timing health does not predict output correctness.** The 107-frame run was clean
on every counter — encode 0.283 s, denoise 632 s against ~620 s predicted by linear
scaling from 39 frames, stream 1.486 GiB/s fully hidden (0.003 s unhidden), peak
2.289 GiB, exit 0, valid MP4 with a 141-frame AAC track, ffprobe clean — and its
pixels are mush. So the 124-frame encode explosion (1582.687 s, 7.9 s of CPU per
block) is a SEPARATE symptom, not the cause of corruption. Corruption happens
silently on a run that looks perfect.

**2. The "decoder chunk" theory is dead.** `h3.c:861` says generation "requires at
least one trained 22-frame decoder chunk", which invites the guess that frame counts
must be multiples of 22. But 39 is not a multiple of 22 and is clean, while 107 is
not a multiple of 22 and is corrupt. Divisibility by 22 explains nothing here.

## Verification method — do not skip this

Every corrupt file **exited 0, was a valid h264+AAC MP4, and passed ffprobe.** This is
the constraint-#2 failure mode (a valid MP4 containing a flat grey frame) recurring in
a new form. `mean YAVG` via ffprobe's signalstats filter separates them cheaply:

    ffprobe -v error -f lavfi "movie=<file>,signalstats" \
      -show_entries frame_tags=lavfi.signalstats.YAVG -of csv=p=0

Healthy sits at 107-114, corrupt at 36-65. But YAVG is a screen, not a verdict —
107's uniform ~65 could plausibly have been a darker scene. **Extract a frame grid and
look at it.** A tile montage is one ffmpeg call:

    ffmpeg -i <file> -vf "select='not(mod(n\,18))',scale=304:176,tile=3x2" -frames:v 1 grid.png

## Not investigated

Why corruption starts between 39 and 107 frames, and why 124 additionally triggers the
encode blowup. Owner had paused optimization work before this came up; these runs were
for the 5-second deliverable, not for perf.
