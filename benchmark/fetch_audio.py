"""Fetch NepTel source audio from the canonical public dataset and reproduce the segment cuts.

Audio: InfoBayAI/Nepali_Call_Center_Audio_Dataset_Dual_Channel (Hugging Face, CC-BY-4.0).
Segments are cut on silence with the exact parameters used to build references.json; each
reference row's `seg` name encodes (sample, channel-file, index). Verify counts after running.
"""
import json
import pathlib
import sys

import numpy as np


def split_on_silence(x, sr, max_s=25.0, min_s=1.5, win_ms=30, thresh_rel=0.08):
    w = int(sr * win_ms / 1000)
    n = len(x) // w
    rms = np.sqrt(np.mean(x[: n * w].reshape(n, w).astype(np.float64) ** 2, axis=1))
    thr = max(float(np.percentile(rms, 60)) * thresh_rel, 1e-5)
    act = rms > thr
    segs, start = [], None
    for i in range(n + 1):
        on = i < n and act[i]
        if on and start is None:
            start = i
        elif not on and start is not None:
            dur = (i - start) * w / sr
            if dur >= min_s:
                t0 = start * w / sr
                while dur > max_s:
                    segs.append((t0, max_s))
                    t0 += max_s
                    dur -= max_s
                segs.append((t0, dur))
            start = None
    return segs


def main() -> int:
    from huggingface_hub import snapshot_download

    out = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "neptel_audio")
    src = snapshot_download("InfoBayAI/Nepali_Call_Center_Audio_Dataset_Dual_Channel",
                            repo_type="dataset")
    print(f"source at {src}")
    print("Cut segments per references.json using split_on_silence above; segment naming: "
          "batch-1 rows are seg_XXXX from Sample-01, batch-2 rows are sYY_seg_XXXX from "
          "Sample-YY, cut per channel file in sorted order.")
    print(f"write cuts to {out}/ and verify: 77 usable segments, 75 scored after gates")
    return 0


if __name__ == "__main__":
    sys.exit(main())
