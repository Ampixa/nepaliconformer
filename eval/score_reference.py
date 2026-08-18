"""Score a model (or a hypothesis file) against the NepTel reference set.

This is the exact instrument behind every WER in RESULTS.md: word-level Levenshtein on
normalized text (digits -> spoken Nepali words on BOTH sides), with substitutions, deletions and
insertions reported separately, and a speaking-rate gate that drops any reference whose implied
rate exceeds 6 words/second (no human speaks that fast; such rows are transcription-engine
hallucinations, and one of them once contributed 32% of all reference words).

Two modes:
    --nemo model.nemo         decode with a NeMo hybrid checkpoint, then score
    --hyp  hypotheses.json    score precomputed outputs: [{"seg": ..., "text": ...}, ...]

Paired bootstrap (over segments) against a second hypothesis file:
    --compare other_hypotheses.json
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "asr"))
from nepali_normalize import normalize  # noqa: E402

MAX_WORDS_PER_SEC = 6.0


def align(ref, hyp):
    r, h = ref.split(), hyp.split()
    R, H = len(r), len(h)
    d = np.zeros((R + 1, H + 1), dtype=np.int32)
    d[:, 0] = np.arange(R + 1)
    d[0, :] = np.arange(H + 1)
    for i in range(1, R + 1):
        for j in range(1, H + 1):
            d[i, j] = min(d[i-1, j] + 1, d[i, j-1] + 1, d[i-1, j-1] + (r[i-1] != h[j-1]))
    i, j = R, H
    S = D = I = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i, j] == d[i-1, j-1] + (r[i-1] != h[j-1]):
            S += r[i-1] != h[j-1]
            i -= 1
            j -= 1
        elif i > 0 and d[i, j] == d[i-1, j] + 1:
            D += 1
            i -= 1
        else:
            I += 1
            j -= 1
    return S, D, I, R


def load_refs(manifest):
    man = json.load(open(manifest, encoding="utf-8"))
    segs = []
    for m in man["segments"]:
        if not m.get("reference") and not m.get("chirp2"):
            continue
        text = m.get("reference") or m["chirp2"]
        rate = len(text.split()) / max(float(m.get("dur_s", 0)) or 0.1, 0.1)
        if rate > MAX_WORDS_PER_SEC:
            continue
        segs.append({"seg": m["seg"], "ref": normalize(text)})
    return segs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.path.join(os.path.dirname(__file__),
                                                       "..", "benchmark", "references.json"))
    ap.add_argument("--segdir", help="directory of wavs (required with --nemo)")
    ap.add_argument("--nemo")
    ap.add_argument("--hyp", help="JSON list of {seg, text}")
    ap.add_argument("--compare", help="second hypothesis JSON for a paired bootstrap")
    ap.add_argument("--out")
    a = ap.parse_args()

    segs = load_refs(a.manifest)
    print(f"{len(segs)} reference segments, "
          f"{sum(len(s['ref'].split()) for s in segs)} words")

    if a.nemo:
        if not a.segdir:
            print("--segdir is required with --nemo", file=sys.stderr)
            return 2
        import torch
        from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel
        torch.set_num_threads(4)
        model = EncDecHybridRNNTCTCBPEModel.restore_from(a.nemo, map_location="cpu")
        model.eval()
        outs = model.transcribe([os.path.join(a.segdir, s["seg"]) for s in segs],
                                batch_size=4, verbose=False)
        hyps = {s["seg"]: (o.text if hasattr(o, "text") else str(o)) for s, o in zip(segs, outs)}
    elif a.hyp:
        hyps = {r["seg"]: r["text"] for r in json.load(open(a.hyp, encoding="utf-8"))}
    else:
        print("need --nemo or --hyp", file=sys.stderr)
        return 2

    rows = []
    tS = tD = tI = tN = 0
    for s in segs:
        hyp = normalize(" ".join(t for t in hyps.get(s["seg"], "").split() if t != "<breath>"))
        S, D, I, N = align(s["ref"], hyp)
        tS += S; tD += D; tI += I; tN += N
        rows.append({"seg": s["seg"], "err": S + D + I, "n": N})
    wer = (tS + tD + tI) / max(tN, 1)
    print(f"WER {wer:.4f}   sub {tS/tN:.4f}  del {tD/tN:.4f}  ins {tI/tN:.4f}   ({tN} words)")

    if a.compare:
        other = {r["seg"]: r["text"] for r in json.load(open(a.compare, encoding="utf-8"))}
        rows2 = []
        for s in segs:
            hyp = normalize(" ".join(t for t in other.get(s["seg"], "").split()
                                     if t != "<breath>"))
            S, D, I, N = align(s["ref"], hyp)
            rows2.append({"seg": s["seg"], "err": S + D + I, "n": N})
        e1 = np.array([r["err"] for r in rows]); n1 = np.array([r["n"] for r in rows])
        e2 = np.array([r["err"] for r in rows2])
        rng = np.random.default_rng(7)
        diffs = []
        for _ in range(20000):
            idx = rng.integers(0, len(rows), len(rows))
            diffs.append(e2[idx].sum() / n1[idx].sum() - e1[idx].sum() / n1[idx].sum())
        lo, mid, hi = np.percentile(diffs, [2.5, 50, 97.5])
        print(f"compare - this: {mid:+.4f}  [95% CI {lo:+.4f}, {hi:+.4f}]")

    if a.out:
        json.dump({"wer": round(wer, 4), "sub": round(tS/tN, 4), "del": round(tD/tN, 4),
                   "ins": round(tI/tN, 4), "words": int(tN), "rows": rows},
                  open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
