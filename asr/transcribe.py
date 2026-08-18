"""Transcribe Nepali audio with a released checkpoint.

Usage:
    python transcribe.py --nemo nepali-conformer-offline.nemo audio1.wav [audio2.wav ...]

Outputs one line per file: <path>\t<transcript>. Add --normalize to apply the same text
normalization used for every number in RESULTS.md (digits -> spoken Nepali words, punctuation
stripped) — do this before computing WER against NepTel references.
"""

import argparse
import logging
import sys

logging.basicConfig(level=logging.WARNING)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nemo", required=True, help="path to a .nemo checkpoint")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--threads", type=int, default=4, help="CPU threads")
    ap.add_argument("--normalize", action="store_true")
    ap.add_argument("audio", nargs="+")
    a = ap.parse_args()

    import torch
    from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel

    torch.set_num_threads(a.threads)
    model = EncDecHybridRNNTCTCBPEModel.restore_from(
        a.nemo, map_location="cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    outs = model.transcribe(a.audio, batch_size=a.batch, verbose=False)
    if a.normalize:
        from nepali_normalize import normalize
    for path, o in zip(a.audio, outs):
        text = o.text if hasattr(o, "text") else str(o)
        text = " ".join(t for t in text.split() if t != "<breath>")
        if a.normalize:
            text = normalize(text)
        print(f"{path}\t{text}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
