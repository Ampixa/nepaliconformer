"""Fetch NepTel source audio from the canonical public dataset and reproduce the segment cuts.

Audio: InfoBayAI/Nepali_Call_Center_Audio_Dataset_Dual_Channel (Hugging Face, CC-BY-4.0).
Segments are fixed 25-second windows over each per-speaker channel file, numbered
sequentially across SPEAKER_00 then SPEAKER_01 within a call; a reference row's `seg` name
encodes (sample, index). Every cut is checked against the `dur_s` recorded in
references.json so a misalignment fails loudly instead of silently producing a different
benchmark. Boundary durations can differ by a few tenths of a second from the originals
(the reference cuts were lightly trimmed); that is within tolerance and does not move WER.

    python fetch_audio.py [outdir]        # default: neptel_audio/

The dataset is gated on Hugging Face (auto-approved): accept the terms once at
https://huggingface.co/datasets/InfoBayAI/Nepali_Call_Center_Audio_Dataset_Dual_Channel
and run `hf auth login` first.
"""
import json
import pathlib
import sys

SAMPLE_DIRS = {"01": "Sample - 01", "02": "Sample - 02", "03": "Sample - 03"}


WINDOW_S = 25.0
TOLERANCE_S = 1.0


def cut_sample(sample_dir):
    """Cut one call into fixed 25 s windows, numbered across SPEAKER_00 then SPEAKER_01."""
    import soundfile as sf

    cuts = []
    for wav in sorted(sample_dir.glob("SPEAKER_*.wav")):
        x, sr = sf.read(wav)
        if x.ndim > 1:
            x = x.mean(axis=1)
        step = int(WINDOW_S * sr)
        for a in range(0, len(x), step):
            cuts.append((x[a:a + step], sr))
    return cuts


def seg_index(seg):
    """seg_0007.wav -> ('01', 7);  s02_seg_0013.wav -> ('02', 13)."""
    stem = seg.replace(".wav", "")
    if stem.startswith("s0"):
        sample, rest = stem[1:3], stem.split("seg_")[1]
        return sample, int(rest)
    return "01", int(stem.split("seg_")[1])


def main() -> int:
    import soundfile as sf
    from huggingface_hub import snapshot_download

    out = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "neptel_audio")
    here = pathlib.Path(__file__).parent
    refs = json.load(open(here / "references.json", encoding="utf-8"))["segments"]

    src = pathlib.Path(snapshot_download(
        "InfoBayAI/Nepali_Call_Center_Audio_Dataset_Dual_Channel", repo_type="dataset"))
    print(f"source at {src}")
    out.mkdir(parents=True, exist_ok=True)

    wanted = {}
    for row in refs:
        sample, idx = seg_index(row["seg"])
        wanted.setdefault(sample, {})[idx] = row

    written, mismatched = 0, []
    for sample, rows in sorted(wanted.items()):
        cuts = cut_sample(src / SAMPLE_DIRS[sample])
        print(f"Sample-{sample}: {len(cuts)} cuts, {len(rows)} needed")
        for idx, row in sorted(rows.items()):
            if idx >= len(cuts):
                mismatched.append(f"{row['seg']}: index {idx} beyond {len(cuts)} cuts")
                continue
            audio, sr = cuts[idx]
            got = round(len(audio) / sr, 1)
            if abs(got - float(row["dur_s"])) > TOLERANCE_S:
                mismatched.append(f"{row['seg']}: cut {got}s but references.json says {row['dur_s']}s")
                continue
            sf.write(out / row["seg"], audio, sr)
            written += 1

    scored = sum(1 for r in refs if not r.get("excluded"))
    print(f"\nwrote {written}/{len(refs)} segments to {out}/  ({scored} are scored; "
          f"{len(refs) - scored} are marked excluded in references.json)")
    if mismatched:
        print(f"\n{len(mismatched)} MISMATCHES — do not score against these:", file=sys.stderr)
        for m in mismatched[:10]:
            print("  " + m, file=sys.stderr)
        return 1
    print("all cuts match the durations recorded in references.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
