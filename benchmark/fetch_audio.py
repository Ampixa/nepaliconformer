"""Fetch the NepTel benchmark audio.

Default path: download the ready-cut segments from `ampixa/neptel` on Hugging Face. That
mirror is public and ungated — no login, no access request — because a benchmark nobody can
run is not a benchmark.

    python fetch_audio.py [outdir]              # default: neptel_audio/
    python fetch_audio.py [outdir] --from-source

`--from-source` re-derives the cuts from the original vendor dataset instead
(InfoBayAI/Nepali_Call_Center_Audio_Dataset_Dual_Channel, CC-BY-4.0, gated on their side, so
this path needs `hf auth login` and a one-click access request). Segments are fixed
25-second windows over each per-speaker channel file, numbered sequentially across SPEAKER_00
then SPEAKER_01 within a call; a reference row's `seg` name encodes (sample, index). Every cut
is checked against the `dur_s` in references.json so a misalignment fails loudly instead of
silently producing a different benchmark. Boundary durations can differ by a few tenths of a
second from the originals (the reference cuts were lightly trimmed); that is within tolerance
and does not move WER.
"""
import json
import pathlib
import shutil
import sys

MIRROR = "ampixa/neptel"
SOURCE = "InfoBayAI/Nepali_Call_Center_Audio_Dataset_Dual_Channel"
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
        return stem[1:3], int(stem.split("seg_")[1])
    return "01", int(stem.split("seg_")[1])


def from_mirror(out, refs):
    from huggingface_hub import snapshot_download

    src = pathlib.Path(snapshot_download(MIRROR, repo_type="dataset",
                                         allow_patterns="audio/*.wav"))
    written, missing = 0, []
    for row in refs:
        wav = src / "audio" / row["seg"]
        if not wav.exists():
            missing.append(f"{row['seg']}: not present in {MIRROR}")
            continue
        shutil.copyfile(wav, out / row["seg"])
        written += 1
    return written, missing


def from_source(out, refs):
    import soundfile as sf
    from huggingface_hub import snapshot_download

    src = pathlib.Path(snapshot_download(SOURCE, repo_type="dataset"))
    print(f"source at {src}")
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
                mismatched.append(
                    f"{row['seg']}: cut {got}s but references.json says {row['dur_s']}s")
                continue
            sf.write(out / row["seg"], audio, sr)
            written += 1
    return written, mismatched


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    use_source = "--from-source" in sys.argv

    out = pathlib.Path(args[0] if args else "neptel_audio")
    here = pathlib.Path(__file__).parent
    refs = json.load(open(here / "references.json", encoding="utf-8"))["segments"]
    out.mkdir(parents=True, exist_ok=True)

    written, problems = (from_source if use_source else from_mirror)(out, refs)

    scored = sum(1 for r in refs if not r.get("excluded"))
    print(f"\nwrote {written}/{len(refs)} segments to {out}/  ({scored} are scored; "
          f"{len(refs) - scored} are marked excluded in references.json)")
    if problems:
        print(f"\n{len(problems)} PROBLEMS — do not score against these:", file=sys.stderr)
        for m in problems[:10]:
            print("  " + m, file=sys.stderr)
        return 1
    print("done — score with: python ../eval/score_reference.py --hyp your_outputs.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
