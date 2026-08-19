"""Run any ASR system over NepTel and write a scorable hypothesis file.

Pick whichever adapter matches your system — you should not have to write a loop:

    # anything on the HF hub with a transformers ASR pipeline (Whisper, wav2vec2, MMS, ...)
    python run_system.py --hf openai/whisper-large-v3 --name whisper --lang ne

    # a NeMo checkpoint (.nemo file or hub id)
    python run_system.py --nemo nepali_conformer_offline.nemo --name mine

    # any command-line tool; {audio} is replaced with the wav path, stdout is the transcript
    python run_system.py --cmd "whisper-cli -f {audio} --output-txt -" --name whispercpp

    # your own Python function, `def transcribe(path: str) -> str`
    python run_system.py --py mypackage.mymodule:transcribe --name mine

Writes `outputs/<name>.json` in the format the scorer expects and prints the WER. Add
`--compare outputs/nepali-conformer-offline.json` for a paired bootstrap against ours.

Audio comes from `python fetch_audio.py neptel_audio` (public, no login).
"""
import argparse
import json
import os
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).parent


def load_segments(manifest):
    """The scored segments, in manifest order. Excluded rows are not transcribed."""
    man = json.load(open(manifest, encoding="utf-8"))
    return [s["seg"] for s in man["segments"] if not s.get("excluded")]


def adapter_hf(model_id, lang):
    from transformers import pipeline

    kwargs = {}
    if lang:
        kwargs["generate_kwargs"] = {"language": lang}
    asr = pipeline("automatic-speech-recognition", model=model_id)

    def run(path):
        out = asr(path, **kwargs)
        if isinstance(out, dict):
            return str(out.get("text", ""))
        return str(out)
    return run


def adapter_nemo(checkpoint):
    import nemo.collections.asr as nemo_asr

    restore = (nemo_asr.models.ASRModel.from_pretrained
               if not os.path.exists(checkpoint) else nemo_asr.models.ASRModel.restore_from)
    model = restore(checkpoint, map_location="cpu")
    model.eval()

    def run(path):
        out = model.transcribe([path], batch_size=1, verbose=False)[0]
        return out.text if hasattr(out, "text") else str(out)
    return run


def adapter_cmd(template):
    def run(path):
        cmd = template.replace("{audio}", path)
        proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"command failed ({proc.returncode}): {proc.stderr.strip()[:200]}")
        return proc.stdout.strip()
    return run


def adapter_py(spec):
    import importlib

    if ":" not in spec:
        raise SystemExit("--py expects module:function, e.g. mypkg.mymod:transcribe")
    modname, funcname = spec.rsplit(":", 1)
    sys.path.insert(0, os.getcwd())
    return getattr(importlib.import_module(modname), funcname)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="system name; writes outputs/<name>.json")
    ap.add_argument("--segdir", default="neptel_audio", help="directory of segment wavs")
    ap.add_argument("--manifest", default=str(HERE / "references.json"))
    ap.add_argument("--compare", help="hypothesis JSON to paired-bootstrap against")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--hf", help="Hugging Face model id for a transformers ASR pipeline")
    src.add_argument("--nemo", help="NeMo .nemo path or hub id")
    src.add_argument("--cmd", help="shell command; {audio} is the wav path, stdout the text")
    src.add_argument("--py", help="module:function taking a path and returning text")
    ap.add_argument("--lang", help="language hint for --hf (e.g. ne)")
    a = ap.parse_args()

    segdir = pathlib.Path(a.segdir)
    if not segdir.is_dir():
        print(f"no such directory: {segdir}\nrun: python fetch_audio.py {segdir}",
              file=sys.stderr)
        return 2

    segments = load_segments(a.manifest)
    missing = [s for s in segments if not (segdir / s).exists()]
    if missing:
        print(f"{len(missing)} segments missing from {segdir} (first: {missing[0]})\n"
              f"run: python fetch_audio.py {segdir}", file=sys.stderr)
        return 2

    if a.hf:
        run = adapter_hf(a.hf, a.lang)
    elif a.nemo:
        run = adapter_nemo(a.nemo)
    elif a.cmd:
        run = adapter_cmd(a.cmd)
    else:
        run = adapter_py(a.py)

    rows, failures = [], 0
    for i, seg in enumerate(segments, 1):
        try:
            text = run(str(segdir / seg))
        except Exception as exc:                      # one bad segment must not lose the run
            print(f"  [{i}/{len(segments)}] {seg} FAILED: {exc}", file=sys.stderr)
            text, failures = "", failures + 1
        rows.append({"seg": seg, "text": text})
        if i % 10 == 0 or i == len(segments):
            print(f"  {i}/{len(segments)} segments", flush=True)

    out_dir = HERE / "outputs"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{a.name}.json"
    json.dump(rows, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\nwrote {out_path}" + (f"  ({failures} segments failed and are scored as empty)"
                                   if failures else ""), flush=True)

    cmd = [sys.executable, str(HERE.parent / "eval" / "score_reference.py"),
           "--manifest", a.manifest, "--hyp", str(out_path)]
    if a.compare:
        cmd += ["--compare", a.compare]
    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())
