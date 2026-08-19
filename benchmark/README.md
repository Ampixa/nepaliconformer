# NepTel v0.1 — Nepali real-telephony ASR benchmark

75 segments / 2,375 reference words of **real Nepali call-center audio** (genuine 8 kHz line
recordings, two-party customer-support calls), with human-reviewed reference transcripts. To our
knowledge this is the first public Nepali ASR benchmark on real call audio.

**Do not train on this data. Evaluation only.**
Canary: `NEPTEL-CANARY-2026-8f3a1c92`

## Files

- `references.json` — segment list with reference transcripts (CC-BY-4.0, ours) and provenance
- `outputs/` — per-segment hypotheses for every system in RESULTS.md, so every number is
  re-derivable with `../eval/score_reference.py --hyp outputs/<system>.json`
- `fetch_audio.py` — downloads the ready-cut segments from
  [`ampixa/neptel`](https://huggingface.co/datasets/ampixa/neptel), our **public, ungated**
  mirror: no login, no access request. `--from-source` instead re-derives the cuts from the
  vendor dataset ([InfoBayAI Nepali Call Center Dual Channel](https://huggingface.co/datasets/InfoBayAI/Nepali_Call_Center_Audio_Dataset_Dual_Channel),
  CC-BY-4.0, gated on their side) and verifies each against the duration in `references.json`

`references.json` ships all 77 segments; the 2 carrying an `excluded` field (one rate-gated,
one flagged unintelligible in review) are skipped by the scorer, leaving the 75 scored
segments / 2,375 words quoted in the results. They are kept in the file rather than deleted so
the exclusions stay auditable.

## Reference protocol

1. Audio segmented on silence (≤25 s segments) from the vendor's dual-channel per-speaker files.
2. Reference drafts by Google Chirp 2 (`ne-NP`, `chirp_2` model).
3. A native Nepali speaker reviewed every segment against the audio: accept / correct / flag.
   Flagged segments (unintelligible, cut mid-word, or non-Nepali speech) are excluded entirely.
   Review statistics live in the provenance field of `references.json` and in the release notes.
4. A speaking-rate gate (>6 words/sec ⇒ excluded) guards against transcription-engine
   hallucinations becoming ground truth.

**Known limitations:** 3 calls, one vendor, vendor-side PII muting leaves digital-zero gaps in
some segments; references are reviewed drafts, not from-scratch transcriptions; systems trained
on Chirp 2 pseudo-labels (including ours) share label lineage with the reference drafts. The set
is sized for ±2-point deltas between systems, not for absolute-truth WERs. v0.2 will grow toward
100+ calls with speaker-disjoint dev/test splits.

## Run it yourself

```bash
pip install huggingface_hub soundfile
python fetch_audio.py neptel_audio  # writes 77 wavs — no login needed

# reproduce any published number without touching the audio:
python ../eval/score_reference.py --hyp outputs/nepali-conformer-offline.json
# -> 75 reference segments, 2375 words / WER 0.3381
```

## Plug in your system

`run_system.py` runs your ASR over the segments and scores it — pick the adapter that fits,
you should not have to write a loop:

```bash
# anything with a transformers ASR pipeline (Whisper, wav2vec2, MMS, ...)
python run_system.py --hf openai/whisper-large-v3 --name whisper --lang ne

# a NeMo checkpoint (.nemo path or hub id)
python run_system.py --nemo nepali_conformer_offline.nemo --name mine

# any CLI tool: {audio} becomes the wav path, stdout is the transcript
python run_system.py --cmd "whisper-cli -f {audio} --output-txt -" --name whispercpp

# your own function, `def transcribe(path: str) -> str`
python run_system.py --py mypackage.mymodule:transcribe --name mine
```

It writes `outputs/<name>.json` and prints the WER. Add
`--compare outputs/nepali-conformer-offline.json` for a paired bootstrap against ours, which
is the number that tells you whether a difference is real.

If your system does not fit any adapter, write the JSON yourself —
`[{"seg": "...", "text": "..."}, ...]` over the 75 scored segments — and score it directly:

```bash
python ../eval/score_reference.py --hyp your_outputs.json
```

Open a PR adding your outputs file and the score line. We will happily include any system,
especially ones that beat ours.
