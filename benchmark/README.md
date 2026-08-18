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
- `fetch_audio.py` — downloads the source audio from the canonical public dataset
  ([InfoBayAI Nepali Call Center Dual Channel](https://huggingface.co/datasets/InfoBayAI/Nepali_Call_Center_Audio_Dataset_Dual_Channel),
  CC-BY-4.0) and reproduces the segment cuts

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

## Submit a system

Run your system over the audio, write `[{"seg": "...", "text": "..."}, ...]`, then:

```bash
python ../eval/score_reference.py --hyp your_outputs.json
```

Open a PR adding your outputs file and the score line. We will happily include any system,
especially ones that beat ours.
