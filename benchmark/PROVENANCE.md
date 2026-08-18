# NepTel v0.1 — full provenance

This document exists so that every property of the benchmark can be checked without trusting
the people who built it — who are also the people whose model currently tops it. Read this
before comparing numbers.

## Authorship and conflict of interest

NepTel was constructed by the NepaliConformer authors. Our models were developed *before* the
benchmark existed and were not tuned on it; nevertheless, benchmark authorship is a structural
advantage (we chose the domain we are strongest in) and claims made on NepTel should be read as
scoped to it. All competitor systems were evaluated by us with per-segment outputs published in
`outputs/` so any scoring decision can be audited or re-run.

## Source audio

- **Origin:** [InfoBayAI/Nepali_Call_Center_Audio_Dataset_Dual_Channel](https://huggingface.co/datasets/InfoBayAI/Nepali_Call_Center_Audio_Dataset_Dual_Channel)
  — public sample recordings published by the data vendor under **CC-BY-4.0**. Genuine two-party
  Nepali customer-support calls, one WAV per speaker, 8 kHz.
- **Vendor preprocessing (disclosed by the vendor, visible in the audio):** automatic
  background-noise reduction, low-activity-voice removal, and PII muting. The muting leaves
  runs of exact digital zeros — Sample-01 is ~58% zeros; Samples 02/03 are lightly affected.
  This makes the audio *cleaner than raw line audio*; treat absolute WERs accordingly.
- **Samples used:** the three published calls ("Sample - 01/02/03"), both channels.

## Segmentation

Silence-based cutting (30 ms RMS windows, threshold 0.08 × 60th percentile, min 1.5 s,
max 25 s), per channel file, implemented in `fetch_audio.py`. 85 raw segments.

## References

1. **Draft:** Google Cloud Speech-to-Text v2, model `chirp_2`, language `ne-NP`
   (Sample-01: 2026-08-13; Samples 02/03: 2026-08-18).
2. **Human review:** a native Nepali speaker reviewed every segment against the audio in a
   purpose-built tool (play, correct, flag). Statistics:
   - Batch 1 (Sample-01, 28 segments): 27 accepted, 1 rejected (Chirp emitted a digit sequence
     over unintelligible audio).
   - Batch 2 (Samples 02/03, 57 segments): 49 usable, **8 flagged** (7 sub-6-second fragments,
     1 Hindi-language turn), **2 word-level corrections** (a star-code digit misheard as an
     honorific; a company name), 47/49 accepted verbatim — **95.9% verbatim acceptance**.
3. **Hallucination gate:** any reference implying >6 words/second is excluded automatically.
   This gate exists because one Chirp output looped a two-word phrase 170 times (122 words/s)
   and would otherwise have been 32% of the reference mass.
4. Final set: **77 usable segments; 75 scored** after gates; **2,375 reference words**.

## Scoring

`eval/score_reference.py`: word-level Levenshtein after normalization of both sides
(`asr/nepali_normalize.py`: Devanagari and Arabic digits → spoken Nepali words, punctuation
stripped, breath tokens removed). Deltas use a paired bootstrap over segments (20k resamples).

## Known biases, stated plainly

- **Reference lineage:** references are *reviewed Chirp drafts*, not from-scratch human
  transcriptions. Systems trained on Chirp pseudo-labels (ours) share error lineage with the
  drafts and gain some agreement advantage; systems that were not (Kriti, MMS, Whisper) do not.
  The reviewer's 95.9% verbatim acceptance bounds how much room this leaves, and it cannot
  explain multi-tens-of-points gaps, but it is nonzero and it favors us.
- **Size and mix:** 3 calls, one vendor, one telecom-support domain. Absolute levels moved
  ~5–8 points between call mixes during construction; **treat levels as mix-dependent and
  deltas as the robust quantity.**
- **Vendor cleaning** (above) makes this easier than raw telephony.

## Contamination protection

Evaluation only — **do not train on NepTel**. Canary: `NEPTEL-CANARY-2026-8f3a1c92`.

## Versioning

v0.1 (this release). v0.2 goals: 100+ calls, speaker-disjoint dev/test splits, from-scratch
human transcription for a subset, multi-engine reference drafting with disagreement review.
