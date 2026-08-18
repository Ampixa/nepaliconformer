# Results — every number, with its instrument and its caveats

All WERs are word-level Levenshtein on normalized text (both sides through
`asr/nepali_normalize.py`: Devanagari and Arabic digits → spoken Nepali words, punctuation
stripped, `<breath>` tokens removed). The scorer is `eval/score_reference.py`. Deltas carry
paired-bootstrap 95% CIs over segments (20k resamples).

## NepTel v0.1 — real Nepali call-center audio (the headline table)

75 segments / 2,375 reference words across 3 calls. References were drafted by Google Chirp 2 and
reviewed against the audio by a native speaker (49/57 of the newest batch accepted verbatim —
95.9% of usable segments unchanged; 8 flagged segments excluded; 2 word-level corrections).

| system | WER | CER | sub | del | ins |
|---|---|---|---|---|---|
| **nepali-conformer-offline** (ours) | **33.81** | 16.63 | — | — | — |
| teacher-v2 (ours, rejected lineage) | 34.69 | 17.75 | — | — | — |
| **nepali-conformer-streaming** (ours, 520 ms) | 59.87 | 41.08 | 35.8* | 28.1* | 1.1* |
| Whisper-large-v3, zero-shot, anti-hallucination tuned | 99.41 | — | — | — | — |
| Whisper-large-v3, zero-shot, defaults | 107.49 | — | — | — | — |
| Kriti (Naamche Labs) | *pending — their runtime pins a NeMo fork revision; being reproduced* | | | | |

*S/D/I split measured on the 26-segment first batch; full-set split reproducible from
`benchmark/outputs/`.

Paired bootstrap deltas: teacher-v2 − offline = **+0.9 [−0.9, +2.8]** (statistical tie);
streaming − offline = **+26.1 [+23.2, +29.2]**.

**Reference caveats, stated plainly:** references are Chirp-2-drafted and human-*reviewed*, not
transcribed from scratch; our models trained on Chirp 2 pseudo-labels, so shared-error
circularity inflates our agreement somewhat (it cannot explain a 66-point gap to Whisper).
Absolute levels moved ~5–8 points when the call mix changed from 1 to 3 calls — treat levels as
call-mix-dependent and deltas as the trustworthy quantity until the set reaches 100+ calls.
Chirp 2 itself cannot be fairly scored on this set (it drafted the references).

## Why is Whisper at ~100%?

Not (only) hallucination loops. With `condition_on_previous_text=False`, temperature 0, and
silence-trimmed audio, the loops mostly disappear and the residual failure is **Hindi-drifted
Devanagari and English-caption leakage** on genuinely Nepali phone-band speech. Whisper's
fine-tuned read-speech Nepali numbers (≈15% WER on OpenSLR-54 in the literature) and this result
are both true: domain dominates. Raw per-segment outputs: `benchmark/outputs/`.

## The streaming penalty, decomposed

The 26-point offline↔streaming gap is **not primarily the streaming mask**:

| condition | WER |
|---|---|
| streaming checkpoint, maximal context `[1000,199]` | 60.86† |
| streaming checkpoint, served context `[70,13]` | 65.04† |
| offline checkpoint, full attention | 42.20† |

†First-batch (26-segment) reference. Context restriction costs ~4 points; the remaining ~19 is
training-lineage damage in the streaming checkpoint itself — its deletion rate barely moves with
lookahead. A blank-penalty sweep converts those deletions into substitutions, never into correct
words (WER 65.6 → 107.8 across the sweep): the information is absent upstream, not decoded away.

## Known holes (all measured, none hidden)

| hole | measurement |
|---|---|
| English | 94.6 WER (streaming) / 76.3 (offline) on real English call audio transliterated to Devanagari; the vocabulary has 3 multi-char Latin pieces; the model emits zero Latin on real audio |
| Melodic / sung speech | both models emit at ~¼ of their normal words-per-voiced-second on 57 Demucs-isolated Nepali vocal segments |
| Slow speech | 0.6× pitch-preserving stretch costs +11 WER points on both checkpoints (equal offline/streaming ⇒ data hole) |
| End-of-turn | the trained `<breath>` EOU token fires **zero** times across 108 real turn boundaries; endpointing in the demo is energy VAD |
| Noise: music | worst noise condition at every SNR despite music augmentation (pool was 700 excerpts from one catalog) |

## Read-speech anchor

On a held-out gold read slice (500 OpenSLR-54 utterances with human references, verified absent
from this checkpoint's training corpus), the released offline model scores **31.5% WER**
(35.6% through the telephony chain). Published fine-tuned systems reach ~15% on comparable read
data — on read speech we are mid-pack, and we say so. The point of this release is the other
direction: from 31.5% (read) to 33.8% (real calls) our degradation is small, while systems
optimized on read/prompted speech collapse on real calls. Read-speech WERs do not predict
telephony performance, and until NepTel there was no public way to see that for Nepali.

*Correction note: an earlier revision of this file quoted "≈22%" here; that number belonged to a
different, unreleased checkpoint. 31.5% is the released model's measured number.*

**Why no OpenSLR-54 test row?** All of OpenSLR-54 sits inside our training corpus (it is our
only human-labeled training source), so any SLR54 number from us would be train-set performance.
We refuse to publish that as a benchmark; the held-out W1 read slice above is the honest
substitute.

## Training data, honestly

~1,655 h: conversational YouTube speech (podcasts, interviews — ~60–80% of hours),
OpenSLR-54 read speech (105 h, the only human-labeled source), news/prompted corpora. Labels for
everything except OpenSLR-54 are **Google Chirp 2 pseudo-labels** (Chirp 2 measures ≈19.5% WER on
FLEURS ne_np and ≈11% of its words carry meaning-changing errors on hard audio — this is the
label ceiling of the lineage). Augmentation: real codec encode/decode (AMR-NB 4.75/7.4/12.2k,
G.711 µ/a, G.726, Opus), 300–3400 Hz bandpass, packet loss, AGC, additive noise at 5–25 dB SNR,
room impulse responses, tempo perturbation 0.7–1.25×.

One negative result we think is worth publishing: fine-tuning the offline model for 34 GPU-hours
on the fully-augmented corpus improved our synthetic noisy dev set from ~28% to 22.3% and moved
real-call WER by **+0.9 [−0.9, +2.8]** — nothing. Synthetic-dev validation was blind to this.
Real benchmarks or bust.
