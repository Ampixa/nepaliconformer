# Results — every number, with its instrument and its caveats

All WERs are word-level Levenshtein on normalized text (both sides through
`asr/nepali_normalize.py`: Devanagari and Arabic digits → spoken Nepali words, punctuation
stripped, `<breath>` tokens removed). The scorer is `eval/score_reference.py`. Deltas carry
paired-bootstrap 95% CIs over segments (20k resamples).

## NepTel v0.1 — real Nepali call-center audio (the headline table)

75 segments / 2,375 reference words across 3 calls. References were drafted by Google Chirp 2 and
reviewed against the audio by a native speaker (49/57 of the newest batch accepted verbatim —
95.9% of usable segments unchanged; 8 flagged segments excluded; 2 word-level corrections).

| system | audio | WER | CER | sub | del | ins |
|---|---|---|---|---|---|---|
| [**Kriti Telephony**](https://github.com/Naamche-Labs/kriti-telephony) (Naamche Labs, 119M) — released checkpoint, our decode | canonical | **34.48** | — | 22.95 | 6.86 | 4.67 |
| **nepali-conformer-offline** (ours) — re-decoded from the released weights | canonical | **36.29** | — | 23.87 | 8.21 | 4.21 |
| teacher-v2 (ours, rejected lineage) | canonical | 34.69 | 17.75 | 22.57 | 8.97 | 3.16 |
| [Kriti](https://github.com/Naamche-Labs/kriti) (Naamche Labs, 119M, IndicConformer fine-tune) | canonical | 40.59 | — | 24.59 | 13.81 | 2.19 |
| **nepali-conformer-streaming** (ours, 520 ms) | canonical | 59.87 | 41.08 | 36.55 | 22.32 | 1.01 |
| MMS-1B-all (Meta, `npi` adapter, zero-shot) | canonical | 81.01 | — | — | — | — |
| Whisper-large-v3, zero-shot, anti-hallucination tuned | canonical | 96.29 | — | — | — | — |
| Whisper-large-v3, zero-shot, defaults† | canonical | 107.49 | — | — | — | — |
| IndicWav2Vec-Nepali (community mirror of the AI4Bharat fine-tune) | canonical | 86.57 | — | — | — | — |
| *Kriti Telephony — authors' submitted outputs (PR #2), decoded on their reconstruction of the gated source audio* | reconstructed | *32.38* | *14.36* | 23.12 | 5.89 | 3.37 |
| *nepali-conformer-offline — outputs published 2026-08, not reproducible from the released weights* | canonical | *33.81* | *16.63* | 22.86 | 7.41 | 3.54 |

*Correction note (2026-09-07).* Our published `nepali-conformer-offline.json` scored 33.81 but
does not reproduce from the released checkpoint: a fresh greedy decode of
`ampixa/nepali-conformer-offline` on the canonical audio scores **36.29**, and only 5 of 75
hypotheses match the published file. That file came from a training state that was never
shipped. It is retained as `benchmark/outputs/nepali-conformer-offline.published-2026-08.unreproduced.json`
and `benchmark/outputs/nepali-conformer-offline.json` is now the reproducible decode. Every delta
below is against the reproducible baseline. Separately, Kriti Telephony's submitted outputs were
produced on audio reconstructed from the gated vendor source with their own chunker (documented
in their repository); the same released checkpoint decoded on the canonical wavs scores 34.48,
with 2 of 77 hypotheses identical between the two runs — reconstruction drift is worth about 2
points here. `benchmark/AUDIO_SHA256.json` now pins the canonical audio so future rows can say
which audio they used. The two-call-mix caveat below still applies to every level in this table.

*Correction note (2026-08-19): earlier revisions of this file, the README and the project site
quoted Whisper-large-v3 at **99.4**. That figure predates the v2 reference set and does not
reproduce from the published outputs; scoring `benchmark/outputs/whisper-large-v3-zeroshot.json`
against the shipped references gives **96.29**, which is the number now published. The conclusion
is unchanged — Whisper is unusable on this audio — but the reproducible figure is the one that
belongs in the table.*

†The `defaults` row has no published hypothesis file, so it cannot be re-derived from this repo;
it is retained only as the before/after of anti-hallucination decoding, not as a citable number.

*S/D/I split measured on the 26-segment first batch; full-set split reproducible from
`benchmark/outputs/`.

Paired bootstrap deltas vs our **reproducible** offline decode (36.29), same canonical audio:
Kriti Telephony **−1.8 [−3.9, +0.2]** (statistical tie, lower point estimate); Kriti
**+4.3 [+1.7, +7.2]** (significant); teacher-v2 **−1.6 [−3.6, +0.4]** (statistical tie);
streaming **+23.6 [+20.5, +26.8]**; our unreproduced August file **−2.5 [−4.0, −1.0]**. The
authors' reconstructed-audio row (32.38) is not paired-comparable with any canonical-audio row:
the segment boundaries differ, so a per-segment bootstrap between the two is not meaningful.

*Kriti row methodology: their published checkpoint reduced to Nepali-only exactly as their
own loader does (first-257 embedding rows, `ne` joint head, CTC head dropped — see their
`src/kriti/model.py`), decoded greedily in mainline NeMo; weight surgery verified
`missing=0 unexpected=0`. Their outputs: `benchmark/outputs/kriti-naamche.json`.*

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
direction: from 31.5% (read) to 36.3% (real calls, reproducible decode) our degradation is moderate, while systems
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
