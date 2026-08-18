# Nepali Telephony ASR — research prototype

Streaming and offline speech recognition for **spontaneous Nepali telephone speech**, released
as a research prototype together with **NepTel**, a human-reviewed real-call benchmark, and the
full, honest evaluation — including the numbers that are bad.

**This is not a polished product.** It is a working system with measured strengths and measured
holes, released so the numbers exist in public. Nepali telephony ASR currently has no public
benchmark on real call audio; every published Nepali WER we know of is measured on read or
prompted speech. The gap between those two worlds is the headline finding of this release.

## Models

| model | params | mode | real-call WER† | HF checkpoint |
|---|---|---|---|---|
| `nepali-conformer-offline` | 121.3 M | full-context | **33.8** | [voidash/nepali-conformer-offline](https://huggingface.co/voidash/nepali-conformer-offline) |
| `nepali-conformer-streaming` | 121.3 M | cache-aware, 520 ms lookahead | 59.9 | [voidash/nepali-conformer-streaming](https://huggingface.co/voidash/nepali-conformer-streaming) |

†NepTel benchmark: 75 segments / 2,375 words of real Nepali call-center audio, references drafted
by Google Chirp 2 and reviewed word-by-segment by a native speaker. See `benchmark/` and
[RESULTS.md](RESULTS.md) for the full table, confidence intervals, and every caveat.

Both are 17-layer, d=512 Conformer encoders (subsampling ×4, 40 ms frames) with hybrid TDT/CTC
decoders and a 1,024-piece SentencePiece vocabulary, trained **from scratch** on ~1,655 h of
mostly conversational Nepali YouTube speech with Chirp 2 pseudo-labels, augmented with real
telephony codecs (AMR-NB, G.711, G.726, Opus), noise, reverb, and tempo perturbation. The
streaming model uses chunked-limited attention (`[[70,13],[70,6],[70,1],[70,0]]`) with fully
causal convolutions and runs in a cache-aware incremental loop.

## Quickstart

```bash
pip install nemo_toolkit[asr]
python asr/transcribe.py --nemo <checkpoint.nemo> audio.wav
# streaming demo (WebSocket server + browser client):
python streaming/hybrid_stream_server.py --nemo <streaming checkpoint>
```

## The one-paragraph honest summary

On read Nepali speech these models are competitive (~22% WER on OpenSLR-54-style audio). On
**real call audio** the offline model reaches **33.8% WER** — which, for calibration, beats
Whisper-large-v3 zero-shot on the same audio by **66 points** (Whisper: ~99%, it drifts into
Hindi orthography and hallucination loops on phone-band Nepali). The streaming variant pays a
large, measured penalty (59.9%), which decomposes into ~4 points of context restriction and ~19
points of training-lineage damage — details and the experiments behind that decomposition are in
[RESULTS.md](RESULTS.md). Known holes, all measured and quantified: English (the model is
effectively English-deaf beyond Devanagari loanwords), sung/melodic speech (emits at ~¼ its
normal rate), extreme slow speech, and end-of-turn prediction (the EOU token never fires on real
calls; the demo endpoints with energy VAD).

## NepTel benchmark

`benchmark/` contains the reference transcripts, per-model outputs, the scorer, and a fetch
script for the audio (published by [InfoBayAI](https://huggingface.co/datasets/InfoBayAI/Nepali_Call_Center_Audio_Dataset_Dual_Channel)
under CC-BY-4.0; our reference transcripts are CC-BY-4.0). If you have a Nepali ASR system, we
would genuinely like to see your row: run `eval/score_reference.py` and open a PR.

**Do not train on NepTel.** It is an evaluation set. Canary string for contamination checks:
`NEPTEL-CANARY-2026-8f3a1c92`.

## Layout

```
asr/         transcribe CLI, text normalizer (numbers -> spoken words), telephony simulator
eval/        the scorer used for every number in RESULTS.md
streaming/   cache-aware streaming server + browser client
benchmark/   NepTel: references, per-model outputs, fetch script, its own README
RESULTS.md   every measurement, with instruments and caveats
```

## License

Code: MIT. Model weights: CC-BY-NC-4.0 (training data includes crawled YouTube speech;
non-commercial is the honest license for this lineage). NepTel reference transcripts: CC-BY-4.0.
Benchmark audio: CC-BY-4.0, © InfoBayAI, redistributed with attribution.

## Citation

If you use the models or the benchmark, cite this repository. A technical report is in
preparation; the full experimental record (immutable one-JSON-per-run results store) backs every
number here.
