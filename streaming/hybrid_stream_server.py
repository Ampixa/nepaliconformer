"""Hybrid streaming ASR: live cache-aware partials, wide-context final per turn.

This is HYBRID_TWO_PASS_PLAN.md made real, and the design is not a guess — every choice below
came out of measuring four live-arrival schemes against the same clips:

    whole-file streaming (the reference)   'सेनाको आकम हुनु अघि हाइटीले अठको दशक पछि ...'
    fresh buffer per chunk (v1 of this)    'अ दशक समस्याहरु सामान गर्नु पर्यो'      <- ~70% lost
    rebuild-over-all, step new chunks      'अङ्ख्या'                                <- worse
    online_normalization per chunk         'आको आइसक्ने बस्याहरु सामान्य ...'        <- poor
    OVERLAP per chunk  (what runs here)    'फेल आको म अघि हाइटन अठ वर्ष पछि यो रोग ...'
    full re-decode                         identical to whole-file

PASS 1 — partials. Each 0.5 s chunk is preprocessed together with 0.5 s of PRECEDING audio, and
only the newest chunk's step advances the encoder cache. The overlap supplies the pre-encode left
context the convolutional front-end needs across a chunk boundary; without it the model loses
most of the utterance. Cost is re-preprocessing 0.5 s of audio per update, which is trivial next
to the encoder step.

PASS 2 — finals. When the turn closes, the whole turn is decoded in one pass at full context.
That path measured IDENTICAL to whole-file streaming, i.e. it is the quality ceiling, and the
earlier two-pass measurement put the gain at +0.055 WER over partials with p95 finalize latency
of 205 ms — comfortably inside a turn gap.

TURN BOUNDARY is energy-based VAD, NOT the `<breath>` token. That token is unusable in this
checkpoint: measured recall 0.0045, and a bias sweep found no operating point where precision
exceeds 0.576. It is still surfaced in the partial text when the model emits it, so its rarity is
visible rather than hidden.
"""

import argparse
import asyncio
import json
import logging
import pathlib
import sys
import time

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("hybrid")

SR = 16000
CHUNK = 8000            # 0.50 s — measured to yield exactly one encoder step
OVERLAP = 8000          # 0.50 s of left context; without it partials collapse
EOU = "<breath>"


class Turn:
    """One turn: streaming partials while it runs, one wide-context decode when it closes."""

    def __init__(self, model, ctx, silence_rms, silence_ms, min_speech_ms):
        import torch

        self.torch = torch
        self.model = model
        self.ctx = ctx
        self.silence_rms = silence_rms
        self.silence_chunks = max(1, int(silence_ms / (1000 * CHUNK / SR)))
        self.min_speech_chunks = max(1, int(min_speech_ms / (1000 * CHUNK / SR)))
        self.reset()

    def reset(self):
        self.audio = np.zeros(0, dtype=np.float32)     # the whole turn, for the final pass
        self.pending = np.zeros(0, dtype=np.float32)   # not yet a full chunk
        self.model.encoder.set_default_att_context_size(list(self.ctx))
        self.model.encoder.setup_streaming_params()
        self.cch, self.ct, self.cl = self.model.encoder.get_initial_cache_state(batch_size=1)
        self.prev = None
        self.pred = None
        self.text = ""
        self.quiet = 0
        self.speech = 0
        self.chunks = 0

    def _step(self, sig, ln):
        with self.torch.no_grad():
            self.pred, tr, self.cch, self.ct, self.cl, self.prev = \
                self.model.conformer_stream_step(
                    processed_signal=sig, processed_signal_length=ln,
                    cache_last_channel=self.cch, cache_last_time=self.ct,
                    cache_last_channel_len=self.cl, keep_all_outputs=False,
                    previous_hypotheses=self.prev, previous_pred_out=self.pred,
                    drop_extra_pre_encoded=None, return_transcription=True)
        if tr:
            h = tr[0]
            self.text = (h.text if hasattr(h, "text") else str(h)) or self.text

    def feed(self, audio):
        """Returns a list of events: partials as they update, plus a final when the turn ends."""
        from nemo.collections.asr.parts.utils.streaming_utils import (
            CacheAwareStreamingAudioBuffer,
        )

        self.pending = np.concatenate([self.pending, audio])
        events = []
        while len(self.pending) >= CHUNK:
            chunk = self.pending[:CHUNK]
            self.pending = self.pending[CHUNK:]
            start = len(self.audio)
            self.audio = np.concatenate([self.audio, chunk])
            t0 = time.perf_counter()

            # The overlap is what makes partials legible; see the module docstring.
            lo = max(0, start - OVERLAP)
            seg = self.audio[lo:]
            buf = CacheAwareStreamingAudioBuffer(model=self.model, online_normalization=False)
            buf.append_audio(seg, stream_id=-1)
            got = list(buf)
            if got:
                self._step(*got[-1])

            rms = float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))
            if rms < self.silence_rms:
                self.quiet += 1
            else:
                self.quiet = 0
                self.speech += 1
            self.chunks += 1

            events.append({"type": "partial", "text": self.text,
                           "chunk": self.chunks, "ms": round((time.perf_counter() - t0) * 1000, 1),
                           "rms": round(rms, 4), "eou": self.text.count(EOU)})

            if self.quiet >= self.silence_chunks and self.speech >= self.min_speech_chunks:
                events.append(self.finalize())
        return events

    def finalize(self):
        """Wide-context decode of the whole turn — the quality ceiling, once."""
        import os
        import tempfile

        import soundfile as sf

        audio, partial = self.audio, self.text
        t0 = time.perf_counter()
        final = partial
        if len(audio) > SR * 0.3:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                sf.write(f.name, audio, SR)
                tmp = f.name
            try:
                o = self.model.transcribe([tmp], batch_size=1, verbose=False)[0]
                final = (o.text if hasattr(o, "text") else str(o)).strip() or partial
            except Exception:
                log.exception("finalize failed; keeping the partial")
            finally:
                os.unlink(tmp)
        dt = time.perf_counter() - t0
        dur = len(audio) / SR
        self.reset()
        return {"type": "final", "text": final, "partial": partial,
                "finalize_ms": round(dt * 1000, 1), "turn_s": round(dur, 1)}


def build_app(model, ctx, static_dir, nemo_path, vad):
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse, PlainTextResponse

    app = FastAPI()
    html = (static_dir / "hybrid_stream_client.html").read_text(encoding="utf-8")

    @app.get("/")
    async def index():
        return HTMLResponse(html)

    @app.get("/model")
    async def info():
        e = model.cfg.encoder
        return PlainTextResponse(json.dumps({
            "checkpoint": str(nemo_path),
            "att_context_style": str(e.get("att_context_style")),
            "att_context_size": str(e.get("att_context_size")),
            "conv_context_size": str(e.get("conv_context_size")),
            "streaming_context": str(list(ctx)),
            "chunk_ms": int(1000 * CHUNK / SR),
            "overlap_ms": int(1000 * OVERLAP / SR),
            "eou_token_id": model.cfg.get("eou_token_id"),
            "eou_usable": False,
            "eou_note": "recall 0.0045; bias sweep found no point with precision > 0.576",
        }, indent=2))

    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        await sock.accept()
        turn = Turn(model, ctx, **vad)
        log.info("connection open")
        try:
            while True:
                data = await sock.receive_bytes()
                audio = np.frombuffer(data, dtype=np.float32)
                if audio.size == 0:
                    continue
                for ev in await asyncio.to_thread(turn.feed, audio):
                    await sock.send_text(json.dumps(ev))
        except WebSocketDisconnect:
            log.info("connection closed")
        except Exception:
            log.exception("stream failed")

    return app


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nemo", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--context", default="70,13")
    ap.add_argument("--silence-rms", type=float, default=0.006,
                    help="below this a chunk counts as quiet; raise it in a noisy room")
    ap.add_argument("--silence-ms", type=int, default=1000)
    ap.add_argument("--min-speech-ms", type=int, default=1000)
    a = ap.parse_args()

    p = pathlib.Path(a.nemo)
    if not p.exists():
        log.error("checkpoint not found: %s", p)
        return 1

    import torch
    from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel

    torch.set_num_threads(a.threads)
    log.info("loading %s", p)
    t0 = time.perf_counter()
    model = EncDecHybridRNNTCTCBPEModel.restore_from(str(p), map_location=a.device)
    model.eval()
    model.change_decoding_strategy(decoder_type="rnnt")
    log.info("model ready in %.1f s", time.perf_counter() - t0)

    try:
        import uvicorn
    except ImportError:
        log.error("uvicorn is not installed")
        return 1

    ctx = [int(x) for x in a.context.split(",")]
    vad = {"silence_rms": a.silence_rms, "silence_ms": a.silence_ms,
           "min_speech_ms": a.min_speech_ms}
    app = build_app(model, ctx, pathlib.Path(__file__).resolve().parent, p, vad)
    log.info("open http://%s:%d  (partials %d ms + %d ms overlap; finals at %d ms silence)",
             a.host, a.port, int(1000 * CHUNK / SR), int(1000 * OVERLAP / SR), a.silence_ms)
    if a.host not in ("127.0.0.1", "localhost"):
        log.warning("binding to %s — unauthenticated endpoint, keep it on the tailnet", a.host)
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        log.exception("server failed")
        sys.exit(1)
