"""Telephony channel simulation.

Per DECISIONS.md D009 the load-bearing stage is CODEC COMPRESSION, not bandwidth
reduction: ARTPARK/IISc found pure 8 kHz band-limiting barely hurts modern
architectures while GSM/2G codec compression costs ~20-30% relative. Naive
downsampling is therefore a weak simulation and must not be used as one.

This module exists so that vendor-synthesised audio (Gemini, ElevenLabs) can be
degraded through a REAL codec rather than asked to imitate the sound of a phone
call. An imitation is a stylistic filter; it does not carry the artefacts an ASR
actually has to survive.

Chain:  resample 8k -> 300-3400 Hz bandpass -> codec encode/decode
        -> optional packet loss -> optional AGC -> resample back

NOT IMPLEMENTED: handset impulse response. No IR was available, and convolving
with a fabricated one would add a plausible-looking stage that models nothing.
Named here rather than silently omitted.
"""

from __future__ import annotations

import logging
import pathlib
import shutil
import subprocess
import tempfile

import numpy as np
import soundfile as sf
from scipy import signal

log = logging.getLogger(__name__)

TELEPHONY_SR = 8000
PASSBAND_HZ = (300.0, 3400.0)

# name -> (ffmpeg encode args, container extension)
# Bitrates chosen to bracket what Nepali mobile networks actually carry: AMR-NB
# 4.75k is the worst-case adaptive-rate floor, 12.2k the best case, G.711 is the
# SIP-trunk baseline, GSM-FR the legacy 2G case.
CODEC_PROFILES: dict[str, tuple[list[str], str]] = {
    "none":         ([], ""),
    "g711_ulaw":    (["-acodec", "pcm_mulaw"], ".wav"),
    "g711_alaw":    (["-acodec", "pcm_alaw"], ".wav"),
    "amr_nb_4750":  (["-acodec", "libopencore_amrnb", "-b:a", "4.75k"], ".amr"),
    "amr_nb_7400":  (["-acodec", "libopencore_amrnb", "-b:a", "7.4k"], ".amr"),
    "amr_nb_12200": (["-acodec", "libopencore_amrnb", "-b:a", "12.2k"], ".amr"),
    "g726_32k":     (["-acodec", "g726", "-b:a", "32k"], ".wav"),
    "opus_nb_16k":  (["-acodec", "libopus", "-b:a", "16k",
                      "-application", "voip", "-frame_duration", "20"], ".opus"),
}

# GSM-FR is deliberately absent. This ffmpeg build decodes GSM but has no
# encoder (libgsm/gsm_ms both missing). An earlier version of this file listed
# it because availability was checked with `ffmpeg -codecs`, which reports
# decoders too - the profile then failed at call time instead of at import.
# Rebuild ffmpeg with libgsm to add it.
UNAVAILABLE_PROFILES = {"gsm_fr": "requires an ffmpeg built with libgsm"}


def _require_ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if exe is None:
        raise RuntimeError("ffmpeg not found on PATH; the codec stage cannot run")
    return exe


def available_encoders() -> set[str]:
    """Encoder names this ffmpeg can actually encode with."""
    proc = subprocess.run(
        [_require_ffmpeg(), "-hide_banner", "-encoders"], capture_output=True, text=True
    )
    if proc.returncode != 0:
        raise RuntimeError(f"could not list ffmpeg encoders: {proc.stderr.strip()[:200]}")
    return {
        parts[1]
        for line in proc.stdout.splitlines()
        if (parts := line.split()) and len(parts) > 1
    }


def verify_profiles(profiles: list[str] | None = None) -> None:
    """Fail loudly at startup if a profile's encoder is missing.

    Call this before a batch run. A codec that is absent should stop the run at
    the start, not halfway through a corpus.
    """
    have = available_encoders()
    missing = []
    for name in profiles or list(CODEC_PROFILES):
        args, _ = CODEC_PROFILES[name]
        for i, a in enumerate(args):
            if a == "-acodec" and args[i + 1] not in have:
                missing.append(f"{name} (needs encoder {args[i + 1]!r})")
    if missing:
        raise RuntimeError("ffmpeg lacks encoders for: " + ", ".join(missing))


def bandpass(wav: np.ndarray, sr: int) -> np.ndarray:
    """300-3400 Hz, the analogue telephony passband."""
    nyq = sr / 2.0
    high = min(PASSBAND_HZ[1], nyq * 0.99)
    if PASSBAND_HZ[0] >= high:
        raise ValueError(f"passband {PASSBAND_HZ} invalid at sr={sr}")
    sos = signal.butter(6, [PASSBAND_HZ[0], high], btype="bandpass", fs=sr, output="sos")
    return signal.sosfilt(sos, wav).astype(np.float32)


# Handset sending response, as a rising tilt across the passband.
#
# WHY THIS EXISTS. Measuring real Nepali call audio against our own synthetic chain
# (experiments/w3_training/measure_sim_to_real_gap.py) found the codec and the SNR were both
# fine and the SPECTRAL BALANCE was not. Fraction of total energy, real calls vs our clean
# source material:
#
#       band          real     ours     ratio
#       800-1500 Hz   0.132    0.045     2.9x
#       1500-2500 Hz  0.082    0.009     8.8x
#       2500-3400 Hz  0.064    0.002    35.4x
#
# Our training audio carries almost nothing between 1.5 and 3.4 kHz relative to a real call.
# That band is where fricatives, stops and affricates are discriminated, which is exactly the
# class of error we see: mangled names, places and English loan words, on a model that handles
# ordinary Nepali well. A real handset's send response rises across this region; broadcast and
# read-speech recordings do not.
#
# WHY A TILT RANGE AND NOT THE ITU CURVE. The modified IRS characteristic (ITU-T P.830 Annex D)
# is the right reference, and its table could not be verified from a primary source here.
# Writing down a half-remembered curve would repeat the mistake this module previously refused
# to make. The measured ratio above implies roughly +15 dB at 3 kHz, but it rests on three clean
# files and a noise-cleaned call sample, so it is a direction and not a calibration. So the tilt
# is a RANGE, drawn per utterance: the model sees flat, IRS-like, and steeper-than-IRS responses
# and cannot overfit to any single one. Coverage is the goal, not a calibrated handset.
IRS_TILT_DB_RANGE = (0.0, 12.0)


def handset_response(wav: np.ndarray, sr: int, tilt_db: float) -> np.ndarray:
    """Rising send response: 0 dB below 500 Hz, +tilt_db by 3 kHz, rolled off past the passband.

    tilt_db=0 is a no-op, so this is safe to leave in the chain unconditionally.
    """
    if abs(tilt_db) < 0.01:
        return wav.astype(np.float32)
    nyq = sr / 2.0
    # Breakpoints in Hz with the gain each should reach. firwin2 needs a monotonic 0..1 grid
    # that starts at 0 and ends at Nyquist.
    pts = [(0.0, 0.0), (300.0, 0.0), (500.0, 0.0), (1000.0, tilt_db * 0.35),
           (2000.0, tilt_db * 0.8), (3000.0, tilt_db), (3400.0, tilt_db)]
    freqs = [f for f, _ in pts if f < nyq] + [nyq]
    gains = [g for f, g in pts if f < nyq] + [pts[-1][1]]
    # Strictly increasing frequencies are required; duplicates appear when nyq lands on a point.
    keep = [0] + [i for i in range(1, len(freqs)) if freqs[i] > freqs[i - 1]]
    freqs = [freqs[i] for i in keep]
    gains = [gains[i] for i in keep]
    taps = signal.firwin2(129, [f / nyq for f in freqs],
                          [10.0 ** (g / 20.0) for g in gains])
    out = signal.lfilter(taps, [1.0], wav)
    # Preserve level: this is a shaping stage, and any gain it adds would otherwise be read as a
    # loudness change by the AGC downstream.
    rms_in = float(np.sqrt(np.mean(wav.astype(np.float64) ** 2)))
    rms_out = float(np.sqrt(np.mean(out ** 2)))
    if rms_out > 1e-9 and rms_in > 1e-9:
        out = out * (rms_in / rms_out)
    return out.astype(np.float32)


# Speaking-rate perturbation.
#
# WHY THIS EXISTS. Field report (2026-08-16): both arm A and the streaming model miss speech
# entirely when a caller draws words out or speaks slowly. Checking the corpus build confirmed
# the cause: NOTHING in the training chain ever time-stretched a clip — noise, RIR, codec and
# band were all varied, tempo never was. This model is also TDT, whose duration head was fitted
# to normal-rate speech, so slow input pushes the frame-skip distribution off its support and
# the failure mode is the usual one: deletion.
#
# WHY FFMPEG ATEMPO. Pitch-preserving stretch (WSOLA) is what a slow TALKER sounds like;
# resampling would shift pitch and simulate a tape machine instead. scipy has no WSOLA, and this
# module already requires ffmpeg for the codec stage, so atempo adds no dependency. atempo
# accepts factors in [0.5, 100]; this wrapper rejects anything outside [0.5, 2.0] because a
# single-stage stretch beyond that is not a speaking-rate change but a sound effect.
TEMPO_RANGE = (0.5, 2.0)


def tempo_perturb(wav: np.ndarray, sr: int, factor: float) -> np.ndarray:
    """Pitch-preserving speaking-rate change. factor < 1 is SLOWER (duration / factor).

    factor=1 is a no-op, so this is safe to leave in a chain unconditionally.
    """
    if abs(factor - 1.0) < 1e-3:
        return np.ascontiguousarray(wav, dtype=np.float32)
    if not TEMPO_RANGE[0] <= factor <= TEMPO_RANGE[1]:
        raise ValueError(f"tempo factor {factor} outside {TEMPO_RANGE}")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    ffmpeg = _require_ffmpeg()
    with tempfile.TemporaryDirectory(prefix="tempo_") as td:
        d = pathlib.Path(td)
        src, out = d / "src.wav", d / "out.wav"
        sf.write(src, np.ascontiguousarray(wav, dtype=np.float32), sr, subtype="FLOAT")
        proc = subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", "-i", str(src),
             "-filter:a", f"atempo={factor}", "-ar", str(sr), str(out)],
            capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg atempo={factor} failed (rc={proc.returncode}): "
                               f"{proc.stderr.strip()[:400]}")
        y, out_sr = sf.read(str(out), dtype="float32", always_2d=False)
    if out_sr != sr:
        raise RuntimeError(f"atempo returned {out_sr} Hz, expected {sr}")
    if y.ndim > 1:
        y = y.mean(axis=1)
    got, want = len(y) / sr, len(wav) / sr / factor
    if abs(got - want) > 0.05 * want + 0.1:
        raise RuntimeError(f"atempo={factor} produced {got:.2f}s, expected {want:.2f}s — "
                           "the stretch did not happen; do not train on it")
    return np.ascontiguousarray(y, dtype=np.float32)


def resample(wav: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return wav.astype(np.float32)
    g = np.gcd(int(src_sr), int(dst_sr))
    return signal.resample_poly(wav, dst_sr // g, src_sr // g).astype(np.float32)


def apply_codec(wav8k: np.ndarray, profile: str) -> np.ndarray:
    """Encode and decode through a real codec. Input and output are 8 kHz mono."""
    if profile not in CODEC_PROFILES:
        raise KeyError(f"unknown codec profile {profile!r}; have {sorted(CODEC_PROFILES)}")
    enc_args, ext = CODEC_PROFILES[profile]
    if not enc_args:
        return wav8k.astype(np.float32)

    ffmpeg = _require_ffmpeg()
    with tempfile.TemporaryDirectory(prefix="telephony_") as td:
        d = pathlib.Path(td)
        src, coded, back = d / "src.wav", d / f"coded{ext}", d / "back.wav"
        sf.write(src, wav8k, TELEPHONY_SR, subtype="PCM_16")

        for stage, cmd in (
            ("encode", [ffmpeg, "-y", "-loglevel", "error", "-i", str(src),
                        "-ar", str(TELEPHONY_SR), "-ac", "1", *enc_args, str(coded)]),
            ("decode", [ffmpeg, "-y", "-loglevel", "error", "-i", str(coded),
                        "-ar", str(TELEPHONY_SR), "-ac", "1", str(back)]),
        ):
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                raise RuntimeError(
                    f"ffmpeg {stage} failed for profile {profile!r} "
                    f"(rc={proc.returncode}): {proc.stderr.strip()[:400]}"
                )

        out, sr = sf.read(str(back), dtype="float32", always_2d=False)
        if sr != TELEPHONY_SR:
            raise RuntimeError(f"decoded at {sr} Hz, expected {TELEPHONY_SR}")
    if out.ndim > 1:
        out = out.mean(axis=1)
    return np.ascontiguousarray(out, dtype=np.float32)


def packet_loss(wav8k: np.ndarray, rate: float, frame_ms: int = 20, seed: int = 0) -> np.ndarray:
    """Zero-fill dropped RTP frames. Zero-fill, not interpolation - a plain jitter
    buffer with no PLC is the pessimistic and more honest default."""
    if not 0.0 <= rate < 1.0:
        raise ValueError(f"loss rate must be in [0,1), got {rate}")
    if rate == 0.0:
        return wav8k
    n = int(TELEPHONY_SR * frame_ms / 1000)
    out = wav8k.copy()
    rng = np.random.default_rng(seed)
    for start in range(0, len(out) - n + 1, n):
        if rng.random() < rate:
            out[start : start + n] = 0.0
    return out


def agc(wav: np.ndarray, target_rms: float = 0.06, eps: float = 1e-9) -> np.ndarray:
    """Single-shot gain to a target RMS, then hard-limit. Approximates the fixed
    gain a handset applies; not a real time-varying AGC."""
    rms = float(np.sqrt(np.mean(wav.astype(np.float64) ** 2)))
    if rms < eps:
        return wav
    return np.clip(wav * (target_rms / rms), -1.0, 1.0).astype(np.float32)


def degrade(
    wav: np.ndarray,
    sr: int,
    codec: str = "amr_nb_12200",
    loss_rate: float = 0.0,
    use_agc: bool = True,
    return_sr: int | None = None,
    tilt_db: float = 0.0,
) -> tuple[np.ndarray, int]:
    """Full chain. Returns (degraded_audio, sample_rate).

    `return_sr` defaults to the input rate so degraded and clean stay aligned for
    paired comparison. Pass 8000 to keep the narrowband output.
    """
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav = np.ascontiguousarray(wav, dtype=np.float32)
    out_sr = sr if return_sr is None else return_sr

    x = resample(wav, sr, TELEPHONY_SR)
    x = bandpass(x, TELEPHONY_SR)
    # Handset before the codec: a real line encodes what the handset produced, so shaping after
    # the codec would let the codec allocate bits to a spectrum no handset ever sent.
    x = handset_response(x, TELEPHONY_SR, tilt_db)
    x = apply_codec(x, codec)
    x = packet_loss(x, loss_rate)
    if use_agc:
        x = agc(x)
    return resample(x, TELEPHONY_SR, out_sr), out_sr
