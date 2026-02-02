# car_podcast_clean_simple.py
# Car speech cleanup:
# 1) If input is .m4a/.mp3, auto-convert to temp 48k WAV via ffmpeg (if available)
# 2) High-pass @50 Hz → (optional NoiseGate) → **Dynamic De-Esser (5–9 kHz)** → Low-pass @12 kHz → Gentle Compressor → Limiter
# 3) Optional breath attenuation (heuristic)
# 4) Loudness normalize to -16 LUFS (podcast)

import argparse
import os
import tempfile
import subprocess
import shutil

import numpy as np
import soundfile as sf
import pyloudnorm as pyln
from pedalboard import Pedalboard, HighpassFilter, LowpassFilter, Compressor, Limiter

# Optional modules (present on most 0.9.x installs)
try:
    from pedalboard import NoiseGate
    HAVE_NOISEGATE = True
except Exception:
    HAVE_NOISEGATE = False

try:
    from pedalboard import LowShelfFilter
    HAVE_LOWSHELF = True
except Exception:
    HAVE_LOWSHELF = False

try:
    import webrtcvad
    HAVE_WEBRTCVAD = True
except Exception:
    HAVE_WEBRTCVAD = False

# ---- User-tunable defaults ----
TARGET_LUFS = -16.0        # -16 for podcasts; -14 for YouTube
TRUE_PEAK_CEILING = -1.0   # dBTP limiter ceiling

HPF_CUTOFF = 50.0          # preserve deep fundamentals
LPF_CUTOFF = 12000.0       # shave broadband hiss

# Low-end warmth post-compression (optional; you set this to 2.0 earlier)
SHELF_GAIN_DB = 2.0        # 0.0 to disable (requires LowShelfFilter)

# Dynamic De-Esser settings (pure NumPy/SciPy; no Pedalboard dependency)
DEESS_BAND = (5000.0, 9000.0)     # Hz (sibilance zone)
DEESS_THRESHOLD_DBFS = -26.0      # start reducing above this (relative to 0 dBFS)
DEESS_RATIO = 4.0                 # 4:1 is typical
DEESS_MAX_REDUCTION_DB = 8.0      # cap maximum attenuation
DEESS_ATTACK_MS = 5.0             # fast attack
DEESS_RELEASE_MS = 80.0           # moderate release

# Breath attenuation settings
BREATH_FRAME_MS = 20.0
BREATH_HOP_MS = 10.0
BREATH_MIN_MS = 80.0
BREATH_ATTACK_MS = 8.0
BREATH_RELEASE_MS = 120.0
BREATH_PRE_MS_DEFAULT = 250.0
BREATH_MAX_MS_DEFAULT = 500.0
BREATH_RMS_OFFSET_DB = 6.0
BREATH_SPEECH_MARGIN_DB = 3.0
BREATH_PITCH_CONFIDENCE_THRESHOLD = 0.35
BREATH_ZCR_THRESHOLD = 0.12

# ---------- I/O helpers ----------
def ensure_wav(path: str, target_sr: int = 48000) -> str:
    """If non-WAV (e.g., .m4a/.mp3), convert to temp mono 48k WAV via ffmpeg; return WAV path."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".wav":
        return path
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit("FFmpeg not found. Install it or convert manually before running this script.")
    tmp_wav = os.path.join(tempfile.gettempdir(), "tmp_input_clean.wav")
    cmd = [ffmpeg, "-y", "-i", path, "-ac", "1", "-ar", str(target_sr), "-c:a", "pcm_s24le", tmp_wav]
    subprocess.run(cmd, check=True)
    return tmp_wav

def read_wav_mono_48k(path: str):
    """Read WAV, downmix to mono, resample to 48k if needed."""
    x, sr = sf.read(path, always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != 48000:
        try:
            import librosa  # lazy import
            x = librosa.resample(x, orig_sr=sr, target_sr=48000)
            sr = 48000
        except Exception:
            raise SystemExit("Resample required but librosa not installed. Run: pip install librosa (or pre-convert to 48k).")
    return x.astype(np.float32), sr

# ---------- Dynamic De-Esser (NumPy/SciPy) ----------
def de_ess(x: np.ndarray, sr: int) -> np.ndarray:
    """
    Dynamic sibilance reduction:
      - band-pass 5–9 kHz to detect sibilant content
      - fast attack / slower release envelope
      - reduce that band's gain only when envelope exceeds a threshold
      - recombine: x - (1-g)*band
    """
    from scipy.signal import butter, sosfilt

    lo, hi = DEESS_BAND
    # Clamp band edges safely
    hi = min(hi, sr * 0.49)
    lo = max(10.0, min(lo, hi - 100.0))
    # 4th-order bandpass
    sos = butter(4, [lo/(sr*0.5), hi/(sr*0.5)], btype='band', output='sos')
    band = sosfilt(sos, x).astype(np.float32)

    # Envelope follower on |band| with separate attack/release
    absb = np.abs(band) + 1e-9
    atk = np.exp(-1.0 / max(1, int(sr * (DEESS_ATTACK_MS / 1000.0))))
    rel = np.exp(-1.0 / max(1, int(sr * (DEESS_RELEASE_MS / 1000.0))))
    env = np.zeros_like(absb)
    e = 0.0
    for i in range(len(absb)):
        coeff = atk if absb[i] > e else rel
        e = coeff * e + (1.0 - coeff) * absb[i]
        env[i] = e

    # Convert envelope to dBFS
    env_db = 20.0 * np.log10(env + 1e-12)

    # Compute desired reduction in dB (soft knee via ratio), clamp to max reduction
    over_db = np.maximum(0.0, env_db - DEESS_THRESHOLD_DBFS)
    red_db = np.minimum(DEESS_MAX_REDUCTION_DB, over_db * (1.0 - 1.0/DEESS_RATIO))

    # Smooth reduction just a bit (reuse release)
    # (simple one-pole smoother is enough)
    alpha = rel
    red_db_s = np.zeros_like(red_db)
    r = 0.0
    for i in range(len(red_db)):
        r = alpha * r + (1.0 - alpha) * red_db[i]
        red_db_s[i] = r

    # Turn dB reduction into linear gain for the sibilant band
    gain = 10.0 ** (-red_db_s / 20.0)

    # Apply only to the band and recombine with dry (phase-coherent)
    band_reduced = band * gain
    y = x - (band - band_reduced)  # x - (1-gain)*band
    return y.astype(np.float32)

# ---------- Breath attenuation ----------
def _frame_rms(x: np.ndarray, frame_len: int, hop: int) -> np.ndarray:
    if len(x) < frame_len:
        return np.array([], dtype=np.float32)
    x = np.ascontiguousarray(x)
    frame_count = 1 + (len(x) - frame_len) // hop
    shape = (frame_count, frame_len)
    strides = (x.strides[0] * hop, x.strides[0])
    frames = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    return np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12).astype(np.float32)


def _frame_zcr(x: np.ndarray, frame_len: int, hop: int) -> np.ndarray:
    if len(x) < frame_len:
        return np.array([], dtype=np.float32)
    x = np.ascontiguousarray(x)
    frame_count = 1 + (len(x) - frame_len) // hop
    shape = (frame_count, frame_len)
    strides = (x.strides[0] * hop, x.strides[0])
    frames = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    signs = np.sign(frames)
    signs[signs == 0] = 1.0
    zc = np.sum(signs[:, 1:] != signs[:, :-1], axis=1)
    return (zc / float(frame_len)).astype(np.float32)


def _frame_pitch_confidence(x: np.ndarray, sr: int, frame_len: int, hop: int) -> np.ndarray:
    if len(x) < frame_len:
        return np.array([], dtype=np.float32)
    x = np.ascontiguousarray(x)
    frame_count = 1 + (len(x) - frame_len) // hop
    confidences = np.zeros(frame_count, dtype=np.float32)
    min_lag = int(sr / 300.0)
    max_lag = int(sr / 80.0)
    for i in range(frame_count):
        start = i * hop
        frame = x[start:start + frame_len]
        if len(frame) < frame_len:
            break
        frame = frame - np.mean(frame)
        if np.allclose(frame, 0.0):
            continue
        autocorr = np.correlate(frame, frame, mode="full")[frame_len - 1:]
        if max_lag >= len(autocorr):
            continue
        window = autocorr[min_lag:max_lag]
        if window.size == 0:
            continue
        peak = np.max(window)
        confidences[i] = float(peak / (autocorr[0] + 1e-12))
    return confidences


def _speech_mask_vad(x: np.ndarray, sr: int, frame_len: int, hop: int) -> np.ndarray:
    if not HAVE_WEBRTCVAD:
        return np.array([], dtype=bool)
    vad = webrtcvad.Vad(2)
    x16 = np.clip(x, -1.0, 1.0)
    x16 = (x16 * 32768.0).astype(np.int16)
    frame_count = 1 + (len(x16) - frame_len) // hop if len(x16) >= frame_len else 0
    mask = np.zeros(frame_count, dtype=bool)
    for i in range(frame_count):
        start = i * hop
        frame = x16[start:start + frame_len]
        if len(frame) < frame_len:
            break
        mask[i] = vad.is_speech(frame.tobytes(), sr)
    return mask


def _speech_mask_energy(x: np.ndarray, frame_len: int, hop: int) -> np.ndarray:
    rms = _frame_rms(x, frame_len, hop)
    if rms.size == 0:
        return np.array([], dtype=bool)
    rms_db = 20.0 * np.log10(rms + 1e-12)
    noise_floor = np.percentile(rms_db, 20)
    threshold = noise_floor + 10.0
    return rms_db > threshold


def _min_duration(mask: np.ndarray, min_frames: int) -> np.ndarray:
    if mask.size == 0 or min_frames <= 1:
        return mask
    cleaned = mask.copy()
    start = None
    for i, is_on in enumerate(mask):
        if is_on and start is None:
            start = i
        if not is_on and start is not None:
            if i - start < min_frames:
                cleaned[start:i] = False
            start = None
    if start is not None and len(mask) - start < min_frames:
        cleaned[start:] = False
    return cleaned


def _breath_envelope(target_gain: np.ndarray, sr: int, attack_ms: float, release_ms: float) -> np.ndarray:
    if target_gain.size == 0:
        return target_gain
    attack_coeff = np.exp(-1.0 / max(1, int(sr * (attack_ms / 1000.0))))
    release_coeff = np.exp(-1.0 / max(1, int(sr * (release_ms / 1000.0))))
    env = np.zeros_like(target_gain)
    g = 1.0
    for i, tgt in enumerate(target_gain):
        coeff = attack_coeff if tgt < g else release_coeff
        g = coeff * g + (1.0 - coeff) * tgt
        env[i] = g
    return env


def _select_inhale_segments(candidate: np.ndarray, speech_mask: np.ndarray, pre_frames: int, min_frames: int, max_frames: int) -> np.ndarray:
    selected = np.zeros_like(candidate, dtype=bool)
    if candidate.size == 0:
        return selected
    speech_indices = np.where(speech_mask)[0]
    start = None
    for i, is_on in enumerate(candidate):
        if is_on and start is None:
            start = i
        if (not is_on or i == len(candidate) - 1) and start is not None:
            end = i if is_on else i - 1
            length = end - start + 1
            if length >= min_frames and length <= max_frames:
                next_speech = speech_indices[speech_indices > end]
                if next_speech.size > 0:
                    gap = next_speech[0] - end
                    if gap <= pre_frames:
                        selected[start:end + 1] = True
            start = None
    return selected


def attenuate_breaths(x: np.ndarray, sr: int, attenuation_db: float, pre_ms: float, max_ms: float) -> np.ndarray:
    frame_len = int(sr * (BREATH_FRAME_MS / 1000.0))
    hop = int(sr * (BREATH_HOP_MS / 1000.0))
    if frame_len <= 0 or hop <= 0 or len(x) < frame_len:
        return x

    speech_mask = _speech_mask_vad(x, sr, frame_len, hop)
    if speech_mask.size == 0:
        speech_mask = _speech_mask_energy(x, frame_len, hop)

    if speech_mask.size == 0:
        return x

    rms = _frame_rms(x, frame_len, hop)
    if rms.size == 0:
        return x
    rms_db = 20.0 * np.log10(rms + 1e-12)

    if np.any(speech_mask):
        speech_rms_db = np.mean(rms_db[speech_mask])
    else:
        return x

    non_speech = ~speech_mask
    if np.any(non_speech):
        noise_floor = np.percentile(rms_db[non_speech], 20)
    else:
        noise_floor = np.percentile(rms_db, 20)

    min_db = noise_floor + BREATH_RMS_OFFSET_DB
    max_db = speech_rms_db - BREATH_SPEECH_MARGIN_DB
    if max_db <= min_db:
        return x

    pitch_conf = _frame_pitch_confidence(x, sr, frame_len, hop)
    if pitch_conf.size == 0:
        return x
    zcr = _frame_zcr(x, frame_len, hop)
    if zcr.size == 0:
        return x

    low_pitch_or_noisy = (pitch_conf < BREATH_PITCH_CONFIDENCE_THRESHOLD) | (zcr > BREATH_ZCR_THRESHOLD)

    candidate = (
        (rms_db > min_db)
        & (rms_db < max_db)
        & low_pitch_or_noisy
        & non_speech
    )

    min_frames = int(np.ceil(BREATH_MIN_MS / BREATH_HOP_MS))
    max_frames = int(np.ceil(max_ms / BREATH_HOP_MS))
    pre_frames = int(np.ceil(pre_ms / BREATH_HOP_MS))

    candidate = _min_duration(candidate, min_frames)
    inhale_mask = _select_inhale_segments(candidate, speech_mask, pre_frames, min_frames, max_frames)

    if not np.any(inhale_mask):
        return x

    target_gain_frames = np.where(inhale_mask, 10.0 ** (-attenuation_db / 20.0), 1.0).astype(np.float32)

    frame_centers = (np.arange(target_gain_frames.size) * hop + frame_len / 2) / sr
    sample_times = np.arange(len(x)) / sr
    target_gain = np.interp(sample_times, frame_centers, target_gain_frames, left=1.0, right=1.0).astype(np.float32)

    env = _breath_envelope(target_gain, sr, BREATH_ATTACK_MS, BREATH_RELEASE_MS)
    y = x * env
    return y.astype(np.float32)

# ---------- Pedalboard chain ----------
def build_board() -> Pedalboard:
    chain = [HighpassFilter(cutoff_frequency_hz=HPF_CUTOFF)]
    if HAVE_NOISEGATE:
        chain.append(NoiseGate(threshold_db=-42.0, ratio=4.0, attack_ms=5.0, release_ms=120.0))
    chain.append(LowpassFilter(cutoff_frequency_hz=LPF_CUTOFF))
    # Gentle compressor that won’t drag the low end up
    chain.append(Compressor(
        threshold_db=-12.0,
        ratio=1.8,
        attack_ms=20.0,
        release_ms=140.0
    ))
    # Optional warmth AFTER compression so it doesn't re-trigger the comp
    if HAVE_LOWSHELF and SHELF_GAIN_DB != 0.0:
        chain.append(__import__('pedalboard').LowShelfFilter(cutoff_frequency_hz=120.0, gain_db=SHELF_GAIN_DB))
    chain.append(Limiter(threshold_db=TRUE_PEAK_CEILING))
    return Pedalboard(chain)

def process_audio(x: np.ndarray, sr: int) -> np.ndarray:
    # De-ess BEFORE the LPF/compressor so the compressor isn't fed harshness
    x = de_ess(x, sr)
    board = build_board()
    return board.process(x, sample_rate=sr)

# ---------- Loudness ----------
def loudness_normalize(x: np.ndarray, sr: int, target_lufs: float = TARGET_LUFS) -> np.ndarray:
    """EBU R128/BS.1770 integrated loudness normalization with safety re-limit."""
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(x)
    gain_db = target_lufs - loudness
    y = (x * (10.0 ** (gain_db / 20.0))).astype(np.float32)
    y = Pedalboard([Limiter(threshold_db=TRUE_PEAK_CEILING)]).process(y, sample_rate=sr)  # safety re-limit
    return y

# ---------- CLI ----------
def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Clean car podcast audio with optional de-breathing.")
    parser.add_argument("input", nargs="?", default="input.wav", help="Input audio path (wav/m4a/mp3).")
    parser.add_argument("output", nargs="?", default="output_clean.wav", help="Output WAV path.")
    parser.add_argument("--debreath", action="store_true", help="Enable breath attenuation before loudness normalization.")
    parser.add_argument("--debreath-db", type=float, default=8.0, help="Breath attenuation amount in dB (default: 8).")
    parser.add_argument("--debreath-pre-ms", type=float, default=BREATH_PRE_MS_DEFAULT, help="Max gap in ms before speech (default: 250).")
    parser.add_argument("--debreath-max-ms", type=float, default=BREATH_MAX_MS_DEFAULT, help="Max inhale duration in ms (default: 500).")
    return parser.parse_args(argv)

# ---------- Main ----------
def main(argv=None):
    args = parse_args(argv)
    src = ensure_wav(args.input, target_sr=48000)
    x, sr = read_wav_mono_48k(src)
    y = process_audio(x, sr)
    if args.debreath:
        y = attenuate_breaths(y, sr, attenuation_db=args.debreath_db, pre_ms=args.debreath_pre_ms, max_ms=args.debreath_max_ms)
    y = loudness_normalize(y, sr, TARGET_LUFS)
    sf.write(args.output, y, sr, subtype="PCM_24")
    print(f"Done -> {args.output}")

if __name__ == "__main__":
    main()
