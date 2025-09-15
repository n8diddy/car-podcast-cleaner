# car_podcast_clean_simple.py
# Car speech cleanup:
# 1) If input is .m4a/.mp3, auto-convert to temp 48k WAV via ffmpeg (if available)
# 2) High-pass @50 Hz → (optional NoiseGate) → **Dynamic De-Esser (5–9 kHz)** → Low-pass @12 kHz → Gentle Compressor → Limiter
# 3) Loudness normalize to -16 LUFS (podcast)

import os
import sys
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

IN_PATH  = sys.argv[1] if len(sys.argv) > 1 else "input.wav"
OUT_PATH = sys.argv[2] if len(sys.argv) > 2 else "output_clean.wav"

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

# ---------- Main ----------
def main():
    src = ensure_wav(IN_PATH, target_sr=48000)
    x, sr = read_wav_mono_48k(src)
    y = process_audio(x, sr)
    y = loudness_normalize(y, sr, TARGET_LUFS)
    sf.write(OUT_PATH, y, sr, subtype="PCM_24")
    print(f"Done -> {OUT_PATH}")

if __name__ == "__main__":
    main()
