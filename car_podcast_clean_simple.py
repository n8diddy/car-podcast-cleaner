# car_podcast_clean_simple.py
# Car speech cleanup:
#   1) If input is .m4a/.mp3, auto-convert to temp 48k WAV via ffmpeg (if available)
#   2) High-pass @50 Hz → (optional NoiseGate) → Low-pass @12 kHz → Gentle Compressor → Limiter
#   3) Loudness normalize to -16 LUFS (podcast)

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

TARGET_LUFS = -16.0        # -16 for podcasts; -14 for YouTube
TRUE_PEAK_CEILING = -1.0   # dBTP limiter ceiling
HPF_CUTOFF = 50.0          # preserve deep fundamentals
LPF_CUTOFF = 12000.0       # shave broadband hiss
SHELF_GAIN_DB = 2.0        # set to +1.5 or +2.0 to add warmth post-compression

IN_PATH  = sys.argv[1] if len(sys.argv) > 1 else "input.wav"
OUT_PATH = sys.argv[2] if len(sys.argv) > 2 else "output_clean.wav"


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


def build_board() -> Pedalboard:
    chain = [HighpassFilter(cutoff_frequency_hz=HPF_CUTOFF)]

    # Light gate to stop room/static being lifted by compression (tune threshold if it chops words)
    if HAVE_NOISEGATE:
        chain.append(NoiseGate(threshold_db=-42.0, ratio=4.0, attack_ms=5.0, release_ms=120.0))

    # Shave top-end hiss that doesn't help intelligibility
    chain.append(LowpassFilter(cutoff_frequency_hz=LPF_CUTOFF))

    # Gentle compression that won’t drag the low end up:
    # - higher threshold, lower ratio
    # - longer attack lets bass/fundamentals breathe
    chain.append(Compressor(
        threshold_db=-12.0,
        ratio=1.8,
        attack_ms=20.0,
        release_ms=140.0
    ))

    # Optional: restore warmth AFTER compression so it doesn't over-trigger the comp
    if HAVE_LOWSHELF and SHELF_GAIN_DB != 0.0:
        chain.append(LowShelfFilter(cutoff_frequency_hz=120.0, gain_db=SHELF_GAIN_DB))

    # Brickwall safety
    chain.append(Limiter(threshold_db=TRUE_PEAK_CEILING))
    return Pedalboard(chain)


def process_audio(x: np.ndarray, sr: int) -> np.ndarray:
    board = build_board()
    return board.process(x, sample_rate=sr)


def loudness_normalize(x: np.ndarray, sr: int, target_lufs: float = TARGET_LUFS) -> np.ndarray:
    """EBU R128/BS.1770 integrated loudness normalization with safety re-limit."""
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(x)
    gain_db = target_lufs - loudness
    y = (x * (10.0 ** (gain_db / 20.0))).astype(np.float32)
    y = Pedalboard([Limiter(threshold_db=TRUE_PEAK_CEILING)]).process(y, sample_rate=sr)  # safety re-limit
    return y


def main():
    src = ensure_wav(IN_PATH, target_sr=48000)
    x, sr = read_wav_mono_48k(src)
    y = process_audio(x, sr)
    y = loudness_normalize(y, sr, TARGET_LUFS)
    sf.write(OUT_PATH, y, sr, subtype="PCM_24")
    print(f"Done -> {OUT_PATH}")


if __name__ == "__main__":
    main()
