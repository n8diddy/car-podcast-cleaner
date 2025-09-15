# car_podcast_clean_simple.py
# Car speech cleanup:
#   1) Convert non-WAV to temp WAV via ffmpeg (if needed)
#   2) High-pass @80 Hz  → NoiseGate (light) → Low-pass @12 kHz → Compressor → Limiter
#   3) Loudness normalize to -16 LUFS (podcast)

import os, sys, tempfile, subprocess, shutil
import numpy as np
import soundfile as sf
import pyloudnorm as pyln
from pedalboard import Pedalboard, HighpassFilter, LowpassFilter, Compressor, Limiter
try:
    from pedalboard import NoiseGate  # present in pedalboard 0.9.x
    HAVE_NOISEGATE = True
except Exception:
    HAVE_NOISEGATE = False  # will run fine without it

TARGET_LUFS = -16.0        # -16 podcast, use -14 for YouTube
TRUE_PEAK_CEILING = -1.0   # dBTP
HPF_CUTOFF = 80.0          # rumble removal
LPF_CUTOFF = 12000.0       # tame broadband hiss above speech band

IN_PATH  = sys.argv[1] if len(sys.argv) > 1 else "input.wav"
OUT_PATH = sys.argv[2] if len(sys.argv) > 2 else "output_clean.wav"

def ensure_wav(path: str, target_sr: int = 48000) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".wav":
        return path
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit("FFmpeg not found. Install it or convert manually.")
    tmp_wav = os.path.join(tempfile.gettempdir(), "tmp_input_clean.wav")
    cmd = [ffmpeg, "-y", "-i", path, "-ac", "1", "-ar", str(target_sr), "-c:a", "pcm_s24le", tmp_wav]
    subprocess.run(cmd, check=True)
    return tmp_wav

def read_wav_mono_48k(path: str):
    x, sr = sf.read(path, always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != 48000:
        try:
            import librosa
            x = librosa.resample(x, orig_sr=sr, target_sr=48000)
            sr = 48000
        except Exception:
            raise SystemExit("Resample needed. `pip install librosa` or pre-convert to 48 kHz.")
    return x.astype(np.float32), sr

def build_board():
    chain = [
        HighpassFilter(cutoff_frequency_hz=HPF_CUTOFF),
    ]
    # Light gate to stop room/static from being raised by compression
    if HAVE_NOISEGATE:
        # Start conservative; adjust threshold if it chops words
        chain.append(NoiseGate(threshold_db=-42.0, ratio=4.0, attack_ms=5.0, release_ms=120.0))
    # Shave hiss that adds harshness; 12 kHz preserves intelligibility well
    chain.append(LowpassFilter(cutoff_frequency_hz=LPF_CUTOFF))
    # Softer compression so noise isn't dragged up as much
    chain.append(Compressor(threshold_db=-14.0, ratio=2.0, attack_ms=10.0, release_ms=120.0))
    chain.append(Limiter(threshold_db=TRUE_PEAK_CEILING))
    return Pedalboard(chain)

def process_audio(x: np.ndarray, sr: int) -> np.ndarray:
    board = build_board()
    return board.process(x, sample_rate=sr)

def loudness_normalize(x: np.ndarray, sr: int, target_lufs: float = TARGET_LUFS) -> np.ndarray:
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(x)
    gain_db = target_lufs - loudness
    y = (x * (10.0 ** (gain_db / 20.0))).astype(np.float32)
    # Safety re-limit after loudness bump
    y = Pedalboard([Limiter(threshold_db=TRUE_PEAK_CEILING)]).process(y, sample_rate=sr)
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
