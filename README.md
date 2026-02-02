# Car Podcast Cleaner

Car cabins are noisy, narrow-band spaces that make lavalier recordings a challenge.
This project provides a ready-to-run Python script—`car_podcast_clean_simple.py`—that
converts a raw in-car interview into a polished, podcast-ready WAV. The chain covers
rumble removal, gentle dynamic control, optional warmth, true-peak limiting, and
loudness normalization to the -16 LUFS industry target.

## Features
- Converts non-WAV sources (e.g., `.m4a`, `.mp3`) to mono 48 kHz WAV via FFmpeg.
- High-pass filter at 50 Hz to tame engine rumble while preserving vocal body.
- Dynamic de-esser tuned for lavalier sibilance (5–9 kHz focus band).
- Gentle compressor followed by a true-peak limiter at -1 dBTP.
- Optional low-shelf boost for warmth when the Pedalboard module is available.
- Loudness normalization to -16 LUFS using ITU-R BS.1770 (pyloudnorm).

## Requirements
- Python 3.11 or newer.
- `FFmpeg` available on your `PATH` (only required when feeding non-WAV sources).
- Python dependencies listed in `requirements.txt` (see below for optional add-ons).

## Installation
1. (Optional) Create and activate a virtual environment with Python 3.11+.
2. Install the core dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Install FFmpeg if you plan to process `.m4a`/`.mp3` files:
   - macOS (Homebrew): `brew install ffmpeg`
   - Ubuntu/Debian: `sudo apt-get install ffmpeg`
   - Windows: [Download from ffmpeg.org](https://ffmpeg.org/download.html) and add it to your PATH.

## Usage
Basic invocation writes a 24-bit WAV at 48 kHz:

```bash
python car_podcast_clean_simple.py input.wav cleaned.wav
```

Supplying an `.m4a` automatically triggers FFmpeg conversion to WAV first:

```bash
python car_podcast_clean_simple.py raw_car_chat.m4a cleaned.wav
```

If you omit the output filename, `output_clean.wav` is used by default.

## Optional dependencies and modules
- **Resampling (`librosa`)** – Only required when your source audio is not already
  48 kHz. If `librosa` is missing, the script exits with guidance to install it or
  pre-convert the audio.
- **Dynamic de-esser (`scipy`)** – The sibilance reduction block depends on
  `scipy.signal`. Install SciPy to retain the tuned de-esser for lav mics.
- **Breath attenuation (`py-webrtcvad`)** – Optional lightweight VAD to focus the
  debreathing stage on speech-adjacent frames. If missing, the script falls back
  to a simple energy-based speech detector.
- **Pedalboard extras (`NoiseGate`, `LowShelfFilter`)** – These processors ship
  with most Pedalboard 0.9.x builds. The script automatically enables the noise gate
  and gentle low-shelf warmth when the modules exist; otherwise, it quietly skips them.

## Breath attenuation
For breath-heavy recordings, enable the optional debreath stage to attenuate
breath-like noise before loudness normalization:

```bash
python car_podcast_clean_simple.py input.wav cleaned.wav --debreath
```

Flags:
- `--debreath` enables breath attenuation.
- `--debreath-db` sets the attenuation amount in dB (default: 8).
- `--debreath-hi` limits attenuation to the upper band (2–9 kHz) if you want to
  preserve low/mid energy.

Tuning tips:
- Increase `--debreath-db` if breaths still jump out after LUFS normalization.
- If fricatives feel dulled, lower `--debreath-db` or use `--debreath-hi`.

## Tips
- Try adjusting `TARGET_LUFS` or `SHELF_GAIN_DB` inside the script if you need a
  hotter mix or a warmer/brighter tonal balance.
- For batch processing, wrap the script in a shell loop or a simple Python driver
  that iterates over a folder of takes.

## License
This project is distributed under the [MIT License](LICENSE).
