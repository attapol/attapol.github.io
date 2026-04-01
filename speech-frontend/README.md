# Speech Frontend Demo

This is a dependency-free browser demo for a speech preprocessing lesson. It lets you:

- record audio from the machine microphone
- view the waveform live and after capture
- compute and visualize a spectrogram
- inspect a single time-domain frame vector
- compute log-mel features and visualize them

## Run locally

Browser microphone access works reliably from `http://localhost`, so start a small static server in this folder:

```bash
python3 -m http.server 8000
```

Then open <http://localhost:8000> in your browser.

## Analysis settings

- Input is recorded from the browser microphone.
- Audio is downsampled to `16 kHz` for analysis.
- Frame length is `400` samples (`25 ms` at 16 kHz).
- Hop length is `160` samples (`10 ms` at 16 kHz).
- FFT size is `512`.
- Mel filterbank size is `40`.
