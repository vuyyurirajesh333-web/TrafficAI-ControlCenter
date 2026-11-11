# FFmpeg for H.264 browser videos

`perception.py` resolves an encoder in this order:

1. `ffmpeg` on system `PATH`
2. `app/backend/bin/ffmpeg.exe` (place a real Windows build here)
3. `imageio-ffmpeg` bundled binary (installed via `requirements.txt`)

## Note on `ffmpeg.zip`

The file `ffmpeg.zip` currently in this folder is **not a valid ZIP archive** (corrupt or incomplete download). Do not rely on extracting it.

**Recommended:** keep using `imageio-ffmpeg`, or download an official Windows build from https://www.gyan.dev/ffmpeg/builds/ and place `ffmpeg.exe` in this `bin/` directory.
