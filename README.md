# Universal Auto Clipper

A beginner-friendly local Streamlit website for turning uploaded videos into short clips. It scores potential moments using audio intensity, scene changes, and optional Whisper speech activity. No API key needed.

## Setup

1. Install Python 3.10+ and FFmpeg (including `ffprobe`). Make sure `ffmpeg -version` works in your terminal.
2. Open this folder in a terminal.
3. Create a virtual environment:
   - Windows: `py -m venv .venv` then `.venv\Scripts\activate`
   - macOS/Linux: `python3 -m venv .venv` then `source .venv/bin/activate`
4. Install dependencies: `python -m pip install -r requirements.txt`
5. Start: `python -m streamlit run app.py`
6. Open http://localhost:8501, upload a video, choose options, and generate clips.

## Notes

- Test with a short MP4 first. Long videos can take a while and need significant disk space.
- Whisper transcription is optional and downloads a model the first time; transcription is used to estimate speech density, not to understand narrative importance.
- 9:16 export preserves the full frame with black padding; it does not automatically track subjects.
- This is a local prototype. Do not deploy publicly without upload size limits, authentication, job queues, and resource isolation.
- If you get an error installing PySceneDetect, check your Python version. If FFmpeg is missing, install it and restart your terminal.
