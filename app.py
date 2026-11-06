import json
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import streamlit as st
from scenedetect import ContentDetector, detect

st.set_page_config(page_title="Universal Auto Clipper", page_icon="🎬", layout="wide")
st.title("🎬 Universal Auto Clipper")
st.caption("Upload a video and automatically make short clips using scene changes, audio energy and optional speech transcription.")


def run(*args):
    return subprocess.run(args, check=True, capture_output=True, text=True)


def duration_of(path):
    data = json.loads(run("ffprobe", "-v", "error", "-show_format", "-of", "json", str(path)).stdout)
    return float(data["format"]["duration"])


def audio_energy(path, duration, interval=2.0):
    # Decode mono PCM at 8 kHz. Read in chunks to avoid keeping long recordings in RAM.
    proc = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", "8000", "-f", "s16le", "pipe:1"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    values = []
    block = int(8000 * interval) * 2
    while True:
        raw = proc.stdout.read(block)
        if not raw:
            break
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
        values.append(float(np.sqrt(np.mean((samples / 32768) ** 2))) if len(samples) else 0)
    proc.wait()
    if not values:
        return np.zeros(max(1, int(np.ceil(duration / interval))), dtype=np.float32)
    return np.asarray(values, dtype=np.float32)


def transcript_segments(path, enabled):
    if not enabled:
        return []
    from faster_whisper import WhisperModel
    model = WhisperModel("tiny", device="cpu", compute_type="int8")
    segments, _ = model.transcribe(str(path), vad_filter=True, beam_size=1)
    return [{"start": float(s.start), "end": float(s.end), "text": s.text.strip()} for s in segments]


def scene_times(path, duration):
    try:
        scenes = detect(str(path), ContentDetector(threshold=30.0), show_progress=False)
        cuts = [float(scene[0].get_seconds()) for scene in scenes]
        return sorted(set([0.0] + cuts + [duration]))
    except Exception:
        return [0.0, duration]


def pick_clips(duration, energies, cuts, transcript, clip_length, count):
    # Evaluate candidates every five seconds, plus scene boundaries.
    last_start = max(0, duration - clip_length)
    starts = set(float(x) for x in np.arange(0, last_start + 0.01, 5.0))
    starts.update(min(last_start, max(0, c)) for c in cuts)
    starts.add(last_start)
    maximum = max(float(np.percentile(energies, 95)), 0.00001)
    candidates = []
    for start in sorted(starts):
        end = min(duration, start + clip_length)
        i, j = int(start // 2), min(len(energies), int(np.ceil(end / 2)))
        region = energies[i:max(i + 1, j)]
        audio_score = min(1.0, float(np.mean(region)) / maximum) if len(region) else 0
        peak_score = min(1.0, float(np.max(region)) / maximum) if len(region) else 0
        scene_score = min(1.0, sum(start < c < end for c in cuts) / 4)
        speech = [s for s in transcript if s["end"] > start and s["start"] < end]
        speech_score = min(1.0, sum(len(s["text"].split()) for s in speech) / max(1, clip_length * 2))
        score = 0.4 * audio_score + 0.2 * peak_score + 0.2 * scene_score + 0.2 * speech_score
        candidates.append((score, start, end))
    chosen = []
    for score, start, end in sorted(candidates, reverse=True):
        if all(max(0, min(end, b) - max(start, a)) < clip_length * 0.25 for _, a, b in chosen):
            chosen.append((score, start, end))
        if len(chosen) >= count:
            break
    return sorted(chosen, key=lambda x: x[1])


def render_clip(source, dest, start, end, vertical):
    vf = (["-vf", "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2"] if vertical else [])
    run("ffmpeg", "-y", "-v", "error", "-ss", str(start), "-i", str(source), "-t", str(end - start), *vf, "-c:v", "libx264", "-preset", "veryfast", "-crf", "24", "-c:a", "aac", "-movflags", "+faststart", str(dest))


if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
    st.error("FFmpeg and ffprobe are required. Install FFmpeg and restart the app.")
    st.stop()

file = st.file_uploader("Upload your video", type=["mp4", "mov", "mkv", "webm"])
left, right = st.columns(2)
with left:
    clip_length = st.slider("Clip length (seconds)", 10, 90, 30, 5)
    count = st.slider("Number of clips", 1, 10, 3)
with right:
    vertical = st.checkbox("Export 9:16 vertical clips (with padding)", value=True)
    transcribe = st.checkbox("Use AI speech transcription (slower; downloads model first time)", value=False)

if file and st.button("✂️ Find highlights and create clips", type="primary"):
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        suffix = Path(file.name).suffix.lower()
        source = folder / ("source" + suffix)
        with source.open("wb") as out:
            while chunk := file.read(1024 * 1024):
                out.write(chunk)
        try:
            with st.spinner("Analyzing scenes and sound..."):
                duration = duration_of(source)
                if duration <= 0:
                    raise ValueError("Video has no valid duration")
                energies = audio_energy(source, duration)
                cuts = scene_times(source, duration)
                speech = transcript_segments(source, transcribe)
                selected = pick_clips(duration, energies, cuts, speech, min(clip_length, duration), count)
            output_zip = folder / "clips.zip"
            with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_STORED) as archive:
                for index, (score, start, end) in enumerate(selected, 1):
                    clip = folder / f"clip_{index:02d}.mp4"
                    with st.spinner(f"Rendering clip {index}/{len(selected)}..."):
                        render_clip(source, clip, start, end, vertical)
                    archive.write(clip, clip.name)
                    st.subheader(f"Clip {index} — {start:.1f}s to {end:.1f}s")
                    st.video(clip.read_bytes())
                    st.download_button(f"Download clip {index}", clip.read_bytes(), file_name=clip.name, mime="video/mp4")
            st.download_button("⬇️ Download all clips (ZIP)", output_zip.read_bytes(), file_name="auto_clips.zip", mime="application/zip", type="primary")
            st.success("Done! Download the clips before refreshing the page.")
        except Exception as exc:
            st.error(f"Processing failed: {exc}")
            st.info("Try a short MP4 with H.264 video and AAC audio first.")

st.divider()
st.caption("Highlights are heuristic suggestions, not guaranteed best moments. Videos without audio are ranked mainly by scene changes. Uploaded files are processed locally when you run the app on your computer.")
