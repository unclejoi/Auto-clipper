import math
import shutil
import subprocess
import tempfile
import wave
import zipfile
from pathlib import Path

import numpy as np
import streamlit as st
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector


st.set_page_config(
    page_title="Universal AI Auto Clipper",
    page_icon="✂️",
    layout="wide"
)
st.title("✂️ Universal AI Auto Clipper")
st.caption(
    "Automatically ranks promising moments and exports "
    "30-second highlights."
)


def cmd(args):
    process = subprocess.run(args, capture_output=True, text=True)
    if process.returncode:
        raise RuntimeError(
            process.stderr[-2500:] or "Command failed"
        )
    return process.stdout


def check_ffmpeg():
    missing = [
        executable
        for executable in ("ffmpeg", "ffprobe")
        if not shutil.which(executable)
    ]
    if missing:
        raise RuntimeError(
            f"Missing {', '.join(missing)}. "
            "Install FFmpeg and add its bin folder to PATH."
        )


def duration_of(path):
    output = cmd([
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path)
    ])
    return float(output.strip())


def get_audio(path, wavpath):
    cmd([
        "ffmpeg", "-y",
        "-i", str(path),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        str(wavpath)
    ])


def audio_scores(wavpath, dur):
    with wave.open(str(wavpath), "rb") as audio_file:
        rate = audio_file.getframerate()
        width = audio_file.getsampwidth()
        channels = audio_file.getnchannels()
        raw = audio_file.readframes(audio_file.getnframes())

    if width != 2:
        raise RuntimeError("Expected 16-bit extracted audio.")

    samples = (
        np.frombuffer(raw, dtype=np.int16).astype(np.float32)
        / 32768.0
    )

    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)

    step = rate
    values = np.array([
        np.sqrt(np.mean(samples[i:i + step] ** 2) + 1e-12)
        for i in range(0, len(samples), step)
    ], dtype=np.float32)

    if not len(values):
        return np.zeros(max(1, math.ceil(dur)), dtype=np.float32)

    low, high = np.percentile(values, [10, 95])
    return np.clip(
        (values - low) / (high - low + 1e-8),
        0,
        1
    )


def scene_scores(path, dur, threshold):
    video = open_video(str(path))
    manager = SceneManager()
    manager.add_detector(ContentDetector(threshold=threshold))
    manager.detect_scenes(video=video, show_progress=False)
    scenes = manager.get_scene_list()

    scores = np.zeros(max(1, math.ceil(dur)), dtype=np.float32)

    for scene in scenes[1:]:
        timestamp = float(scene[0].seconds)
        index = min(
            len(scores) - 1,
            max(0, int(timestamp))
        )
        for offset, value in [
            (-2, 0.3),
            (-1, 0.6),
            (0, 1.0),
            (1, 0.6),
            (2, 0.3)
        ]:
            target = index + offset
            if 0 <= target < len(scores):
                scores[target] = max(scores[target], value)

    return scores, len(scenes) - 1


KEYWORDS = {
    "no way": 1.4,
    "oh my god": 1.5,
    "let's go": 1.2,
    "unbelievable": 1.4,
    "insane": 1.2,
    "amazing": 1.0,
    "hilarious": 1.2,
    "funny": 0.8,
    "crazy": 0.8,
    "shocking": 1.2,
    "surprise": 0.8,
    "victory": 1.0,
    "finally": 0.7,
    "wait": 0.4,
    "what": 0.3,
    "wow": 1.0,
    "secret": 0.8,
    "best": 0.5,
    "worst": 0.5
}


def whisper_scores(wavpath, dur, model_size):
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        return (
            np.zeros(max(1, math.ceil(dur)), dtype=np.float32),
            [],
            "Whisper unavailable. Install faster-whisper to enable speech analysis."
        )

    model = WhisperModel(
        model_size,
        device="cpu",
        compute_type="int8"
    )
    segments, _ = model.transcribe(
        str(wavpath),
        vad_filter=True
    )

    output = []
    scores = np.zeros(max(1, math.ceil(dur)), dtype=np.float32)

    for segment in segments:
        text = (segment.text or "").strip()
        if not text:
            continue

        lowered = text.lower()
        hits = sum(
            weight
            for keyword, weight in KEYWORDS.items()
            if keyword in lowered
        )
        value = min(
            1.0,
            0.15 + hits * 0.35
            + min(len(lowered.split()), 20) * 0.012
        )

        start_index = max(0, int(segment.start))
        end_index = min(
            len(scores) - 1,
            int(max(segment.start, segment.end))
        )

        for index in range(start_index, end_index + 1):
            scores[index] = max(scores[index], value)

        output.append({
            "start": float(segment.start),
            "end": float(segment.end),
            "text": text,
            "score": value
        })

    return scores, output, None


def smooth(values):
    if len(values) <= 1:
        return values
    return np.convolve(
        values,
        np.ones(5, dtype=np.float32) / 5,
        mode="same"
    )


def select_clips(
    dur,
    audio,
    visual,
    speech,
    has_speech,
    length=30,
    min_gap=30,
    min_score=25
):
    """
    Select every qualifying non-overlapping highlight.

    Clips are exactly 30 seconds. A source shorter than 30 seconds
    does not produce a clip.
    """
    if length != 30:
        raise ValueError("Clip duration must be exactly 30 seconds.")

    if dur < length:
        return []

    count = math.ceil(dur)

    def normalize_signal(signal):
        signal = np.asarray(signal, dtype=np.float32)

        if len(signal) == count:
            return signal

        if len(signal) == 0:
            return np.zeros(count, dtype=np.float32)

        return np.interp(
            np.linspace(0, len(signal) - 1, count),
            np.arange(len(signal)),
            signal
        ).astype(np.float32)

    audio_signal = normalize_signal(audio)
    visual_signal = normalize_signal(visual)
    speech_signal = normalize_signal(speech)

    if has_speech:
        combined = (
            0.30 * audio_signal
            + 0.20 * visual_signal
            + 0.50 * speech_signal
        )
    elif np.max(audio_signal) < 0.05:
        combined = visual_signal
    else:
        combined = (
            0.60 * audio_signal
            + 0.40 * visual_signal
        )

    combined = smooth(combined)
    candidates = []

    # Score each whole-second start position.
    last_start = int(dur - length)

    for start in range(last_start + 1):
        end = start + length
        segment = combined[start:end]

        if len(segment) != length:
            continue

        peak = float(np.max(segment))
        average = float(np.mean(segment))
        variation = float(np.std(segment))

        score = 100 * (
            0.60 * peak
            + 0.30 * average
            + 0.10 * min(variation * 2, 1.0)
        )

        if score >= min_score:
            candidates.append({
                "start": float(start),
                "end": float(end),
                "score": score
            })

    candidates.sort(
        key=lambda candidate: candidate["score"],
        reverse=True
    )

    selected = []
    effective_gap = max(length, min_gap)

    for candidate in candidates:
        if all(
            abs(candidate["start"] - previous["start"])
            >= effective_gap
            for previous in selected
        ):
            selected.append(candidate)

    return sorted(
        selected,
        key=lambda candidate: candidate["start"]
    )


def cut(source, output, start, length=30, vertical=True):
    if length != 30:
        raise ValueError("Clip duration must be exactly 30 seconds.")

    video_filter = (
        "scale=1080:1920:force_original_aspect_ratio=increase,"
        "crop=1080:1920"
        if vertical else
        "scale=1920:1080:force_original_aspect_ratio=decrease,"
        "pad=1920:1080:(ow-iw)/2:(oh-ih)/2"
    )

    # Input-side seeking is efficient; -t limits the output to 30 seconds.
    cmd([
        "ffmpeg", "-y",
        "-ss", f"{start:.3f}",
        "-i", str(source),
        "-t", "30",
        "-map", "0:v:0",
        "-map", "0:a?",
        "-vf", video_filter,
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "23",
        "-c:a", "aac",
        "-b:a", "128k",
        "-t", "30",
        "-movflags", "+faststart",
        str(output)
    ])

    actual_duration = duration_of(output)

    if abs(actual_duration - 30.0) > 0.15:
        raise RuntimeError(
            "Exported clip duration was "
            f"{actual_duration:.3f} seconds, not 30 seconds."
        )


with st.sidebar:
    st.header("Clip settings")

    length = 30
    st.info("All exported clips are targeted at exactly 30 seconds.")

    vertical = st.checkbox(
        "Vertical 9:16 export",
        value=True
    )

    use_whisper = st.checkbox(
        "Enable Whisper speech analysis",
        value=False
    )

    model_size = st.selectbox(
        "Whisper model",
        ["tiny", "base", "small"],
        disabled=not use_whisper
    )

    threshold = st.slider(
        "Scene-change sensitivity",
        10, 50, 27
    )

    gap = st.slider(
        "Minimum gap between clips (seconds)",
        30, 180, 30
    )

    min_score = st.slider(
        "Minimum highlight score",
        0, 100, 25
    )

    st.caption(
        "All qualifying non-overlapping highlights are selected."
    )


upload = st.file_uploader(
    "Upload a video",
    type=["mp4", "mov", "mkv", "webm", "avi"]
)

st.info(
    "Test with a short 1–5 minute video first. "
    "Whisper is optional and can be slow on free hosting."
)

if upload and st.button(
    "Detect ALL highlights and create 30-second clips",
    type="primary",
    use_container_width=True
):
    try:
        check_ffmpeg()

        with tempfile.TemporaryDirectory() as temporary_directory:
            temp = Path(temporary_directory)
            suffix = Path(upload.name).suffix or ".mp4"
            source = temp / f"input{suffix}"
            wav_path = temp / "audio.wav"

            source.write_bytes(upload.getbuffer())

            progress = st.progress(
                5,
                text="Reading video..."
            )

            duration = duration_of(source)

            if duration < 30:
                st.error(
                    "The uploaded video must be at least 30 seconds long."
                )
                st.stop()

            try:
                progress.progress(
                    15,
                    text="Analyzing audio..."
                )
                get_audio(source, wav_path)
                audio = audio_scores(wav_path, duration)
                has_audio = True
            except Exception as audio_error:
                st.warning(
                    f"Audio analysis unavailable: {audio_error}"
                )
                audio = np.zeros(
                    math.ceil(duration),
                    dtype=np.float32
                )
                has_audio = False

            progress.progress(
                35,
                text="Detecting visual scene changes..."
            )

            try:
                visual, scene_count = scene_scores(
                    source,
                    duration,
                    threshold
                )
            except Exception as scene_error:
                st.warning(
                    f"Scene detection issue: {scene_error}"
                )
                visual = np.zeros(
                    math.ceil(duration),
                    dtype=np.float32
                )
                scene_count = 0

            speech = np.zeros(
                math.ceil(duration),
                dtype=np.float32
            )
            transcript = []

            if use_whisper and has_audio:
                progress.progress(
                    45,
                    text="Transcribing speech with Whisper..."
                )
                speech, transcript, warning = whisper_scores(
                    wav_path,
                    duration,
                    model_size
                )
                if warning:
                    st.warning(warning)
            elif use_whisper:
                st.warning(
                    "No usable audio track; Whisper skipped."
                )

            progress.progress(
                65,
                text="Ranking highlights..."
            )

            clips = select_clips(
                duration,
                audio,
                visual,
                speech,
                bool(transcript),
                length=30,
                min_gap=gap,
                min_score=min_score
            )

            if not clips:
                st.error(
                    "No highlights met your minimum score. "
                    "Try lowering the minimum highlight score."
                )
                st.stop()

            output_directory = Path("generated_clips")
            output_directory.mkdir(exist_ok=True)

            for old_file in output_directory.glob("highlight_*.mp4"):
                old_file.unlink(missing_ok=True)

            made = []

            for index, clip in enumerate(clips, 1):
                output = output_directory / (
                    f"highlight_{index:03d}.mp4"
                )

                progress.progress(
                    70 + int(30 * index / len(clips)),
                    text=(
                        f"Exporting highlight "
                        f"{index}/{len(clips)}..."
                    )
                )

                cut(
                    source,
                    output,
                    clip["start"],
                    30,
                    vertical
                )

                clip["path"] = output
                clip["duration"] = 30.0
                made.append(clip)

            progress.empty()

            if not made:
                st.error("No clips were exported.")
                st.stop()

            st.success(
                f"Created {len(made)} highlight clip(s)."
            )
            st.caption(
                f"Video duration: {duration:.1f}s · "
                f"Scene changes: {scene_count} · "
                f"Whisper transcript segments: {len(transcript)}"
            )

            for index, clip in enumerate(made, 1):
                with st.expander(
                    f"Clip {index} · "
                    f"{clip['start']:.1f}s–"
                    f"{clip['start'] + 30:.1f}s · "
                    f"Score {clip['score']:.0f}/100",
                    expanded=index == 1
                ):
                    st.video(str(clip["path"]))
                    st.write(
                        "Selected using combined audio intensity, "
                        "visual scene activity"
                        + (
                            " and speech signals."
                            if transcript else "."
                        )
                    )
                    st.download_button(
                        "Download MP4",
                        clip["path"].read_bytes(),
                        file_name=clip["path"].name,
                        mime="video/mp4",
                        key=f"download_{index}"
                    )

            archive = output_directory / "all_highlights.zip"

            with zipfile.ZipFile(
                archive,
                "w",
                zipfile.ZIP_DEFLATED
            ) as zip_file:
                for clip in made:
                    zip_file.write(
                        clip["path"],
                        arcname=clip["path"].name
                    )

            st.download_button(
                "Download all clips (ZIP)",
                archive.read_bytes(),
                file_name="all_highlights.zip",
                mime="application/zip"
            )

    except Exception as error:
        st.error(f"Processing failed: {error}")
        st.exception(error)


st.divider()
st.caption(
    "Highlight ranking is heuristic, not a guarantee of virality. "
    "Preview clips before publishing."
)
