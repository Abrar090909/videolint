"""FFmpeg-backed media measurements. All times use milliseconds."""
from __future__ import annotations

import array
import math
import os
import re
import subprocess
import sys
from pathlib import Path

try:
    import imageio_ffmpeg
except ImportError:
    imageio_ffmpeg = None


def ffmpeg() -> str:
    configured = os.environ.get("FFMPEG_BINARY")
    if configured:
        return configured
    if imageio_ffmpeg is None:
        raise RuntimeError("FFmpeg is unavailable. Run: python -m pip install -r requirements.txt")
    return imageio_ffmpeg.get_ffmpeg_exe()


def run(*args: str, timeout: int = 180) -> subprocess.CompletedProcess:
    return subprocess.run([ffmpeg(), "-hide_banner", "-nostdin", *map(str, args)],
                          capture_output=True, timeout=timeout)


def probe(path: Path) -> dict:
    result = run("-i", str(path), timeout=30)
    output = result.stderr.decode("utf-8", "replace")
    duration = re.search(r"Duration:\s*(\d+):(\d+):([\d.]+)", output)
    video = re.search(r"Stream #.*?: Video: ([^\r\n]+)", output)
    audio = re.search(r"Stream #.*?: Audio: ([^\r\n]+)", output)
    if not video or not duration:
        raise ValueError("This file has no readable video stream or duration.")
    width_height = re.search(r"\b(\d{2,5})x(\d{2,5})\b", video.group(1))
    fps = re.search(r"\b(\d+(?:\.\d+)?) fps\b", video.group(1))
    seconds = int(duration.group(1)) * 3600 + int(duration.group(2)) * 60 + float(duration.group(3))
    if seconds <= 0 or seconds > 3600:
        raise ValueError("Video duration must be between 0 and 60 minutes.")
    return {
        "durationMs": round(seconds * 1000), "width": int(width_height.group(1)) if width_height else None,
        "height": int(width_height.group(2)) if width_height else None,
        "fps": float(fps.group(1)) if fps else None, "hasAudio": bool(audio),
        "videoCodec": video.group(1).split(",")[0],
        "audioCodec": audio.group(1).split(",")[0] if audio else None,
    }


def _issue(kind: str, category: str, start: int, end: int, severity: str,
           title: str, explanation: str, metrics: dict, action: str) -> dict:
    return {"id": f"{kind}_{start}", "type": kind, "category": category,
            "startMs": start, "endMs": end, "severity": severity, "confidence": 0.98,
            "title": title, "explanation": explanation, "source": "rule",
            "evidence": {"metrics": metrics}, "suggestedAction": action}


def _runs(flags: list[bool], step_ms: int, minimum_ms: int) -> list[tuple[int, int]]:
    out = []
    start = None
    for i in range(len(flags) + 1):
        active = i < len(flags) and flags[i]
        if active and start is None:
            start = i * step_ms
        elif not active and start is not None:
            end = i * step_ms
            if end - start >= minimum_ms:
                out.append((start, end))
            start = None
    return out


def analyze_frames(path: Path, duration_ms: int) -> tuple[list[dict], list[dict]]:
    # A fixed small frame gives consistent measurements without loading the whole video.
    width, height, fps = 160, 90, 2
    frame_size = width * height
    proc = subprocess.Popen([ffmpeg(), "-hide_banner", "-loglevel", "error", "-nostdin",
                             "-i", str(path), "-vf", f"fps={fps},scale={width}:{height}",
                             "-f", "rawvideo", "-pix_fmt", "gray", "-"],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    brightness, changes = [], []
    previous = None
    try:
        while True:
            frame = proc.stdout.read(frame_size)
            if len(frame) < frame_size:
                break
            brightness.append(sum(frame) / frame_size)
            changes.append(sum(abs(a - b) for a, b in zip(frame, previous)) / frame_size
                           if previous is not None else 255)
            previous = frame
            if len(brightness) > 7200:
                raise ValueError("Too many frames to analyze.")
    finally:
        proc.stdout.close()
    err = proc.stderr.read().decode("utf-8", "replace")
    proc.stderr.close()
    if proc.wait(timeout=30) != 0 or not brightness:
        raise ValueError(f"Frame decoding failed: {err[-300:]}")
    issues = []
    for start, end in _runs([v < 5 for v in brightness], 500, 500):
        # Ignore black lead-in/out shorter than 1.2s as likely intentional fades.
        if (start == 0 or end >= duration_ms - 500) and end - start < 1200:
            continue
        issues.append(_issue("black_frames", "technical", start, min(end, duration_ms),
            "error", "Black video", f"The sampled frames are almost black for {(end-start)/1000:.1f} seconds.",
            {"meanLumaThreshold": 5, "sampleFps": fps}, "Inspect this section for a missing shot or unintended gap."))
    for start, end in _runs([d < 0.25 and brightness[i] >= 5 for i, d in enumerate(changes)], 500, 2500):
        issues.append(_issue("frozen_frames", "technical", max(0, start - 500), min(end, duration_ms),
            "warning", "Possible frozen picture", f"Almost no sampled pixel change for {(end-start+500)/1000:.1f} seconds.",
            {"meanPixelChangeThreshold": 0.25, "sampleFps": fps}, "Check whether this still image is intentional."))
    shots = []
    boundaries = [0] + [i * 500 for i, change in enumerate(changes) if i and change >= 18]
    boundaries.append(duration_ms)
    for start, end in zip(boundaries, boundaries[1:]):
        if end > start:
            shots.append({"startMs": start, "endMs": end})
    return issues, shots


def analyze_audio(path: Path, duration_ms: int, has_audio: bool) -> tuple[list[dict], list[dict]]:
    if not has_audio:
        return [_issue("missing_audio", "audio", 0, duration_ms, "warning", "No audio stream",
            "The video contains no audio stream.", {}, "Confirm that a silent export was intended.")], []
    rate, window_samples = 16000, 4000
    proc = subprocess.Popen([ffmpeg(), "-hide_banner", "-loglevel", "error", "-nostdin",
                             "-i", str(path), "-vn", "-ac", "1", "-ar", str(rate),
                             "-f", "s16le", "-acodec", "pcm_s16le", "-"],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    windows = []
    try:
        while True:
            raw = proc.stdout.read(window_samples * 2)
            if len(raw) < window_samples * 2:
                break
            samples = array.array("h")
            samples.frombytes(raw)
            if sys.byteorder != "little":
                samples.byteswap()
            peak = max(abs(x) for x in samples) / 32768
            rms = math.sqrt(sum(x*x for x in samples) / len(samples)) / 32768
            clipped = sum(abs(x) >= 32700 for x in samples) / len(samples)
            start = len(windows) * 250
            windows.append({"startMs": start, "endMs": min(start + 250, duration_ms),
                            "rms": round(rms, 5), "peak": round(peak, 5),
                            "clippedFraction": round(clipped, 5), "silence": rms < 0.008})
            if len(windows) > 14400:
                raise ValueError("Too much audio to analyze.")
    finally:
        proc.stdout.close()
    err = proc.stderr.read().decode("utf-8", "replace")
    proc.stderr.close()
    if proc.wait(timeout=30) != 0:
        raise ValueError(f"Audio decoding failed: {err[-300:]}")
    issues = []
    for start, end in _runs([w["silence"] for w in windows], 250, 1500):
        issues.append(_issue("silence", "audio", start, min(end, duration_ms), "warning", "Extended silence",
            f"Audio RMS stays below −42 dBFS for {(end-start)/1000:.1f} seconds.",
            {"rmsThreshold": 0.008, "durationSeconds": round((end-start)/1000, 2)},
            "Listen here and trim the gap if it interrupts the edit."))
    for start, end in _runs([w["clippedFraction"] >= 0.005 for w in windows], 250, 250):
        issues.append(_issue("audio_clipping", "audio", start, min(end, duration_ms), "error", "Audio clipping",
            "Multiple audio samples reach the digital ceiling in this window.",
            {"clippedSampleFraction": round(max(w["clippedFraction"] for w in windows[start//250:end//250]), 4)},
            "Lower the source or mix level and re-export."))
    return issues, windows


def extract_speech_audio(path: Path, destination: Path) -> None:
    result = run("-y", "-i", str(path), "-vn", "-ac", "1", "-ar", "16000",
                 "-b:a", "48k", "-c:a", "aac", str(destination), timeout=180)
    if result.returncode:
        stderr_tail = result.stderr.decode("utf-8", "replace")[-300:].strip()
        raise ValueError(f"Could not extract speech audio. ffmpeg: {stderr_tail}")


def sample_frames(path: Path, duration_ms: int, destination: Path,
                  count: int = 6, width: int = 512) -> list[dict]:
    """Extract *count* representative JPEG frames spread evenly across the video.

    Returns a list of ``{"timestampMs": int, "path": Path}`` sorted by time.
    """
    if duration_ms <= 0 or count < 1:
        return []
    destination.mkdir(parents=True, exist_ok=True)
    # Spread timestamps evenly, avoiding the very first and last frame
    step = duration_ms / (count + 1)
    timestamps = [round(step * (i + 1)) for i in range(count)]
    frames = []
    for idx, ts_ms in enumerate(timestamps):
        out = destination / f"frame_{idx:03d}.jpg"
        ts_sec = ts_ms / 1000
        result = run(
            "-y", "-ss", f"{ts_sec:.3f}", "-i", str(path),
            "-frames:v", "1", "-vf", f"scale={width}:-2",
            "-q:v", "3", str(out), timeout=30,
        )
        if result.returncode == 0 and out.is_file() and out.stat().st_size > 0:
            frames.append({"timestampMs": ts_ms, "path": out})
    return frames
