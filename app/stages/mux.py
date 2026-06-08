from __future__ import annotations

import json
import subprocess

from app.jobs import Job
from app.srt import segments_to_srt
from app.stages import StageError


def mux_job(job: Job) -> None:
    vi_segments = json.loads((job.dir / "vi_segments.json").read_text(encoding="utf-8"))
    srt_path = job.dir / "subtitles.vi.srt"
    srt_path.write_text(segments_to_srt(vi_segments), encoding="utf-8")
    has_audio = _has_audio(job)
    # ffmpeg rejects an empty subtitle file as an input, so only attach the
    # subtitle stream when there is at least one translated segment.
    has_subs = bool(vi_segments)

    command = ["ffmpeg", "-y", "-i", str(job.input_path), "-i", str(job.dir / "dub.wav")]
    if has_subs:
        command += ["-i", str(srt_path)]

    if has_audio:
        command += [
            "-filter_complex",
            "[0:a]volume=0.15[bg];[bg][1:a]amix=inputs=2:duration=longest:dropout_transition=0[a]",
            "-map",
            "0:v",
            "-map",
            "[a]",
        ]
    else:
        command += ["-map", "0:v", "-map", "1:a"]

    if has_subs:
        command += ["-map", "2", "-c:s", "mov_text"]

    command += ["-c:v", "copy", str(job.dir / "output.mp4")]
    _run(command, "mux failed")


def _has_audio(job: Job) -> bool:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a",
        "-show_entries",
        "stream=index",
        "-of",
        "csv=p=0",
        str(job.input_path),
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise StageError("ffprobe is not installed") from exc
    except subprocess.CalledProcessError:
        return False
    return bool(result.stdout.strip())


def _run(command: list[str], label: str) -> None:
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise StageError("ffmpeg is not installed") from exc
    except subprocess.CalledProcessError as exc:
        raise StageError(f"{label}: {exc.stderr.strip()}") from exc
