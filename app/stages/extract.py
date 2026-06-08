from __future__ import annotations

import subprocess
from pathlib import Path

from app.jobs import Job
from app.stages import StageError


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def extract_audio(job: Job) -> None:
    input_path = job.input_path
    if not _inside(input_path, job.dir):
        raise StageError("Input path is outside the job directory")
    output_path = job.dir / "audio.wav"
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-ar",
        "16000",
        "-ac",
        "1",
        str(output_path),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise StageError("ffmpeg is not installed") from exc
    except subprocess.CalledProcessError as exc:
        raise StageError(f"ffmpeg audio extract failed: {exc.stderr.strip()}") from exc
