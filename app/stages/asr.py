from __future__ import annotations

import json
from pathlib import Path

from app.jobs import Job
from app.stages import StageError


def transcribe(audio_path: Path, model: str) -> list[dict]:
    try:
        import mlx_whisper

        result = mlx_whisper.transcribe(
            str(audio_path),
            path_or_hf_repo=model,
            word_timestamps=False,
        )
    except Exception as exc:
        raise StageError(f"transcription failed: {exc}") from exc

    segments = []
    for segment in result.get("segments", []):
        text = str(segment.get("text", "")).strip()
        if text:
            segments.append(
                {
                    "start": float(segment["start"]),
                    "end": float(segment["end"]),
                    "text": text,
                }
            )
    return segments


def transcribe_job(job: Job) -> None:
    segments = transcribe(job.dir / "audio.wav", job.model)
    (job.dir / "segments.json").write_text(json.dumps(segments, indent=2), encoding="utf-8")
