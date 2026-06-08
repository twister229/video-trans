from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path

from app.config import get_config
from app.jobs import Job
from app.stages import StageError


async def synthesize_segment(text: str, voice: str, out_path: Path, *, retries: int = 3) -> None:
    delay = 0.5
    for attempt in range(retries):
        try:
            import edge_tts

            await edge_tts.Communicate(text, voice).save(str(out_path))
            return
        except Exception as exc:
            if attempt == retries - 1:
                raise StageError(f"edge-tts failed after {retries} attempts: {exc}") from exc
            await asyncio.sleep(delay)
            delay *= 2


async def synthesize(segments: list[dict], voice: str, work_dir: Path, *, concurrency: int) -> list[Path]:
    semaphore = asyncio.Semaphore(concurrency)
    outputs = [work_dir / f"seg_{index}.mp3" for index in range(len(segments))]

    async def run_one(index: int, segment: dict) -> None:
        async with semaphore:
            try:
                await synthesize_segment(segment["text"], voice, outputs[index])
            except StageError as exc:
                raise StageError(f"segment {index + 1}: {exc}") from exc

    await asyncio.gather(*(run_one(index, segment) for index, segment in enumerate(segments)))
    return outputs


def dub_job(job: Job) -> None:
    config = get_config()
    segments = json.loads((job.dir / "vi_segments.json").read_text(encoding="utf-8"))
    if not segments:
        _make_silence(job.dir / "dub.wav", duration=1.0)
        return
    clips = asyncio.run(synthesize(segments, job.voice, job.dir, concurrency=config.max_tts_concurrency))
    _assemble_timeline(clips, segments, job.dir / "dub.wav")


def _assemble_timeline(clips: list[Path], segments: list[dict], out_path: Path) -> None:
    command = ["ffmpeg", "-y"]
    filter_parts = []
    mix_inputs = []
    for index, (clip, segment) in enumerate(zip(clips, segments, strict=True)):
        command += ["-i", str(clip)]
        delay_ms = int(float(segment["start"]) * 1000)
        filter_parts.append(f"[{index}:a]adelay={delay_ms}:all=1[a{index}]")
        mix_inputs.append(f"[a{index}]")
    filter_parts.append(f"{''.join(mix_inputs)}amix=inputs={len(clips)}:duration=longest:dropout_transition=0[out]")
    command += ["-filter_complex", ";".join(filter_parts), "-map", "[out]", str(out_path)]
    _run_ffmpeg(command, "TTS timeline assembly failed")


def _make_silence(out_path: Path, *, duration: float) -> None:
    command = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "anullsrc=channel_layout=mono:sample_rate=16000",
        "-t",
        str(duration),
        str(out_path),
    ]
    _run_ffmpeg(command, "silence generation failed")


def _run_ffmpeg(command: list[str], label: str) -> None:
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise StageError("ffmpeg is not installed") from exc
    except subprocess.CalledProcessError as exc:
        raise StageError(f"{label}: {exc.stderr.strip()}") from exc
