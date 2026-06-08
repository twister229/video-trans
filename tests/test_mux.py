from __future__ import annotations

import json

import pytest

from app.jobs import Job
from app.stages import StageError
from app.stages.mux import mux_job


def make_job(tmp_path):
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    return Job(
        id="job",
        status="running",
        stage=None,
        completed_stages=[],
        progress=0,
        message="",
        error=None,
        filename="video.mp4",
        voice="voice",
        model="model",
        created_at="now",
        dir=job_dir,
    )


def test_mux_raises_on_bad_input(tmp_path):
    job = make_job(tmp_path)
    # A non-video file as input + empty dub makes ffmpeg exit nonzero.
    job.input_path.write_text("not a video", encoding="utf-8")
    (job.dir / "dub.wav").write_text("not audio", encoding="utf-8")
    (job.dir / "vi_segments.json").write_text("[]", encoding="utf-8")

    with pytest.raises(StageError):
        mux_job(job)


@pytest.mark.slow
def test_mux_with_subtitles_produces_playable_output(sample_video, tmp_path):
    job = make_job(tmp_path)
    job.input_path.write_bytes(sample_video.read_bytes())
    # Reuse the source audio as a stand-in dub track.
    import subprocess

    subprocess.run(
        ["ffmpeg", "-y", "-i", str(job.input_path), "-ar", "16000", "-ac", "1", str(job.dir / "dub.wav")],
        check=True,
        capture_output=True,
    )
    segments = [{"start": 0.0, "end": 1.0, "text": "Xin chào"}]
    (job.dir / "vi_segments.json").write_text(json.dumps(segments), encoding="utf-8")

    mux_job(job)

    assert (job.dir / "output.mp4").stat().st_size > 0
    assert (job.dir / "subtitles.vi.srt").read_text(encoding="utf-8").strip()
