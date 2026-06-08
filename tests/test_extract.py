from app.jobs import Job
from app.stages import StageError
from app.stages.extract import extract_audio

import pytest


def make_job(tmp_path, source):
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    target = job_dir / "input.mp4"
    target.write_bytes(source.read_bytes())
    return Job(
        id="job",
        status="queued",
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


def test_extract_audio(sample_video, tmp_path):
    job = make_job(tmp_path, sample_video)

    extract_audio(job)

    assert (job.dir / "audio.wav").stat().st_size > 0


def test_extract_bad_input_raises(tmp_path):
    source = tmp_path / "bad.mp4"
    source.write_text("not video", encoding="utf-8")
    job = make_job(tmp_path, source)

    with pytest.raises(StageError):
        extract_audio(job)
