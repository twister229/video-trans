from __future__ import annotations

import json

import pytest

from app.jobs import Job
from app.pipeline import run_pipeline


@pytest.mark.slow
def test_pipeline_empty_audio_completes_with_silence(monkeypatch, sample_video, tmp_path):
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    (job_dir / "input.mp4").write_bytes(sample_video.read_bytes())

    job = Job(
        id="job",
        status="queued",
        stage=None,
        completed_stages=[],
        progress=0,
        message="",
        error=None,
        filename="video.mp4",
        voice="vi-VN-HoaiMyNeural",
        model="model",
        created_at="now",
        dir=job_dir,
    )

    monkeypatch.setattr("app.pipeline.transcribe_job", lambda job: (job.dir / "segments.json").write_text("[]", encoding="utf-8"))
    monkeypatch.setattr("app.pipeline.translate_job", lambda job: (job.dir / "vi_segments.json").write_text("[]", encoding="utf-8"))

    run_pipeline(job)

    data = json.loads((job.dir / "job.json").read_text(encoding="utf-8"))
    assert data["status"] == "done"
    assert (job.dir / "output.mp4").exists()
