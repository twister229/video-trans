from __future__ import annotations

import json

import pytest

import app.pipeline as pipeline
from app.jobs import Job


def make_job(tmp_path, completed):
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    return Job(
        id="job",
        status="running",
        stage=None,
        completed_stages=list(completed),
        progress=0,
        message="",
        error=None,
        filename="video.mp4",
        voice="voice",
        model="model",
        created_at="now",
        dir=job_dir,
    )


def test_run_pipeline_skips_completed_stages(tmp_path, monkeypatch):
    ran = []
    for name in ("extract_audio", "transcribe_job", "translate_job", "dub_job", "mux_job"):
        monkeypatch.setattr(pipeline, name, (lambda n: lambda job: ran.append(n))(name))

    job = make_job(tmp_path, completed=["extract", "transcribe"])
    pipeline.run_pipeline(job)

    assert ran == ["translate_job", "dub_job", "mux_job"]
    assert job.status == "done"
    assert job.completed_stages == ["extract", "transcribe", "translate", "dub", "mux"]


def test_run_pipeline_stops_and_records_failing_stage(tmp_path, monkeypatch):
    from app.stages import StageError

    monkeypatch.setattr(pipeline, "extract_audio", lambda job: None)

    def boom(job):
        raise StageError("transcribe blew up")

    monkeypatch.setattr(pipeline, "transcribe_job", boom)

    job = make_job(tmp_path, completed=[])
    pipeline.run_pipeline(job)

    assert job.status == "error"
    assert job.stage == "transcribe"
    assert "transcribe blew up" in job.error
    saved = json.loads((job.dir / "job.json").read_text(encoding="utf-8"))
    assert saved["status"] == "error"
