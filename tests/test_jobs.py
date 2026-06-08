from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from app.config import Config
from app.jobs import Job, JobBusyError, JobManager, load, save


def make_config(tmp_path: Path) -> Config:
    return Config(data_dir=tmp_path / "jobs")


def make_source(tmp_path: Path) -> Path:
    source = tmp_path / "video.mp4"
    source.write_bytes(b"fake video")
    return source


def test_save_load_round_trips(tmp_path):
    config = make_config(tmp_path)
    job = Job(
        id="abc",
        status="running",
        stage="translate",
        completed_stages=["extract", "transcribe"],
        progress=40,
        message="Translating",
        error=None,
        filename="original.mp4",
        voice="vi-VN-HoaiMyNeural",
        model="large-v3",
        created_at="2026-06-08T00:00:00+00:00",
        dir=config.data_dir / "abc",
        input_name="input.mp4",
    )

    save(job)
    loaded = load("abc", config=config)

    assert loaded is not None
    assert loaded.to_dict() == job.to_dict()
    assert loaded.dir == job.dir


def test_busy_guard_blocks_second_job(tmp_path):
    started = threading.Event()
    release = threading.Event()

    def runner(_job):
        started.set()
        release.wait(timeout=3)

    manager = JobManager(config=make_config(tmp_path), runner=runner)
    first = manager.create_from_path(make_source(tmp_path), filename="one.mp4", voice="voice", model="model")
    assert started.wait(timeout=3)

    with pytest.raises(JobBusyError):
        manager.create_from_path(make_source(tmp_path), filename="two.mp4", voice="voice", model="model")

    release.set()
    manager.executor.shutdown(wait=True)
    assert load(first.id, config=manager.config) is not None


def test_recover_on_startup_resubmits_running_job(tmp_path):
    config = make_config(tmp_path)
    job_dir = config.data_dir / "abc"
    job_dir.mkdir(parents=True)
    (job_dir / "job.json").write_text(
        json.dumps(
            {
                "id": "abc",
                "status": "running",
                "stage": "translate",
                "completed_stages": ["extract", "transcribe"],
                "progress": 40,
                "message": "Translating",
                "error": None,
                "filename": "video.mp4",
                "voice": "voice",
                "model": "model",
                "created_at": "2026-06-08T00:00:00+00:00",
                "input_name": "input.mp4",
            }
        ),
        encoding="utf-8",
    )
    seen = []

    def runner(job):
        seen.append((job.id, job.completed_stages))
        job.status = "done"
        save(job)

    manager = JobManager(config=config, runner=runner)
    manager.recover_on_startup()
    manager.executor.shutdown(wait=True)

    assert seen == [("abc", ["extract", "transcribe"])]
