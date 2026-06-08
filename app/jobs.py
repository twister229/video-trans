from __future__ import annotations

import json
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, Literal
from uuid import uuid4

from app.config import Config, get_config


STAGES = ["extract", "transcribe", "translate", "dub", "mux"]
Status = Literal["queued", "running", "done", "error"]
Runner = Callable[["Job"], None]


class JobBusyError(RuntimeError):
    pass


@dataclass
class Job:
    id: str
    status: Status
    stage: str | None
    completed_stages: list[str]
    progress: int
    message: str
    error: str | None
    filename: str
    voice: str
    model: str
    created_at: str
    dir: Path
    input_name: str = "input.mp4"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "status": self.status,
            "stage": self.stage,
            "completed_stages": self.completed_stages,
            "progress": self.progress,
            "message": self.message,
            "error": self.error,
            "filename": self.filename,
            "voice": self.voice,
            "model": self.model,
            "created_at": self.created_at,
            "input_name": self.input_name,
        }

    @classmethod
    def from_dict(cls, data: dict, job_dir: Path) -> "Job":
        return cls(
            id=data["id"],
            status=data["status"],
            stage=data.get("stage"),
            completed_stages=list(data.get("completed_stages", [])),
            progress=int(data.get("progress", 0)),
            message=data.get("message", ""),
            error=data.get("error"),
            filename=data.get("filename", ""),
            voice=data.get("voice", get_config().tts_voice),
            model=data.get("model", get_config().whisper_model),
            created_at=data.get("created_at", datetime.now(UTC).isoformat()),
            dir=job_dir,
            input_name=data.get("input_name", "input.mp4"),
        )

    @property
    def input_path(self) -> Path:
        return self.dir / self.input_name


def save(job: Job) -> None:
    job.dir.mkdir(parents=True, exist_ok=True)
    (job.dir / "job.json").write_text(json.dumps(job.to_dict(), indent=2), encoding="utf-8")


def load(job_id: str, *, config: Config | None = None) -> Job | None:
    config = config or get_config()
    job_dir = config.data_dir / job_id
    job_file = job_dir / "job.json"
    if not job_file.exists():
        return None
    return Job.from_dict(json.loads(job_file.read_text(encoding="utf-8")), job_dir)


def default_runner(job: Job) -> None:
    from app.pipeline import run_pipeline

    run_pipeline(job)


@dataclass
class JobManager:
    config: Config = field(default_factory=get_config)
    runner: Runner = default_runner
    executor: ThreadPoolExecutor = field(default_factory=lambda: ThreadPoolExecutor(max_workers=1))
    current_job_id: str | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    def create_from_path(self, source: Path, *, filename: str, voice: str, model: str) -> Job:
        with self.lock:
            if self._active_job_locked() is not None:
                raise JobBusyError("A job is already running")

            job_id = uuid4().hex
            job_dir = self.config.data_dir / job_id
            job_dir.mkdir(parents=True, exist_ok=True)
            input_name = f"input{source.suffix or '.mp4'}"
            shutil.copyfile(source, job_dir / input_name)
            job = Job(
                id=job_id,
                status="queued",
                stage=None,
                completed_stages=[],
                progress=0,
                message="Queued",
                error=None,
                filename=filename,
                voice=voice,
                model=model,
                created_at=datetime.now(UTC).isoformat(),
                dir=job_dir,
                input_name=input_name,
            )
            save(job)
            self.current_job_id = job.id
            self.executor.submit(self._run, job)
            return job

    def get(self, job_id: str) -> Job | None:
        return load(job_id, config=self.config)

    def recover_on_startup(self) -> None:
        self.config.data_dir.mkdir(parents=True, exist_ok=True)
        for job_file in self.config.data_dir.glob("*/job.json"):
            job = Job.from_dict(json.loads(job_file.read_text(encoding="utf-8")), job_file.parent)
            if job.status == "running":
                with self.lock:
                    if self._active_job_locked() is None:
                        self.current_job_id = job.id
                        self.executor.submit(self._run, job)
                break

    def _run(self, job: Job) -> None:
        job.status = "running"
        save(job)
        try:
            self.runner(job)
        except Exception as exc:
            job.status = "error"
            job.stage = job.stage or "job"
            job.error = str(exc)
            job.message = "Job failed"
            save(job)
        finally:
            latest = load(job.id, config=self.config)
            with self.lock:
                if latest is None or latest.status in {"done", "error"}:
                    self.current_job_id = None

    def _active_job_locked(self) -> Job | None:
        if self.current_job_id is None:
            return None
        job = load(self.current_job_id, config=self.config)
        if job is None or job.status in {"done", "error"}:
            self.current_job_id = None
            return None
        return job
