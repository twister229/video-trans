from __future__ import annotations

from collections.abc import Callable

from app.jobs import Job, STAGES, save
from app.stages import StageError
from app.stages.asr import transcribe_job
from app.stages.extract import extract_audio
from app.stages.mux import mux_job
from app.stages.translate import translate_job
from app.stages.tts import dub_job


STAGE_FUNCTIONS: dict[str, str] = {
    "extract": "extract_audio",
    "transcribe": "transcribe_job",
    "translate": "translate_job",
    "dub": "dub_job",
    "mux": "mux_job",
}


def run_pipeline(job: Job) -> None:
    job.status = "running"
    save(job)
    for stage in STAGES:
        if stage in job.completed_stages:
            continue
        job.stage = stage
        job.message = f"Running {stage}"
        job.progress = int(len(job.completed_stages) / len(STAGES) * 100)
        save(job)
        try:
            # Resolve the stage function from this module at call time so it
            # honors monkeypatching and swap points instead of a frozen ref.
            stage_fn: Callable[[Job], None] = globals()[STAGE_FUNCTIONS[stage]]
            stage_fn(job)
        except StageError as exc:
            job.status = "error"
            job.error = str(exc)
            job.message = f"Failed at {stage}"
            save(job)
            return
        job.completed_stages.append(stage)
        job.progress = int(len(job.completed_stages) / len(STAGES) * 100)
        job.message = f"Completed {stage}"
        save(job)
    job.status = "done"
    job.stage = None
    job.progress = 100
    job.message = "Done"
    save(job)
