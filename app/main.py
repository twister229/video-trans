from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import get_config
from app.jobs import JobBusyError, JobManager


app = FastAPI(title="Video Trans")
app.mount("/static", StaticFiles(directory="app/static"), name="static")
manager = JobManager()


@app.on_event("startup")
def recover_jobs() -> None:
    manager.recover_on_startup()


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return Path("app/static/index.html").read_text(encoding="utf-8")


@app.post("/jobs")
async def create_job(
    file: UploadFile = File(...),
    voice: str = Form(get_config().tts_voice),
    model: str = Form(get_config().whisper_model),
):
    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)
    try:
        job = manager.create_from_path(tmp_path, filename=file.filename or "video", voice=voice, model=model)
    except JobBusyError as exc:
        tmp_path.unlink(missing_ok=True)
        return JSONResponse(status_code=409, content={"error": str(exc)})
    finally:
        tmp_path.unlink(missing_ok=True)
    return {"id": job.id}


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job.to_dict()


@app.get("/jobs/{job_id}/file/{kind}")
def get_job_file(job_id: str, kind: str):
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    paths = {
        "video": job.dir / "output.mp4",
        "srt": job.dir / "subtitles.vi.srt",
    }
    if kind not in paths:
        raise HTTPException(status_code=404, detail="Unknown file kind")
    path = paths[kind]
    if job.status != "done" or not path.exists():
        raise HTTPException(status_code=404, detail="File not ready")
    return FileResponse(path)
