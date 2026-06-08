# Plan: Video → Vietnamese subtitle + dubbing website

**Date:** 2026-06-08
**Goal:** A single-page local web app where the user uploads a video, picks a Vietnamese voice and whisper model, and gets back a dubbed `.mp4` (Vietnamese voiceover over ducked original audio + soft Vietnamese subtitles) plus a downloadable `.srt`.
**Architecture:** Python + FastAPI. Upload kicks off a 5-stage pipeline (extract → transcribe → translate → dub → mux) run in a single-worker background thread so the status endpoint stays responsive. Browser polls for stage progress. One job at a time.
**Tech stack:** Python 3.12 (pinned via uv), FastAPI + uvicorn, mlx-whisper (ASR), openai client (OpenAI-compatible LLM translate), edge-tts (Vietnamese TTS), ffmpeg via subprocess, pytest.

**Source design doc:** `~/.gstack/projects/video-trans/20260608-design-video-trans-vi.md` (CEO + Design + Eng reviews all CLEAR).

## Prerequisites (one-time, before Task 1)

- `brew install ffmpeg` — pipeline shells out to ffmpeg/ffprobe.
- `curl -LsSf https://astral.sh/uv/install.sh | sh` — install uv.
- Project uses **Python 3.12**, NOT the system 3.14 (ML wheels lag). `uv` pins it.

## Files

| File | Action | Responsibility |
|------|--------|----------------|
| pyproject.toml | create | Deps + Python 3.12 pin |
| .env.example | create | Config template (LLM endpoint, defaults) |
| .gitignore | modify | Ignore data/, .env, .venv |
| app/config.py | create | Load env-var config |
| app/jobs.py | create | Job model, state machine, disk persistence, ThreadPool(1) runner, 409 guard, resume |
| app/pipeline.py | create | Orchestrator: run stages in order, resume from last completed stage |
| app/stages/extract.py | create | ffmpeg audio extract (arg-array) |
| app/stages/asr.py | create | transcribe() — mlx-whisper behind swap interface |
| app/stages/translate.py | create | translate() — OpenAI-compat + line-count validation + per-segment fallback |
| app/stages/tts.py | create | synthesize() — edge-tts, retry/backoff, ≤4 concurrent |
| app/stages/mux.py | create | ffmpeg mux: VI audio ducked over original + soft .srt |
| app/srt.py | create | segments ↔ SRT string |
| app/main.py | create | FastAPI app, routes, static serving |
| app/static/index.html | create | Single page, 2 states, polling, stage checklist, inline player |
| tests/conftest.py | create | Fixtures incl. tiny sample clip |
| tests/test_extract.py | create | extract_audio unit tests |
| tests/test_translate.py | create | translate count-validation + fallback + errors |
| tests/test_tts.py | create | synthesize retry + concurrency cap |
| tests/test_srt.py | create | SRT round-trip |
| tests/test_jobs.py | create | 409 guard, state transitions, resume |
| tests/test_e2e.py | create | Full pipeline on a tiny clip |
| README.md | modify | Install + run instructions |

## Conventions

- All ffmpeg/ffprobe calls: `subprocess.run([...], check=True)` with an **arg array**, never a shell string. User filenames are never passed to ffmpeg — uploads are renamed to `input<ext>` inside the job dir on arrival.
- Job dirs: `data/jobs/<job_id>/` containing `input.<ext>`, `audio.wav`, `segments.json`, `vi_segments.json`, `dub.wav`, `output.mp4`, `subtitles.vi.srt`, `job.json`.
- A "segment" is `{"start": float, "end": float, "text": str}` (seconds).
- Stage functions take and mutate a `Job`, write their output artifact to the job dir, and raise on failure. The orchestrator catches, records the failing stage, sets status=error.

## Tasks

### Task 1: Project scaffold + config

**Files:** `pyproject.toml` (create), `.env.example` (create), `.gitignore` (modify), `app/config.py` (create), `app/__init__.py` (create), `app/stages/__init__.py` (create)

Create `pyproject.toml` pinning Python 3.12 and deps:

```toml
[project]
name = "video-trans"
version = "0.1.0"
requires-python = ">=3.12,<3.13"
dependencies = [
  "fastapi>=0.115",
  "uvicorn[standard]>=0.32",
  "python-multipart>=0.0.12",
  "mlx-whisper>=0.4",
  "edge-tts>=6.1",
  "openai>=1.50",
  "pytest>=8.3",
  "httpx>=0.27",
]
```

`.env.example`:

```
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
WHISPER_MODEL=mlx-community/whisper-large-v3-mlx
TTS_VOICE=vi-VN-HoaiMyNeural
MAX_TTS_CONCURRENCY=4
DATA_DIR=data/jobs
```

Append to `.gitignore`: `data/`, `.env`, `.venv/`, `__pycache__/`, `*.pyc`.

`app/config.py` — a frozen dataclass `Config` loaded from env via `os.environ.get` with the defaults above. Single `get_config()` function returning a cached instance. `app/__init__.py` and `app/stages/__init__.py` are empty.

**Verification:**
```bash
uv sync && uv run python -c "from app.config import get_config; print(get_config().tts_voice)"
# expects: vi-VN-HoaiMyNeural
```

### Task 2: SRT round-trip

**Files:** `app/srt.py` (create), `tests/test_srt.py` (create), `tests/__init__.py` (create)

`app/srt.py` with two functions:
- `segments_to_srt(segments: list[dict]) -> str` — standard SRT: index, `HH:MM:SS,mmm --> HH:MM:SS,mmm`, text, blank line.
- `format_timestamp(seconds: float) -> str` — helper producing `HH:MM:SS,mmm`.

`tests/test_srt.py`: assert a 2-segment list produces exactly the expected SRT string, and that `format_timestamp(3661.5)` == `"01:01:01,500"`.

**Verification:**
```bash
uv run pytest tests/test_srt.py -q
# expects: 2 passed
```

### Task 3: Job model, state machine, persistence

**Files:** `app/jobs.py` (create), `tests/test_jobs.py` (create)

`app/jobs.py`:
- `STAGES = ["extract", "transcribe", "translate", "dub", "mux"]`.
- `Status` = Literal `"queued" | "running" | "done" | "error"`.
- `@dataclass Job`: `id`, `status`, `stage` (current/active stage or None), `completed_stages: list[str]`, `progress: int`, `message: str`, `error: str | None`, `filename: str`, `voice: str`, `model: str`, `created_at`, `dir: Path`. Method `to_dict()` for JSON.
- `save(job)` / `load(job_id)` reading/writing `<dir>/job.json`.
- `JobManager` holding a `ThreadPoolExecutor(max_workers=1)` and a single `current_job_id` field guarded by a `threading.Lock`.
  - `create(upload, voice, model) -> Job`: raises `JobBusyError` if a job is currently `running`. Otherwise allocate `id` (uuid4 hex), make dir, save the uploaded file as `input<ext>`, persist job, submit `run` to the executor.
  - `get(job_id) -> Job | None`.
  - `recover_on_startup()`: scan `DATA_DIR`, any job left `running` → resume by resubmitting to the executor (pipeline skips completed stages).

Do NOT implement the pipeline call yet — submit a placeholder `run_pipeline(job)` imported from `app.pipeline` (created in Task 8). To keep Task 3 testable in isolation, inject the runner: `JobManager(runner=run_pipeline)` defaulting to a no-op lambda in tests.

`tests/test_jobs.py`:
- creating a job while one is `running` raises `JobBusyError` (use a runner that blocks on an `threading.Event`).
- `save`/`load` round-trips all fields.
- a job written to disk with `status="running"` and `completed_stages=["extract"]` is picked up by `recover_on_startup()`.

**Verification:**
```bash
uv run pytest tests/test_jobs.py -q
# expects: 3 passed
```

### Task 4: Stage — extract audio

**Files:** `app/stages/extract.py` (create), `tests/test_extract.py` (create), `tests/conftest.py` (create)

`tests/conftest.py`: a `sample_video` fixture that generates a 2-second silent test clip with ffmpeg into a tmp dir:
`ffmpeg -f lavfi -i color=c=black:s=160x120:d=2 -f lavfi -i anullsrc -shortest <out>.mp4` (arg array). Skip the test with `pytest.skip` if ffmpeg is absent.

`app/stages/extract.py`: `extract_audio(job)` runs
`ffmpeg -y -i <input> -ar 16000 -ac 1 <dir>/audio.wav` as an arg array. Raises `StageError` (define in `app/stages/__init__.py` or a shared `errors.py`) on nonzero exit. Asserts the input path is inside the job dir (defense against traversal).

`tests/test_extract.py`:
- happy: sample_video → `audio.wav` exists and is nonempty.
- bad input: a text file renamed `.mp4` → raises `StageError`.

**Verification:**
```bash
uv run pytest tests/test_extract.py -q
# expects: 2 passed (or skipped if ffmpeg missing — install it first)
```

### Task 5: Stage — transcribe (ASR, mlx-whisper behind swap point)

**Files:** `app/stages/asr.py` (create)

`transcribe(audio_path: Path, model: str) -> list[dict]`: calls `mlx_whisper.transcribe(str(audio_path), path_or_hf_repo=model, word_timestamps=False)`, maps the returned `segments` to the `{start, end, text}` shape, strips whitespace, drops empty-text segments. This is the single swap point — the rest of the code only sees `transcribe()`, never mlx directly.

`transcribe_job(job)`: thin wrapper that calls `transcribe(job.dir/"audio.wav", job.model)`, writes `job.dir/"segments.json"`. Returns nothing; raises `StageError` wrapping any exception (model load fail, OOM).

No unit test asserts transcription *content* (non-deterministic, needs the model). Quality is covered by the eval suite (Task 13). A smoke test in test_e2e (Task 12) exercises the real path on the tiny clip.

**Verification:**
```bash
uv run python -c "import mlx_whisper; print('mlx-whisper import ok')"
# expects: mlx-whisper import ok  (confirms the dep resolves on this machine)
```

### Task 6: Stage — translate (count-validation + per-segment fallback)

**Files:** `app/stages/translate.py` (create), `tests/test_translate.py` (create)

`translate(segments: list[dict], *, client, model) -> list[dict]`:
- Build one prompt with stable numbered markers: each line `[[i]] <text>`.
- System prompt: "Translate each numbered line to natural Vietnamese. Return exactly one line per input, each prefixed with its same [[i]] marker. No commentary."
- Parse the response by `[[i]]` markers into `{i: vi_text}`.
- **Hard check:** if the number of parsed lines != len(segments), OR any index is missing, fall back to `_translate_one_by_one(segments, client, model)` (one API call per segment, guaranteed 1:1).
- Return new segments with original `start`/`end` and translated `text`.
- The OpenAI client is injected so tests can pass a fake.

`_translate_one_by_one`: loop, one chat call per segment, returns translated text.

`translate_job(job)`: builds the real `openai.OpenAI(base_url=, api_key=)` client from config, reads `segments.json`, writes `vi_segments.json`. Wraps API exceptions (5xx, timeout) in `StageError`.

`tests/test_translate.py` (fake client, no network):
- happy: fake returns N markered lines for N segments → batch path, 1:1, timestamps preserved.
- count mismatch: fake batch returns N-1 lines → triggers fallback → fallback fake returns 1 line each → final has N segments.
- empty input list → returns `[]`, no API call.
- client raises on batch AND fallback → `translate()` (via `translate_job`) surfaces `StageError`.

**Verification:**
```bash
uv run pytest tests/test_translate.py -q
# expects: 4 passed
```

### Task 7: Stage — dub (edge-tts, retry, ≤4 concurrent) + timeline

**Files:** `app/stages/tts.py` (create), `tests/test_tts.py` (create)

`synthesize_segment(text, voice, out_path, *, retries=3)`: async; calls `edge_tts.Communicate(text, voice).save(out_path)`; retry with exponential backoff (0.5s, 1s, 2s) on exception; after final failure raise `StageError` naming the segment index.

`synthesize(segments, voice, work_dir, *, concurrency)`: async; bound a `asyncio.Semaphore(concurrency)` so at most `concurrency` (default from config, 4) clips render at once; produce one `seg_<i>.mp3` per segment. Returns list of clip paths.

`dub_job(job)`: sync entrypoint (`asyncio.run`). After clips render, assemble a single `dub.wav` on the original timeline: for each segment, place its clip at `start`; if a clip is longer than `(next_start - start)`, speed it up with ffmpeg `atempo` (cap at 1.5x) else leave it; pad gaps with silence. Build via ffmpeg `adelay` + `amix` or `concat` with silence — produce `dub.wav` at the source duration. Injectable synth function so tests skip real edge-tts.

`tests/test_tts.py` (monkeypatch the synth coroutine, no network):
- happy: 3 segments → 3 clip paths produced.
- one segment's synth fails twice then succeeds → retried, no error.
- a segment fails all retries → `StageError` with the index.
- concurrency cap: with a synth that records max simultaneous in-flight, assert it never exceeds `concurrency`.

**Verification:**
```bash
uv run pytest tests/test_tts.py -q
# expects: 4 passed
```

### Task 8: Stage — mux (ducking) + pipeline orchestrator

**Files:** `app/stages/mux.py` (create), `app/pipeline.py` (create)

`mux_job(job)`: write `subtitles.vi.srt` from `vi_segments.json` (via `app.srt`). Then one ffmpeg arg-array call:
- inputs: `[0]=input video`, `[1]=dub.wav`, `[2]=subtitles.vi.srt`
- filter: `[0:a]volume=0.15[bg];[bg][1:a]amix=inputs=2:duration=longest:dropout_transition=0[a]`
- map: `-map 0:v -map "[a]" -map 2 -c:v copy -c:s mov_text` → `output.mp4`
- If the source has no audio track, skip the `[0:a]` branch and use dub.wav directly (probe with ffprobe). Raise `StageError` on nonzero exit.

`app/pipeline.py` — `run_pipeline(job)`:
```
ORDER = ["extract", "transcribe", "translate", "dub", "mux"]
fns = {extract: extract_audio, transcribe: transcribe_job, ...}
for stage in ORDER:
    if stage in job.completed_stages: continue      # resume
    job.stage = stage; job.status="running"; update progress; save(job)
    try: fns[stage](job)
    except StageError as e:
        job.status="error"; job.error=str(e); save(job); return
    job.completed_stages.append(stage); save(job)
job.status="done"; job.stage=None; job.progress=100; save(job)
```
Progress = `len(completed)/5 * 100`. Each stage updates `job.message` (e.g. "Transcribing…").

**Verification:**
```bash
uv run python -c "from app.pipeline import run_pipeline; from app.stages.mux import mux_job; print('pipeline wired')"
# expects: pipeline wired
```

### Task 9: FastAPI app + routes

**Files:** `app/main.py` (create)

`app/main.py`:
- `app = FastAPI()`. On startup event: `JobManager(runner=run_pipeline)` singleton + `manager.recover_on_startup()`.
- `POST /jobs` — `UploadFile` + form fields `voice`, `model`. Calls `manager.create(...)`. On `JobBusyError` return `JSONResponse(status_code=409, {"error": "A job is already running"})`. Else return `{"id": job.id}`.
- `GET /jobs/{job_id}` — return `job.to_dict()` (status, stage, completed_stages, progress, message, error). 404 if unknown.
- `GET /jobs/{job_id}/file/{kind}` — `kind` in `{"video","srt"}` → `FileResponse` of `output.mp4` / `subtitles.vi.srt`. 404 if not done / missing. Validate `kind` against an allowlist; never interpolate it into a path.
- `GET /` — serve `static/index.html`. Mount `app/static` at `/static`.

**Verification:**
```bash
uv run uvicorn app.main:app --port 8000 &
sleep 3
curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/        # expects 200
curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/jobs/nope # expects 404
kill %1
```

### Task 10: Frontend — single page, two states, polling

**Files:** `app/static/index.html` (create)

One self-contained HTML file (inline CSS + vanilla JS, no build step). Two views toggled by JS, matching design doc §5.5:

- **Pre-job:** drop zone (also a clickable `<label>` wrapping a hidden `<input type=file accept="video/*">` — never drag-only), `<select>` Voice (HoaiMy nữ / NamMinh nam → `vi-VN-HoaiMyNeural` / `vi-VN-NamMinhNeural`), `<select>` Model (small/medium/large-v3 → the mlx repo ids), a Translate button disabled until a file is chosen. Real `<label>`s, visible focus rings, ≥44px buttons.
- On submit: `POST /jobs` (FormData). On 409, show "A job is already running." On success, switch to processing view and start polling `GET /jobs/{id}` every 1500ms.
- **Processing:** filename + elapsed timer (client-side from submit). The 5-stage checklist: `✓` for each in `completed_stages`, `⟳` on `stage`, `○` otherwise. Stage labels in English+VI as in design doc.
- **Done:** `<video controls>` pointing at `/jobs/{id}/file/video`, download links for video + srt, "Translate another" resets to pre-job.
- **Error:** "Failed at: <stage>" + `error` text + "Start over".

**Verification:** Manual — `uv run uvicorn app.main:app` then open http://localhost:8000, confirm the page renders both selects and the disabled button enables after choosing a file. (Full flow verified in Task 12.)

### Task 11: Job manager wiring + concurrency integration test

**Files:** `app/jobs.py` (modify — swap default runner to real `run_pipeline`), `tests/test_jobs.py` (modify — add resume integration)

Wire `JobManager`'s default `runner` to `app.pipeline.run_pipeline` (keep injection for tests). Add a resume integration test: seed a job dir with `completed_stages=["extract","transcribe"]` and stub stage fns recording which ran; `run_pipeline` must skip the two completed and run translate/dub/mux only.

**Verification:**
```bash
uv run pytest tests/test_jobs.py -q
# expects: all passing incl. resume test
```

### Task 12: End-to-end test (real pipeline, tiny clip)

**Files:** `tests/test_e2e.py` (create)

`test_e2e.py` (marked `@pytest.mark.slow`, skipped if ffmpeg or `OPENAI_API_KEY` absent):
- Generate the 2s silent clip (conftest fixture).
- Because silent audio yields no segments, this test asserts the pipeline **completes without error** and produces `output.mp4`, exercising extract → transcribe (empty) → translate (empty, no API call) → dub (silence) → mux. This validates wiring end to end without depending on real speech.
- A second `@pytest.mark.slow` test with a short real speech clip (committed under `tests/fixtures/`) asserts `segments` is nonempty and `output.mp4` plays (ffprobe shows a video + audio stream). Skipped without API key.

**Verification:**
```bash
uv run pytest tests/test_e2e.py -q -m slow
# expects: 1-2 passed (silent always; speech needs API key + ffmpeg)
```

### Task 13: Translation/ASR eval suite (lightweight)

**Files:** `tests/eval_quality.py` (create)

A non-pytest script (`uv run python tests/eval_quality.py`) that runs a known short clip through transcribe + translate and prints the source text + Vietnamese for human eyeballing. No hard asserts (non-deterministic). Documented in README as the way to sanity-check quality after changing models or prompts.

**Verification:**
```bash
uv run python tests/eval_quality.py --help
# expects: usage text, exit 0
```

### Task 14: README + final scope check

**Files:** `README.md` (modify)

Document: prerequisites (`brew install ffmpeg`, install uv), `uv sync`, copy `.env.example` → `.env` and fill the LLM endpoint, `uv run uvicorn app.main:app --reload`, open localhost:8000. Note the eval script. Note GPL: pipeline reimplemented clean, not copied from pyVideoTrans.

**Verification:**
```bash
uv run pytest -q          # full suite green (slow tests skipped without keys)
uv run uvicorn app.main:app --port 8000 &  sleep 3
curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/   # 200
kill %1
```

## NOT in scope (carried from reviews)

Multi-speaker diarization, voice cloning, job queue/multi-user, auth, subtitle proofread UI (E1 cut), perfect lip-sync, local LLM. Tighter dub timing (silence insertion beyond atempo cap) is a TODO.

## Build order / dependencies

```
T1 scaffold
 ├─ T2 srt ──────────────┐
 ├─ T3 jobs ─────────────┤
 ├─ T4 extract           │
 ├─ T5 asr               │
 ├─ T6 translate         │
 ├─ T7 tts               │
 └─ T8 mux+pipeline ◄ needs T2,T4,T5,T6,T7
        │
        ├─ T9 routes ◄ needs T3,T8
        ├─ T10 frontend ◄ needs T9 shape
        ├─ T11 jobs wiring ◄ needs T8
        ├─ T12 e2e ◄ needs all stages
        ├─ T13 eval ◄ needs T5,T6
        └─ T14 readme ◄ last
```
T2–T7 are independent after T1 and can be built in any order (or parallel).
