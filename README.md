# video-trans

Local web app for turning a video into Vietnamese subtitles and a Vietnamese dub.

The app runs on your Mac, accepts one video at a time, and returns:

- `output.mp4` — original video with lowered original audio + Vietnamese voiceover + soft Vietnamese subtitles
- `subtitles.vi.srt` — downloadable Vietnamese subtitle file

## Prerequisites

```bash
brew install ffmpeg
curl -LsSf https://astral.sh/uv/install.sh | sh
```

This project pins Python to 3.12 via `uv`. Do not use the system Python 3.14; ML packages often lag new Python releases.

## Configure

```bash
cp .env.example .env
```

Fill in your OpenAI-compatible provider:

```bash
OPENAI_BASE_URL=https://your-provider.example/v1
OPENAI_API_KEY=...
OPENAI_MODEL=...
```

Defaults:

- ASR: `mlx-community/whisper-large-v3-mlx`
- TTS voice: `vi-VN-HoaiMyNeural`
- Max Edge-TTS concurrency: `4`

## Run

```bash
uv sync
uv run uvicorn app.main:app --reload
```

Open <http://localhost:8000>.

## Test

```bash
uv run pytest -q
```

Slow tests that touch real media processing:

```bash
uv run pytest -q -m slow
```

Human-readable ASR + translation quality check:

```bash
uv run python tests/eval_quality.py path/to/audio-or-video.mp4
```

## Notes

- The pipeline is clean-room reimplemented. It borrows the workflow idea from pyVideoTrans but does not copy GPL code.
- The server runs one job at a time. A second upload returns `409 Conflict` while a job is active.
- All ffmpeg calls use subprocess argument arrays, never shell strings.
