from __future__ import annotations

import os
from dataclasses import dataclass
from functools import cache
from pathlib import Path


@dataclass(frozen=True)
class Config:
    openai_base_url: str = "https://api.openai.com/v1"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    whisper_model: str = "mlx-community/whisper-large-v3-mlx"
    tts_voice: str = "vi-VN-HoaiMyNeural"
    max_tts_concurrency: int = 4
    data_dir: Path = Path("data/jobs")


@cache
def get_config() -> Config:
    return Config(
        openai_base_url=os.environ.get("OPENAI_BASE_URL", Config.openai_base_url),
        openai_api_key=os.environ.get("OPENAI_API_KEY", Config.openai_api_key),
        openai_model=os.environ.get("OPENAI_MODEL", Config.openai_model),
        whisper_model=os.environ.get("WHISPER_MODEL", Config.whisper_model),
        tts_voice=os.environ.get("TTS_VOICE", Config.tts_voice),
        max_tts_concurrency=int(os.environ.get("MAX_TTS_CONCURRENCY", Config.max_tts_concurrency)),
        data_dir=Path(os.environ.get("DATA_DIR", str(Config.data_dir))),
    )
