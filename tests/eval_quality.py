from __future__ import annotations

import argparse
from pathlib import Path

from app.config import get_config
from app.stages.asr import transcribe
from app.stages.translate import translate


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a human-readable ASR + VI translation quality check.")
    parser.add_argument("audio", type=Path, nargs="?", help="Audio/video file to inspect")
    args = parser.parse_args()
    if args.audio is None:
        parser.print_help()
        return
    config = get_config()
    segments = transcribe(args.audio, config.whisper_model)
    print("SOURCE")
    for segment in segments:
        print(f"[{segment['start']:.2f}-{segment['end']:.2f}] {segment['text']}")
    print("\nVIETNAMESE")
    # Intentionally imports the real OpenAI client only when used interactively.
    from openai import OpenAI

    client = OpenAI(base_url=config.openai_base_url, api_key=config.openai_api_key)
    for segment in translate(segments, client=client, model=config.openai_model):
        print(f"[{segment['start']:.2f}-{segment['end']:.2f}] {segment['text']}")


if __name__ == "__main__":
    main()
