from __future__ import annotations

import json
import re

from app.config import get_config
from app.jobs import Job
from app.stages import StageError


MARKER_RE = re.compile(r"\[\[(\d+)\]\]\s*(.*?)(?=\n\[\[\d+\]\]|\Z)", re.S)


def translate(segments: list[dict], *, client, model: str) -> list[dict]:
    if not segments:
        return []
    try:
        translated = _translate_batch(segments, client=client, model=model)
        if len(translated) != len(segments):
            translated = _translate_one_by_one(segments, client=client, model=model)
    except Exception as exc:
        raise StageError(f"translation failed: {exc}") from exc
    return [
        {"start": segment["start"], "end": segment["end"], "text": translated[index]}
        for index, segment in enumerate(segments)
    ]


def _translate_batch(segments: list[dict], *, client, model: str) -> list[str]:
    payload = "\n".join(f"[[{index}]] {segment['text']}" for index, segment in enumerate(segments, start=1))
    text = _chat(
        client,
        model,
        "Translate each numbered line to natural Vietnamese. Return exactly one line per input, each prefixed with its same [[i]] marker. No commentary.",
        payload,
    )
    parsed = {int(match.group(1)): match.group(2).strip() for match in MARKER_RE.finditer(text)}
    if set(parsed) != set(range(1, len(segments) + 1)):
        return []
    return [parsed[index] for index in range(1, len(segments) + 1)]


def _translate_one_by_one(segments: list[dict], *, client, model: str) -> list[str]:
    results = []
    for segment in segments:
        results.append(
            _chat(
                client,
                model,
                "Translate this subtitle line to natural Vietnamese. Return only the translation.",
                segment["text"],
            ).strip()
        )
    return results


def _chat(client, model: str, system: str, user: str) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    return response.choices[0].message.content or ""


def translate_job(job: Job) -> None:
    from openai import OpenAI

    config = get_config()
    client = OpenAI(base_url=config.openai_base_url, api_key=config.openai_api_key)
    segments = json.loads((job.dir / "segments.json").read_text(encoding="utf-8"))
    vi_segments = translate(segments, client=client, model=config.openai_model)
    (job.dir / "vi_segments.json").write_text(json.dumps(vi_segments, indent=2), encoding="utf-8")
