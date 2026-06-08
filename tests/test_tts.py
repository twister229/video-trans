import asyncio

import pytest

from app.stages import StageError
from app.stages import tts


def test_synthesize_respects_concurrency(monkeypatch, tmp_path):
    active = 0
    max_active = 0

    async def fake_segment(_text, _voice, out_path, *, retries=3):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.01)
        out_path.write_bytes(b"mp3")
        active -= 1

    monkeypatch.setattr(tts, "synthesize_segment", fake_segment)
    segments = [{"text": str(i)} for i in range(6)]

    outputs = asyncio.run(tts.synthesize(segments, "voice", tmp_path, concurrency=2))

    assert len(outputs) == 6
    assert max_active <= 2


def test_synthesize_names_failed_segment(monkeypatch, tmp_path):
    async def fail_segment(_text, _voice, _out_path, *, retries=3):
        raise StageError("bad")

    monkeypatch.setattr(tts, "synthesize_segment", fail_segment)

    with pytest.raises(StageError, match="segment 1"):
        asyncio.run(tts.synthesize([{"text": "x"}], "voice", tmp_path, concurrency=1))
