from app.stages import StageError
from app.stages.translate import translate

import pytest


class Message:
    def __init__(self, content):
        self.content = content


class Choice:
    def __init__(self, content):
        self.message = Message(content)


class Response:
    def __init__(self, content):
        self.choices = [Choice(content)]


class FakeCompletions:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return Response(reply)


class FakeClient:
    def __init__(self, replies):
        self.chat = type("Chat", (), {})()
        self.chat.completions = FakeCompletions(replies)


SEGMENTS = [
    {"start": 0.0, "end": 1.0, "text": "hello"},
    {"start": 1.0, "end": 2.0, "text": "world"},
]


def test_translate_batch_happy_path():
    client = FakeClient(["[[1]] xin chào\n[[2]] thế giới"])

    result = translate(SEGMENTS, client=client, model="model")

    assert [segment["text"] for segment in result] == ["xin chào", "thế giới"]
    assert result[0]["start"] == 0.0


def test_count_mismatch_falls_back_to_one_by_one():
    client = FakeClient(["[[1]] xin chào", "xin chào", "thế giới"])

    result = translate(SEGMENTS, client=client, model="model")

    assert [segment["text"] for segment in result] == ["xin chào", "thế giới"]
    assert len(client.chat.completions.calls) == 3


def test_translate_empty_does_not_call_client():
    client = FakeClient([])
    assert translate([], client=client, model="model") == []
    assert client.chat.completions.calls == []


def test_translate_surfaces_api_error():
    client = FakeClient([RuntimeError("boom")])

    with pytest.raises(StageError):
        translate(SEGMENTS, client=client, model="model")
