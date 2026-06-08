from app.srt import format_timestamp, segments_to_srt


def test_format_timestamp():
    assert format_timestamp(3661.5) == "01:01:01,500"


def test_segments_to_srt():
    segments = [
        {"start": 0.0, "end": 1.25, "text": "Xin chào"},
        {"start": 61.5, "end": 63.0, "text": "Tạm biệt"},
    ]

    assert segments_to_srt(segments) == (
        "1\n"
        "00:00:00,000 --> 00:00:01,250\n"
        "Xin chào\n"
        "\n"
        "2\n"
        "00:01:01,500 --> 00:01:03,000\n"
        "Tạm biệt\n"
    )
