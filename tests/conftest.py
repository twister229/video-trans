from __future__ import annotations

import shutil
import subprocess

import pytest


@pytest.fixture
def sample_video(tmp_path):
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is not installed")
    output = tmp_path / "sample.mp4"
    command = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "color=c=black:s=160x120:d=2",
        "-f",
        "lavfi",
        "-i",
        "anullsrc",
        "-shortest",
        str(output),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    return output
