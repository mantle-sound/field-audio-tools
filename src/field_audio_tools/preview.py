"""Lossy preview audio for the multimedia reports.

Shared by lowdom and birdidpv. A hover-playback report is useless without
listenable audio, and neither tool should assume the other has already produced
it: someone who has only a recording and runs one command must end up with a
page that works.

The preview is deliberately lossy and peak-limited. It is for listening in a
browser, not for analysis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from .common import require_program, run

SEGMENT_STEM = "segment"
DEFAULT_CHUNK_SECONDS = 300.0
DEFAULT_SAMPLE_RATE = 48000
DEFAULT_BITRATE = "48k"

_LIMITER = "alimiter=limit=0.95:attack=5:release=50:level=false"


def segment_name(index: int) -> str:
    return f"{SEGMENT_STEM}-{index:03d}.ogg"


def has_segments(preview_dir: Path) -> bool:
    """True when a directory already holds a usable preview."""
    return (preview_dir / segment_name(0)).is_file()


def write_preview_segments(
    sources: Sequence[Path],
    preview_dir: Path,
    duration_seconds: float,
    *,
    chunk_seconds: float = DEFAULT_CHUNK_SECONDS,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    bitrate: str = DEFAULT_BITRATE,
) -> int:
    """Write Opus preview chunks and return how many were written."""
    ffmpeg = require_program("ffmpeg")
    preview_dir.mkdir(parents=True, exist_ok=True)
    chunk_index = 0
    start = 0.0
    while start < duration_seconds - 0.001:
        chunk_duration = min(chunk_seconds, duration_seconds - start)
        destination = preview_dir / segment_name(chunk_index)
        command = [ffmpeg, "-v", "error", "-y"]
        for source in sources:
            command.extend(
                [
                    "-ss",
                    f"{start:.6f}",
                    "-t",
                    f"{chunk_duration:.6f}",
                    "-i",
                    str(source),
                ]
            )
        if len(sources) == 1:
            command.extend(
                [
                    "-af",
                    _LIMITER,
                    "-ar",
                    str(sample_rate),
                    "-c:a",
                    "libopus",
                    "-b:a",
                    bitrate,
                    str(destination),
                ]
            )
        else:
            labels = "".join(f"[{index}:a]" for index in range(len(sources)))
            filter_graph = f"{labels}amerge=inputs={len(sources)},{_LIMITER}[a]"
            command.extend(
                [
                    "-filter_complex",
                    filter_graph,
                    "-map",
                    "[a]",
                    "-ar",
                    str(sample_rate),
                    "-c:a",
                    "libopus",
                    "-b:a",
                    bitrate,
                    str(destination),
                ]
            )
        run(command)
        chunk_index += 1
        start += chunk_seconds
    return chunk_index
