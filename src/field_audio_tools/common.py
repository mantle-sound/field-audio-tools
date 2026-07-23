from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Sequence


class ToolError(RuntimeError):
    """A user-facing error without a traceback."""


def require_program(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise ToolError(f"Required program not found: {name}")
    return path


def run(
    command: Sequence[str],
    *,
    capture_output: bool = True,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            check=True,
            capture_output=capture_output,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        if len(detail) > 1200:
            detail = detail[-1200:]
        message = f"Command failed: {' '.join(command)}"
        if detail:
            message += f"\n{detail}"
        raise ToolError(message) from exc


def probe_audio(path: Path) -> dict[str, Any]:
    ffprobe = require_program("ffprobe")
    result = run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            (
                "stream=index,codec_name,codec_long_name,sample_fmt,sample_rate,"
                "channels,channel_layout,bits_per_sample,duration,bit_rate:"
                "format=format_name,format_long_name,duration,size,bit_rate:"
                "format_tags"
            ),
            "-of",
            "json",
            str(path),
        ]
    )
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ToolError(f"ffprobe returned invalid JSON for {path}") from exc


def parse_time(value: str) -> float:
    """Parse seconds, MM:SS, or HH:MM:SS into non-negative seconds."""
    try:
        parts = [float(part) for part in value.split(":")]
    except ValueError as exc:
        raise ValueError(f"Invalid time value: {value}") from exc
    if not 1 <= len(parts) <= 3 or any(part < 0 for part in parts):
        raise ValueError(f"Invalid time value: {value}")
    if len(parts) > 1 and any(part >= 60 for part in parts[1:]):
        raise ValueError(f"Minutes and seconds must be below 60: {value}")
    seconds = 0.0
    for part in parts:
        seconds = seconds * 60 + part
    return seconds


def format_time(seconds: float) -> str:
    milliseconds = round(seconds * 1000)
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def prepare_output(path: Path, force: bool) -> Path:
    if path.exists():
        if not path.is_dir():
            raise ToolError(f"Output path exists and is not a directory: {path}")
        if any(path.iterdir()) and not force:
            raise ToolError(
                f"Output directory is not empty: {path} (use --force to overwrite)"
            )
    path.mkdir(parents=True, exist_ok=True)
    return path
