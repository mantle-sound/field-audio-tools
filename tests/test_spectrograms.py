import pytest

from field_audio_tools import spectrograms
from field_audio_tools.common import ToolError
from field_audio_tools.spectrograms import (
    DetectionSpectrogram,
    _file_name,
    _valid_size,
    index_by_start,
    render_detections,
)


@pytest.fixture
def renderer(monkeypatch, tmp_path):
    """Stub the FFmpeg call; these tests never spawn a process."""
    state = {"calls": [], "fail_at": set()}

    def fake_render(source, destination, start, duration, *, size, ffmpeg):
        state["calls"].append(
            {"start": start, "duration": duration, "size": size, "dest": destination}
        )
        if round(start * 1000) in state["fail_at"]:
            raise ToolError(f"FFmpeg could not render at {start}")
        destination.write_bytes(b"RIFFstub")

    monkeypatch.setattr(spectrograms, "render_one", fake_render)
    monkeypatch.setattr(spectrograms, "require_program", lambda name: "/usr/bin/ffmpeg")
    source = tmp_path / "take.wav"
    source.write_bytes(b"\0")
    state["source"] = source
    return state


def test_file_name_is_stable_and_sorts_by_time():
    assert _file_name(0.0, "webp") == "frame-000000000.webp"
    assert _file_name(2435.0, "webp") == "frame-002435000.webp"
    assert _file_name(679.5, "png") == "frame-000679500.png"
    assert _file_name(12.0, "webp") < _file_name(2435.0, "webp")


def test_valid_size():
    assert _valid_size("560x260")
    assert not _valid_size("560")
    assert not _valid_size("560x")
    assert not _valid_size("0x260")
    assert not _valid_size("-5x260")
    assert not _valid_size("wide x tall")


def test_render_writes_one_image_per_frame(renderer, tmp_path):
    out = tmp_path / "spec"
    rendered, failed = render_detections(
        renderer["source"], [(0.0, 3.0), (2435.0, 2438.0)], out
    )
    assert failed == []
    assert [item.file_name for item in rendered] == [
        "frame-000000000.webp",
        "frame-002435000.webp",
    ]
    assert (out / "frame-002435000.webp").exists()


def test_frames_sharing_a_start_are_rendered_once(renderer, tmp_path):
    # Two species detected in the same three seconds must not render twice.
    rendered, _ = render_detections(
        renderer["source"],
        [(1480.0, 1483.0), (1480.0, 1483.0), (1483.0, 1486.0)],
        tmp_path / "spec",
    )
    assert len(rendered) == 2
    assert len(renderer["calls"]) == 2


def test_output_is_ordered_by_time_whatever_the_input_order(renderer, tmp_path):
    rendered, _ = render_detections(
        renderer["source"], [(90.0, 93.0), (12.0, 15.0), (50.0, 53.0)], tmp_path / "s"
    )
    assert [item.start_seconds for item in rendered] == [12.0, 50.0, 90.0]


def test_a_failed_frame_is_skipped_not_fatal(renderer, tmp_path):
    renderer["fail_at"] = {2435000}
    rendered, failed = render_detections(
        renderer["source"], [(0.0, 3.0), (2435.0, 2438.0)], tmp_path / "s"
    )
    assert len(rendered) == 1
    assert len(failed) == 1
    assert failed[0][0] == 2435.0


def test_requested_size_and_format_reach_ffmpeg(renderer, tmp_path):
    rendered, _ = render_detections(
        renderer["source"],
        [(0.0, 3.0)],
        tmp_path / "s",
        size="800x400",
        image_format="png",
    )
    assert renderer["calls"][0]["size"] == "800x400"
    assert rendered[0].file_name.endswith(".png")


def test_a_zero_length_frame_still_gets_a_positive_duration(renderer, tmp_path):
    render_detections(renderer["source"], [(10.0, 10.0)], tmp_path / "s")
    assert renderer["calls"][0]["duration"] > 0


def test_unknown_format_is_rejected(renderer, tmp_path):
    with pytest.raises(ToolError):
        render_detections(renderer["source"], [(0.0, 3.0)], tmp_path, image_format="tiff")


def test_bad_size_is_rejected(renderer, tmp_path):
    with pytest.raises(ToolError):
        render_detections(renderer["source"], [(0.0, 3.0)], tmp_path, size="huge")


def test_missing_source_is_rejected(tmp_path):
    with pytest.raises(ToolError):
        render_detections(tmp_path / "absent.wav", [(0.0, 3.0)], tmp_path)


def test_index_by_start_keys_on_milliseconds():
    index = index_by_start(
        [
            DetectionSpectrogram(2435.0, 2438.0, "frame-002435000.webp"),
            DetectionSpectrogram(679.5, 682.5, "frame-000679500.webp"),
        ]
    )
    assert index == {
        2435000: "frame-002435000.webp",
        679500: "frame-000679500.webp",
    }


# --- integration: these actually invoke FFmpeg ------------------------------

import shutil
import subprocess


needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="FFmpeg is not on PATH"
)


@pytest.fixture
def tone(tmp_path):
    """A real five-second mono WAV to render from."""
    path = tmp_path / "tone.wav"
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-f", "lavfi",
            "-i", "sine=frequency=1200:duration=5:sample_rate=48000",
            "-y", str(path),
        ],
        check=True,
        capture_output=True,
    )
    return path


@needs_ffmpeg
def test_ffmpeg_actually_writes_an_image(tone, tmp_path):
    """The stubbed tests cannot catch a malformed FFmpeg command; this can."""
    out = tmp_path / "spec"
    rendered, failed = render_detections(tone, [(1.0, 4.0)], out)
    assert failed == []
    assert len(rendered) == 1
    written = out / rendered[0].file_name
    assert written.exists()
    # RIFF....WEBP
    header = written.read_bytes()[:12]
    assert header[:4] == b"RIFF" and header[8:12] == b"WEBP"


@needs_ffmpeg
def test_ffmpeg_writes_png_when_asked(tone, tmp_path):
    out = tmp_path / "spec"
    rendered, failed = render_detections(
        tone, [(0.0, 3.0)], out, image_format="png", size="320x160"
    )
    assert failed == []
    assert (out / rendered[0].file_name).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


@needs_ffmpeg
def test_a_range_past_the_end_is_skipped_not_fatal(tone, tmp_path):
    rendered, failed = render_detections(tone, [(1.0, 4.0), (900.0, 903.0)], tmp_path)
    assert len(rendered) == 1
    assert len(failed) == 1
