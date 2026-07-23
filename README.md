# field-audio-tools

Two small, local-first command-line tools for field recordings:

- **soundcite** turns an exact time range into a portable evidence package
  containing lossless excerpts, a browser preview, a spectrogram, checksums,
  technical metadata, and a static citation page.
- **windwindow** screens long recordings for windows dominated by broadband
  low-frequency energy, a common sign of wind buffeting. It produces a
  transparent CSV and a static report; it does not delete or alter audio.

Both tools support a regular mono/stereo file or the two synchronized mono WAV
files produced by a Zoom F3. Processing stays on the local machine.

## Status

This is an alpha release. In particular, `windwindow` is a screening heuristic,
not a validated wind classifier. Water, vehicles, handling noise, and other
geophony can look similar. Always listen before excluding recordings from an
analysis.

## Requirements

- Python 3.11 or newer
- FFmpeg and ffprobe available on `PATH`

Install for development:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[test]'
pytest
```

If the checkout is on a FAT/exFAT-style filesystem that does not support
symbolic links, create the virtual environment on a Linux filesystem instead:

```sh
python -m venv ~/.venvs/field-audio-tools
~/.venvs/field-audio-tools/bin/pip install -e '/path/to/field-audio-tools[test]'
```

## soundcite

Create a 20-second evidence package from a stereo file:

```sh
soundcite recording.wav \
  --start 01:23:40 \
  --duration 20 \
  --title "Unidentified call near Yema Haizi" \
  --recordist "Mountaineering Acoustics Network" \
  --recorded-at "2026-01-15T12:19:25+08:00" \
  --location "Yema Haizi, Kangding, Sichuan" \
  --license "CC BY 4.0" \
  --output evidence/yema-012340
```

For a Zoom F3 dual-mono take, provide both tracks in order:

```sh
soundcite 260115_002_Tr1.WAV 260115_002_Tr2.WAV \
  --start 00:10:00 --duration 20 \
  --output evidence/take-002-001000
```

The package contains:

- `excerpt.wav`, or `excerpt-track1.wav` and `excerpt-track2.wav`: 32-bit
  floating-point excerpts
- `preview.ogg`: lossy, peak-limited browser preview
- `spectrogram.png`
- `manifest.json`: source metadata and SHA-256 checksums
- `index.html`: self-contained citation page

Hashing multi-gigabyte source recordings is deliberately optional. Add
`--hash-source` when a complete source checksum is required.

## windwindow

Screen one recording:

```sh
windwindow recording.wav --output reports/recording
```

Screen both tracks of a Zoom F3 take conservatively. The merged score is the
higher score from either track:

```sh
windwindow 260115_002_Tr1.WAV 260115_002_Tr2.WAV \
  --window 10 \
  --threshold 0.6 \
  --output reports/take-002
```

Outputs:

- `windows.csv`: score and component metrics for every window and track
- `candidate-clean-intervals.csv`: contiguous ranges below the threshold
- `summary.json`: sources, parameters, and totals
- `report.html`: offline timeline

The default score measures the proportion of energy in 10–120 Hz relative to
10–1000 Hz, with simple signal-level and low-band spectral-flatness gates. All
parameters are recorded in `summary.json`.

## Design principles

- Never modify source recordings.
- Make lossy or heuristic processing explicit.
- Record parameters and provenance in machine-readable files.
- Prefer static output that can be archived or served without a backend.
- Keep formats simple enough to bridge into other bioacoustic workflows.

## License

GPL-3.0-or-later.
