# field-audio-tools

Three small, local-first command-line tools for field recordings:

- **soundcite** turns an exact time range into a portable evidence package
  containing lossless excerpts, a browser preview, a spectrogram, checksums,
  technical metadata, and a static citation page.
- **lowdom** screens long recordings for windows dominated by broadband
  low-frequency energy, a common sign of wind buffeting. It produces a
  transparent CSV and a static report; it does not delete or alter audio.
- **birdidpv** runs BirdNET over a recording, or over just the ranges `lowdom`
  left unflagged, and reports per-frame detections and a species roll-up. It
  needs an optional extra; see [birdidpv](#birdidpv).

All three tools support a regular mono/stereo file or the two synchronized mono WAV
files produced by a Zoom F3. Processing stays on the local machine.

They are not three equal siblings. `lowdom` feeds `birdidpv` directly, and
`soundcite` is what you reach for once a detection has earned it. See
[how the three fit together](#how-the-three-fit-together).

## Status

This is an alpha release. In particular, `lowdom` is a screening heuristic,
not a validated wind classifier. Water, vehicles, handling noise, and other
geophony can look similar. Always listen before excluding recordings from an
analysis.

The default `lowdom` threshold is **0.95**, chosen provisionally after blind
listening on one F3 take. An earlier CLI default of **0.6** was a placeholder
and was not validated on that recording before it was replaced.

## Requirements

- Python 3.11 or newer
- NumPy (installed with the package)
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

There is no PyPI release yet. The Mantle site hosts a [software
overview](https://mantle-sound.org/software/) and a [test note on Take
002](https://mantle-sound.org/software/running-field-audio-tools-on-a-long-f3-take/)
with linked demonstration packages.

## How the three fit together

```text
lowdom  --candidate-clean-intervals.csv-->  birdidpv  --detections.csv-->  you
                                                                            |
                                                          a timecode worth keeping
                                                                            |
                                                                            v
                                                                        soundcite
```

Only the first arrow is automatic. `birdidpv --intervals` reads the CSV `lowdom`
writes, so screening and identification chain without you retyping anything.

The second arrow is deliberately manual. `soundcite` takes a time range you have
already decided on, and deciding means listening. On a January lake `birdidpv`
will offer waterbirds that are not there. In the run this README quotes, a
Ruddy Shelduck scoring 0.93 was a guide shouting from across the valley. A
script that piped detections straight into evidence packages would remove the
one step that catches that.

### Citing a detection

`detections.csv` carries `start_timecode` in `HH:MM:SS.mmm`, which is one of the
formats `--start` accepts, so the value copies across unchanged. Find the
longest unbroken run of frames rather than the single best one: a run is better
evidence than a frame, and `--duration` then covers it.

```sh
soundcite 260115_002_Tr1.WAV 260115_002_Tr2.WAV \
  --start 00:40:29.000 \
  --duration 21 \
  --pad-start 3 \
  --pad-end 4 \
  --title "Red-billed Chough, Take 002" \
  --recordist "Your Name" \
  --location "Public-safe description" \
  --output evidence/chough-004029
```

Those numbers come from a real run. The longest run in the take is seven frames
starting at 00:40:29, all *Pyrrhocorax pyrrhocorax*, scoring 0.9863 to 0.9998.
Seven three-second frames is 21 seconds, which is the `--duration`. The padding
adds context on both sides, because 00:40:29 is a frame boundary rather than the
moment the bird began, and cutting exactly on it can clip the first call. See
[Context around the cited range](#context-around-the-cited-range).

The written excerpt therefore runs 00:40:26 to 00:40:54, while `manifest.json`
records 00:40:29 for 21 seconds as the range being cited.

## soundcite

Create a 20-second evidence package from a stereo file:

```sh
soundcite recording.wav \
  --start 01:23:40 \
  --duration 20 \
  --title "Unidentified call near Yema Haizi" \
  --recordist "Gewenxin Yu" \
  --recorded-at "2026-01-15T13:43:05+08:00" \
  --location "Yema Haizi, Kangding, Sichuan" \
  --license "CC BY-NC-SA 4.0" \
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
- `spectrogram.png`: generated directly from the lossless excerpts
- `manifest.json`: source metadata, exact sample range, and artifact details
- `SHA256SUMS`: checksums for all other package files
- `index.html` and `index-bare.html`: identical bare HTML 4.01 citation pages
  from the CLI

### Context around the cited range

`--start` and `--duration` say what you are citing. A call rarely begins where
you cut, so `--pad` keeps extra seconds on either side without changing what the
excerpt claims to be:

```sh
soundcite recording.wav --start 00:40:29 --duration 21 --pad 3
soundcite recording.wav --start 00:40:29 --duration 21 --pad-start 3 --pad-end 4
```

`--pad` sets both ends. `--pad-start` and `--pad-end` override it, so an
asymmetric request only names the end that differs. All three take the same
formats as `--start`.

This matters most when the time came from `birdidpv`. Detections are scored on a
fixed three-second grid, and a frame boundary is not the start of a call: it
only says the call falls somewhere inside those three seconds. Cutting exactly
on the boundary can clip the opening of the call.

With padding, `manifest.json` separates the two ranges. The top-level `clip`
describes the audio that was written, because that is what the checksums cover.
`clip.cited` records the range you asked to cite, and `clip.padding` records how
much context was added:

```json
"clip": {
  "start_timecode": "00:40:26.000", "duration_seconds": 28.0,
  "cited": { "start_timecode": "00:40:29.000", "duration_seconds": 21.0 },
  "padding": { "start_seconds": 3.0, "end_seconds": 4.0,
               "start_clamped": false, "end_clamped": false }
}
```

Padding runs out at the ends of a file. Rather than refuse the run, `soundcite`
keeps what is available and sets `start_clamped` or `end_clamped`, so an excerpt
that carried less context than requested says so instead of appearing to have
had none asked for.

Without any padding flag, the manifest keeps the shape it has always had.

For paired mono inputs, `soundcite` checks sample rate, duration, channel count,
embedded recording time, and BWF time reference before extracting audio. When
`--recorded-at` is omitted, it derives the excerpt time from embedded
`date`/`creation_time` tags when available. Input paths are private by default;
add `--include-source-path` only when the path itself belongs in the manifest.

Hashing multi-gigabyte source recordings is deliberately optional. Add
`--hash-source` when a complete source checksum is required.

When publishing on the Mantle site, the demonstration replaces `index.html`
with a site-styled page and keeps `index-bare.html` as the tool output. Styled
pages link `/static/software-package-media.css` for the audio player and
spectrogram framing.

## lowdom

Screen one recording:

```sh
lowdom recording.wav --output reports/recording
```

Screen both tracks of a Zoom F3 take conservatively. The merged score is the
higher score from either track:

```sh
lowdom 260115_002_Tr1.WAV 260115_002_Tr2.WAV \
  --window 10 \
  --threshold 0.95 \
  --report both \
  --output reports/take-002
```

The 0.95 default is provisional. It separated two blinded listening samples
from one F3 take, but it is not yet validated across recording setups or sites.

Outputs:

- `windows.csv`: score and component metrics for every window and track
- `candidate-clean-intervals.csv`: contiguous ranges below the threshold
- `summary.json`: sources, parameters, and totals
- `report.html`: bare HTML 4.01 offline timeline (default `--report bare`)
- `report-multimedia.html` and `recording-preview-48k/`: optional hover-playback
  report (`--report multimedia` or `--report both`; lossy Opus preview, not for analysis)

The default score measures the proportion of energy in 10–120 Hz relative to
10–1000 Hz, with simple signal-level and low-band spectral-flatness gates. All
parameters are recorded in `summary.json`.

Mantle demonstration packages may copy CLI HTML to `report-bare.html` and
`report-multimedia-bare.html` before serving site-styled `report.html` and
`report-js.html` with shared `/static/software-package-media.css`.

## birdidpv

Run [BirdNET](https://birdnet.cornell.edu/) over a recording and report what it
thinks it heard. One recording and one command is the whole of it:

```sh
birdidpv recording.wav \
  --lat 29.898 --lon 102.030 --week 3 \
  --report both --photos --spectrograms \
  --output reports/take
```

That writes a self-contained directory: the CSVs, a bare offline timeline, and a
multimedia report with its own preview audio, per-frame spectrograms, and
reference photographs. Nothing in it points outside itself, and none of it needs
a server.

If you have also run `lowdom`, hand `birdidpv` the ranges it left unflagged rather
than the whole take. `lowdom` already knows which stretches are not buried under
low-frequency energy, so the model spends its time where there is something to
hear:

```sh
birdidpv 260115_002_Tr1.WAV \
  --intervals reports/take-002/candidate-clean-intervals.csv \
  --lat 29.898 --lon 102.030 --week 3 \
  --min-conf 0.25 \
  --report both \
  --output reports/take-002-birdidpv
```

`--windows windows.csv` derives the same ranges from the scores directly. With
neither, the whole recording is analysed.

`--lat`, `--lon`, and `--week` restrict the 6,522-species model to species
plausible at that place and time of year. This matters: unfiltered, the model
will confidently offer species from the wrong continent. Raise
`--filter-threshold` to shorten the list further, or supply your own with
`--species-list`.

Outputs:

- `detections.csv`: every 3-second frame above `--min-conf`, with timecodes
- `species.csv`: per-species count, highest confidence, and first appearance
- `summary.json`: sources, ranges analysed, parameters, and totals
- `report.html`: bare HTML 4.01 offline timeline (default `--report bare`)
- `report-multimedia.html`: hover-playback timeline with a species table that
  filters the timeline when a row is selected
- `recording-preview-48k/`: lossy Opus for the multimedia report, written unless
  one is already there to reuse
- `species-photos/` and `species-photos/credits.json`: optional reference
  photographs (`--photos`)
- `detection-spectrograms/`: optional per-frame spectrograms (`--spectrograms`)

### Detection spectrograms

`--spectrograms` renders one spectrogram per detected frame with FFmpeg, ahead
of time, from the source audio rather than the lossy preview. The multimedia
report shows the frame under the pointer, so a detection can be looked at as
well as listened to:

```sh
birdidpv recording.wav --output reports/take --report multimedia --spectrograms
```

Frames sharing a start time are rendered once, so two species detected in the
same three seconds share an image. Files are named `frame-<milliseconds>.webp`,
and the name is embedded with each detection in the report so the page never
reconstructs it from a convention.

The axes are logarithmic in both frequency and amplitude. A linear frequency
axis spends most of its height on empty ultrasound and squashes everything below
500 Hz into a single line, which is the band that usually decides whether a
detection is a bird at all. `--spectrogram-size` sets the spectrum area; the
axis legend adds roughly 280x130 around it. WebP is the default at about a tenth
the size of PNG; on one 128-detection report that is 2.8 MB rather than 28 MB.

A picture shows you that something is there. It does not tell you what. On the
Take 002 demo the frame BirdNET scored highest after the corvids, Ruddy Shelduck
at 0.928, has clear harmonic stacks between 1 and 3 kHz. We read them as a call,
because ice does not make structure like that. They are a man shouting across
the lake — a pitch contour through its harmonics, formants near 1.2 and 2 kHz,
four syllables in the last second. The spectrogram was worth having, and it
still took listening to settle it.

Rendering failures are recorded as skipped frames rather than raised, so a
report is still produced when the source audio has moved since the analysis.

### Reference photographs

`--photos` adds one reference photograph per species to the report, fetched
from [iNaturalist](https://www.inaturalist.org/). This is the only part of the
toolkit that uses the network, so it is off by default:

```sh
birdidpv recording.wav --output reports/take --photos
```

Photographs are downloaded **into the package**, not hot-linked: a package has
to survive being archived and served offline, and hot-linking would also
disclose every reader's address to a third party. Each file is stored exactly
as iNaturalist served it, so no derivative is made. `credits.json` records the
photographer, licence, iNaturalist taxon, and source URL for each one.

Only exact scientific-name matches are used. iNaturalist's search is fuzzy and
ranks congeners highly — a query for *Pyrrhocorax pyrrhocorax* returns
*Pyrrhocorax graculus* first — and a near miss would caption the report with the
wrong bird. A species with no exact match gets no photograph and a recorded
reason rather than a guess.

Photographs with no licence (all rights reserved) are never downloaded.
`--photo-licenses` narrows what is accepted beyond that; the default accepts the
CC variants including ND, which is sound only because the file is stored
verbatim. Network failures are recorded as skipped species, never as errors —
the report is still worth writing without a picture.

A photograph shows what a species looks like. It is not evidence that the
species was present.

Avibase has no public API and hosts no photographs of its own — the images on
its species pages are Flickr thumbnails matched by scientific name — so it
cannot serve this purpose.

### Preview audio

A hover-playback report needs audio it can load, so `--report multimedia` makes
sure there is some. If `--preview-dir` already holds segments — a `lowdom`
package next door, for instance — they are reused and nothing is re-encoded.
Otherwise `birdidpv` writes its own, exactly as `lowdom` does. Running one command
on one recording gives you a page that works:

```sh
birdidpv recording.wav --output reports/take --report multimedia
```

`--no-preview` skips this. The report is then built without a player rather than
with a dead one: the timeline, the spectrograms, and the species filter all
still work.

### BirdNET, and what it means for your results

None of the identifying here is this project's work. BirdNET is developed by the
K. Lisa Yang Center for Conservation Bioacoustics at the Cornell Lab of
Ornithology with Chemnitz University of Technology, and reached through
[birdnetlib](https://github.com/joeweiss/birdnetlib). This tool decides which
seconds to hand it and what to do with the answer; the answer is theirs.

If identification is all you want, use the official
[BirdNET-Analyzer](https://birdnet-team.github.io/BirdNET-Analyzer/) instead. It
takes the same `--lat`, `--lon`, `--week`, `--min_conf`, `--sensitivity` and
`--overlap`, and adds a GUI, an analysis server, embedding extraction, custom
classifier training, and more output formats. What is here that is not there is
the packaging: ranges taken from a `lowdom` screening, and a single
self-contained page that can be listened to, looked at, and archived.

**Licensing.** BirdNET's source is MIT, but **the models are
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)**. That
non-commercial condition reaches your results, not just the model: it constrains
what you may do with what `birdidpv` writes, whatever licence this toolkit carries.
The BirdNET authors state that educational and research use counts as
non-commercial. This project redistributes no model file of its own: the models
arrive inside the birdnetlib package, which is Apache-2.0 but ships the
CC BY-NC-SA models in its wheel. So this project's GPL-3.0-or-later never meets
the models' terms in distribution, but both apply to you once installed — and
because the files then sit on your disk, handing that environment to someone
else is redistribution under CC BY-NC-SA, share-alike condition included.

**Attribution is a condition of that licence, not a courtesy.** Every run writes
it into `summary.json` and both reports. If you cite the results, cite:

> Kahl, S., Wood, C. M., Eibl, M., & Klinck, H. (2021). BirdNET: A deep learning
> solution for avian diversity monitoring. *Ecological Informatics*, 61, 101236.

This project is not affiliated with or endorsed by the BirdNET team.

### Requirements

BirdNET needs TensorFlow, which is far heavier than the rest of this project, so
it is an optional extra:

```sh
pip install -e '.[birdnet]'
```

Without it, `birdidpv` exits with an install hint and the other tools are
unaffected. The model files come inside the birdnetlib wheel, about 65 MB of
them, so the install is the only step that needs a network. Analysis itself
runs offline.

### What a confidence is not

BirdNET returns a score per 3-second frame. It is not a probability, it is not a
verified record, and a single high-scoring frame is not a sighting. On a frozen
lake in January this tool will still offer waterbirds. Treat the output as a
listening index that tells you where to listen, in the same spirit as `lowdom`.

## Design principles

- Never modify source recordings.
- Make lossy or heuristic processing explicit.
- Record parameters and provenance in machine-readable files.
- Prefer static output that can be archived or served without a backend.
- Keep formats simple enough to bridge into other bioacoustic workflows.

## License

GPL-3.0-or-later.
