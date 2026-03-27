import os
import shutil
import struct

from src.audio.stereo_audio import StereoAudio
from src.audio.note_parsing import NoteParser, build_note_parser
from src.audio.synth_factory import CustomState, build_synth_factories
from src.audio.arrangement import Clip, Master, Track
from src.audio.event_scheduler import RetriggerMode

SAMPLE_RATE = 48_000
EVENT_BIN_WIDTH = 1
BPM = 136

# C5 is 8 edosteps above A4
# The -1 makes it go to the octave below
REFERENCE_A_FREQ = 440
REFERENCE_C_FREQ = REFERENCE_A_FREQ * 2 ** ((4 + 1 + 2 + 1) / 31 - 1)

SCALE = tuple(2 ** (i / 31) for i in range(31))


def make_suffix_cycle(entries: list[tuple[str, list[str]]]) -> list[str]:
    names: list[str] = []
    for letter, suffixes in entries:
        for suffix in suffixes:
            names.append(f"{letter}{suffix}")
    return names


NOTE_NAMES_SHARPWARDS = make_suffix_cycle(
    [
        ("C", ["", "t", "#", "#t", "x"]),
        ("D", ["", "t", "#", "#t", "x"]),
        ("E", ["", "t", "#"]),
        ("F", ["", "t", "#", "#t", "x"]),
        ("G", ["", "t", "#", "#t", "x"]),
        ("A", ["", "t", "#", "#t", "x"]),
        ("B", ["", "t", "#"]),
    ]
)

NOTE_NAMES_FLATWARDS = make_suffix_cycle(
    [
        ("C", ["d", "b"]),
        ("B", ["", "d", "b", "db", "bb"]),
        ("A", ["", "d", "b", "db", "bb"]),
        ("G", ["", "d", "b", "db", "bb"]),
        ("F", ["", "d", "b"]),
        ("E", ["", "d", "b", "db", "bb"]),
        ("D", ["", "d", "b", "db", "bb"]),
        ("C", [""]),
    ]
)

NAMES_TO_INDEX: dict[str, int] = {}
SHARPWARDS_NAMES_TO_INDEX: dict[str, int] = {}
FLATWARDS_NAMES_TO_INDEX: dict[str, int] = {}
for idx in range(31):
    sharp_name = NOTE_NAMES_SHARPWARDS[idx]
    flat_name = NOTE_NAMES_FLATWARDS[30 - idx]
    NAMES_TO_INDEX[sharp_name] = idx
    NAMES_TO_INDEX[flat_name] = idx
    SHARPWARDS_NAMES_TO_INDEX[sharp_name] = idx
    FLATWARDS_NAMES_TO_INDEX[flat_name] = idx

NOTE_PARSER: NoteParser = build_note_parser(
    NAMES_TO_INDEX,
    SCALE,
    REFERENCE_C_FREQ,
    bpm=BPM,
    sharpwards_names_to_index=SHARPWARDS_NAMES_TO_INDEX,
    flatwards_names_to_index=FLATWARDS_NAMES_TO_INDEX,
)

TICKS_PER_BEAT = 480


def ensure_midi_output_dir(output_path: str) -> str:
    base_path, _ = os.path.splitext(output_path)
    midi_dir = f"{base_path}_midi"
    if os.path.isdir(midi_dir):
        shutil.rmtree(midi_dir)
    elif os.path.exists(midi_dir):
        os.remove(midi_dir)
    os.makedirs(midi_dir, exist_ok=True)
    return midi_dir


def _encode_vlq(value: int) -> bytes:
    if value < 0:
        raise ValueError("VLQ value must be non-negative")
    buffer = value & 0x7F
    out = [buffer]
    value >>= 7
    while value:
        buffer = (value & 0x7F) | 0x80
        out.insert(0, buffer)
        value >>= 7
    return bytes(out)


def _clamp_midi_note(note_id: int) -> int:
    if note_id < 0:
        return 0
    if note_id > 127:
        return 127
    return note_id


def _seconds_to_ticks(seconds: float, bpm: float) -> int:
    beats = seconds * bpm / 60
    return int(round(beats * TICKS_PER_BEAT))


def write_single_track_midi(
    path: str,
    events: list[tuple[float, float, int, float]],
    bpm: float,
    *,
    track_name: str,
    channel: int,
) -> None:
    tempo = int(round(60_000_000 / bpm))
    track_events: list[tuple[int, int, bytes]] = []

    if track_name:
        name_bytes = track_name.encode("utf-8")
        track_events.append(
            (
                0,
                0,
                b"\xFF\x03" + _encode_vlq(len(name_bytes)) + name_bytes,
            )
        )

    track_events.append((0, 1, b"\xFF\x51\x03" + tempo.to_bytes(3, "big")))

    for start, duration, note_id, volume in events:
        if duration <= 0:
            continue
        start_tick = _seconds_to_ticks(start, bpm)
        end_tick = _seconds_to_ticks(start + duration, bpm)
        midi_note = _clamp_midi_note(note_id)
        velocity = int(round(volume * 127))
        if velocity <= 0:
            velocity = 1
        if velocity > 127:
            velocity = 127
        track_events.append(
            (start_tick, 3, bytes([0x90 | channel, midi_note, velocity]))
        )
        track_events.append((end_tick, 2, bytes([0x80 | channel, midi_note, 0])))

    track_events.sort(key=lambda item: (item[0], item[1]))
    track_data = bytearray()
    last_tick = 0
    for tick, _order, payload in track_events:
        delta = tick - last_tick
        track_data.extend(_encode_vlq(delta))
        track_data.extend(payload)
        last_tick = tick
    track_data.extend(_encode_vlq(0))
    track_data.extend(b"\xFF\x2F\x00")

    header = b"MThd" + struct.pack(">LHHH", 6, 0, 1, TICKS_PER_BEAT)
    track_chunk = b"MTrk" + struct.pack(">L", len(track_data)) + track_data
    with open(path, "wb") as handle:
        handle.write(header)
        handle.write(track_chunk)


def collect_track_events(
    clip: Clip, parser: NoteParser
) -> list[tuple[float, float, int, float]]:
    events: list[tuple[float, float, int, float]] = []
    for clip_note in clip.notes:
        note = clip_note.note
        if note.name is None:
            continue
        note_name = parser.note_name(note.name, note.octave)
        note_id = int(note_name.get_note_id())
        start_time = clip.start_time + clip_note.start
        events.append((start_time, note.duration, note_id, note.volume))
    return events


def make_state(note_name, note):
    return CustomState(
        pitch=note_name.get_pitch(),
        note_id=note_name.get_note_id(),
        volume=note.volume,
    )


# ============================================================
# SIMPLE TRACK
# ============================================================

simple_track_root_clip = Clip()

synth, adsr = build_synth_factories(
    SAMPLE_RATE,
    "sine",
    (0.01, 0.1, 0.8, 0.2),
)

simple_track = Track(
    "simple",
    1.0,
    simple_track_root_clip,
    sample_rate=SAMPLE_RATE,
    polyphony=8,
    synth_factory=synth,
    adsr_factory=adsr,
    event_bin_width=EVENT_BIN_WIDTH,
    block_size=512,
    retrigger_mode=RetriggerMode.ALLOW_TAILS,
)


# ============================================================
# BASIC EXAMPLE CONTENT
# ============================================================

simple_clip = Clip().insert_string(
    NOTE_PARSER,
    """
B.3.e
+7.e
+3.e
+5.e
R.e
-3.e
B.3.e
+5.e
R.e
+0.e/2
+4.e/2
-4.e
+0.e
""",

)

simple_track_root_clip.add_subclip_at(simple_clip, 0.0)

simple_track.schedule_own_root_clip(NOTE_PARSER.note_name, make_state)


# ============================================================
# MIDI EXPORT
# ============================================================

OUTPUT_WAV_PATH = "output_files/simple.wav"
MIDI_OUTPUT_DIR = ensure_midi_output_dir(OUTPUT_WAV_PATH)

write_single_track_midi(
    os.path.join(MIDI_OUTPUT_DIR, "simple.mid"),
    collect_track_events(simple_track_root_clip, NOTE_PARSER),
    BPM,
    track_name="simple",
    channel=0,
)


# ============================================================
# RENDER
# ============================================================

master = Master([simple_track])
frames = master.render_collect()

print("Playing...")
se = StereoAudio(frames, SAMPLE_RATE)
se.play(blocking=True)
se.export(OUTPUT_WAV_PATH, 24)