import json
import os

from src.audio.mixing import mix
from src.audio.stereo_audio import StereoAudio
from src.audio.event_scheduler import EventScheduler, RetriggerMode
from src.audio.note_parsing import build_note_parser
from src.audio.note_sequence import schedule_parsed_notes
from src.audio.sampler_synth import SamplerState, build_sampler_synth_factory
from src.audio.synth_factory import CustomState, build_synth_factories

SAMPLE_RATE = 48_000

BPM=70

# C5 is 8 edosteps above A4
# The -1 makes it go to the octave below
REFERENCE_A_FREQ = 440
REFERENCE_C_FREQ = REFERENCE_A_FREQ * 2 ** ((4+1+2+1) / 31 - 1)

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
        ("C", [""])
    ]
)

NAMES_TO_INDEX: dict[str, int] = {}
for idx in range(31):
    sharp_name = NOTE_NAMES_SHARPWARDS[idx]
    flat_name = NOTE_NAMES_FLATWARDS[30 - idx]
    NAMES_TO_INDEX[sharp_name] = idx
    NAMES_TO_INDEX[flat_name] = idx

NOTE_PARSER = build_note_parser(NAMES_TO_INDEX, SCALE, REFERENCE_C_FREQ, bpm=BPM)

def find_repo_root(start_path: str) -> str:
    current = os.path.abspath(start_path)
    while True:
        if os.path.isdir(os.path.join(current, ".git")):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            raise FileNotFoundError("Unable to locate repo root with .git directory.")
        current = parent


def resolve_sample_path(path: str, fallback_dir: str) -> str:
    expanded = os.path.expanduser(path)
    if os.path.exists(expanded):
        return expanded
    candidate = os.path.join(fallback_dir, os.path.basename(path))
    if os.path.exists(candidate):
        return candidate
    return expanded


REPO_ROOT = find_repo_root(os.path.dirname(__file__))
DRUM_CONFIG_PATH = os.path.join(
    REPO_ROOT,
    "input_files",
    "drumkits",
    "preset_1",
    "setup.json",
)

with open(DRUM_CONFIG_PATH, "r", encoding="utf-8") as handle:
    drum_config = json.load(handle)

DRUM_SAMPLE_DIR = os.path.dirname(DRUM_CONFIG_PATH)

DRUM_CONFIG_KEYS = {
    "K": "kick",
    "S": "snare",
    "HC": "hat_closed",
    "HO": "hat_open",
    "R": "rimshot",
    "CL": "clap",
    "TL": "tom_low",
    "TM": "tom_mid",
    "TH": "tom_high",
}

DRUM_SAMPLE_PATHS = {
    note: resolve_sample_path(drum_config[key]["file"], DRUM_SAMPLE_DIR)
    for note, key in DRUM_CONFIG_KEYS.items()
    if key in drum_config
}

DRUM_SAMPLE_VOLUMES = {
    note: 10 ** (drum_config[key].get("volume_db", 0) / 20)
    for note, key in DRUM_CONFIG_KEYS.items()
    if key in drum_config
}

DRUM_NAMES_TO_INDEX = {name: idx for idx, name in enumerate(DRUM_SAMPLE_PATHS)}
DRUM_SCALE = tuple(1.0 for _ in DRUM_NAMES_TO_INDEX)
DRUM_NOTE_PARSER = build_note_parser(DRUM_NAMES_TO_INDEX, DRUM_SCALE, 1.0, bpm=BPM)


# ============================================================
# SYNTHS
# ============================================================

synth_track1, adsr_track1 = build_synth_factories(
    SAMPLE_RATE,
    "triangle",
    (0.020, 0.080, 0.7, 0.180),
)

synth_track2, adsr_track2 = build_synth_factories(
    SAMPLE_RATE,
    "square",
    (0.040, 0.120, 0.6, 0.240),
)

synth_track3, adsr_track3 = build_synth_factories(
    SAMPLE_RATE,
    "saw",
    (0.010, 0.090, 0.65, 0.180),
)

synth_track4, adsr_track4 = build_synth_factories(
    SAMPLE_RATE,
    "saw",
    (0.015, 0.070, 0.55, 0.160),
)

drum_synth, _drum_sampler_config = build_sampler_synth_factory(
    SAMPLE_RATE,
    DRUM_SAMPLE_PATHS,
)


# ============================================================
# TRACKS / SCHEDULERS
# ============================================================

track1 = EventScheduler(
    SAMPLE_RATE, 16, synth_track1, adsr_track1, 1, 512, RetriggerMode.ATTACK_FROM_CURRENT_LEVEL
)

track2 = EventScheduler(
    SAMPLE_RATE, 16, synth_track2, adsr_track2, 1, 512, RetriggerMode.ATTACK_FROM_CURRENT_LEVEL
)

track3 = EventScheduler(
    SAMPLE_RATE, 16, synth_track3, adsr_track3, 1, 512, RetriggerMode.ATTACK_FROM_CURRENT_LEVEL
)

track4 = EventScheduler(
    SAMPLE_RATE, 16, synth_track4, adsr_track4, 1, 512, RetriggerMode.ATTACK_FROM_CURRENT_LEVEL
)

track5 = EventScheduler(
    SAMPLE_RATE,
    8,
    drum_synth,
    None,
    1,
    512,
    RetriggerMode.CUT_TAILS,
)


def make_state(note_name, note):
    return CustomState(
        pitch=note_name.get_pitch(),
        note_id=note_name.get_note_id(),
        volume=note.volume,
    )


def make_drum_state(note_name, note):
    sample_volume = DRUM_SAMPLE_VOLUMES.get(note_name.text, 1.0)
    return SamplerState(
        sample_id=note_name.text,
        volume=note.volume * sample_volume,
        note_id=note_name.get_note_id(),
    )


# ============================================================
# MUSIC (dot-joined notation, one note per token)
# ============================================================

BASS_LINES = """

""".splitlines()

HARMONY_LINES = """
""".splitlines()

MELODY_LINES = """
""".splitlines()

COUNTER_LINES = """
""".splitlines()

DRUM_LINES = """

# Swing drums
K.0.e:1.0 HC.0.e:1.0 K.0.e:1.0 HC.0.e:1.0 K.0.e:1.0 HC.0.e*2/3:1.0
K.0.e*(1/3+2/3):1.0 K.0.e*1/3:1.0 HC.0.e:1.0

""".splitlines()

# ============================================================
# SCHEDULE / RENDER
# ============================================================

acc = 0.0
for line in BASS_LINES:
    acc += schedule_parsed_notes(
        track1,
        acc,
        NOTE_PARSER.parse_lines(line),
        NOTE_PARSER.note_name,
        make_state,
    )

acc = 0.0
for line in HARMONY_LINES:
    acc += schedule_parsed_notes(
        track2,
        acc,
        NOTE_PARSER.parse_lines(line),
        NOTE_PARSER.note_name,
        make_state,
    )

acc = 0.0
for line in MELODY_LINES:
    acc += schedule_parsed_notes(
        track3,
        acc,
        NOTE_PARSER.parse_lines(line),
        NOTE_PARSER.note_name,
        make_state,
    )

acc = 0.0
for line in COUNTER_LINES:
    acc += schedule_parsed_notes(
        track4,
        acc,
        NOTE_PARSER.parse_lines(line),
        NOTE_PARSER.note_name,
        make_state,
    )

acc = 0.0
for line in DRUM_LINES:
    acc += schedule_parsed_notes(
        track5,
        acc,
        DRUM_NOTE_PARSER.parse_lines(line),
        DRUM_NOTE_PARSER.note_name,
        make_drum_state,
    )

frames1 = track1.render_collect()
frames2 = track2.render_collect()
frames3 = track3.render_collect()
frames4 = track4.render_collect()
frames5 = track5.render_collect()

frames = mix(
    (
        (10 ** (-3 / 20), frames1), # bass
        (10 ** (-16.5 / 20), frames2), # harmony
        (10 ** (-16.5 / 20), frames3), # melody
        (10 ** (-16.5 / 20), frames4), # counterpoint
        (10 ** (-3 / 20), frames5),  # drums
    )
)

print("Playing...")
se = StereoAudio(frames, SAMPLE_RATE)
se.play(blocking=True)
se.export("output_files/31_edo_song.wav", 24)
