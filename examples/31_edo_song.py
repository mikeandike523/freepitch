import json
import os

from src.audio.stereo_audio import StereoAudio
from src.audio.note_parsing import build_note_parser
from src.audio.sampler_synth import SamplerState, build_sampler_synth_factory
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
    "RS": "rimshot",
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
# TRACKS / SCHEDULERS
# ============================================================

bass_track_root_clip = Clip()
harmony_track_root_clip = Clip()
melody_track_root_clip = Clip()
track4_root_clip = Clip()
drum_track_root_clip = Clip()

synth_for_bass, adsr_for_bass = build_synth_factories(
    SAMPLE_RATE,
    "triangle",
    (0.020, 0.080, 0.7, 0.180),
)

synth_for_harmony, adsr_for_harmony = build_synth_factories(
    SAMPLE_RATE,
    "square",
    (0.040, 0.120, 0.6, 0.240),
)

synth_for_melody, adsr_for_melody = build_synth_factories(
    SAMPLE_RATE,
    "saw",
    (0.010, 0.090, 0.65, 0.180),
)

drum_sampler_synth, _drum_sampler_synth_config = build_sampler_synth_factory(
    SAMPLE_RATE,
    DRUM_SAMPLE_PATHS,
)

bass_track = Track(
    "bass",
    10 ** (-1.5 / 20),
    bass_track_root_clip,
    sample_rate=SAMPLE_RATE,
    polyphony=16,
    synth_factory=synth_for_bass,
    adsr_factory=adsr_for_bass,
    event_bin_width=EVENT_BIN_WIDTH,
    block_size=512,
    retrigger_mode=RetriggerMode.ATTACK_FROM_CURRENT_LEVEL,
)

harmony_track = Track(
    "harmony",
    10 ** (-16.5 / 20),
    harmony_track_root_clip,
    sample_rate=SAMPLE_RATE,
    polyphony=16,
    synth_factory=synth_for_harmony,
    adsr_factory=adsr_for_harmony,
    event_bin_width=EVENT_BIN_WIDTH,
    block_size=512,
    retrigger_mode=RetriggerMode.ATTACK_FROM_CURRENT_LEVEL,
)

melody_track = Track(
    "melody",
    10 ** (-16.5 / 20),
    melody_track_root_clip,
    sample_rate=SAMPLE_RATE,
    polyphony=16,
    synth_factory=synth_for_melody,
    adsr_factory=adsr_for_melody,
    event_bin_width=EVENT_BIN_WIDTH,
    block_size=512,
    retrigger_mode=RetriggerMode.ATTACK_FROM_CURRENT_LEVEL,
)

drum_track = Track(
    "drums",
    10 ** (-9 / 20),
    drum_track_root_clip,
    sample_rate=SAMPLE_RATE,
    polyphony=8,
    synth_factory=drum_sampler_synth,
    adsr_factory=None,
    event_bin_width=1,
    block_size=512,
    retrigger_mode=RetriggerMode.CUT_TAILS,
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

bass_clip_1 = Clip().insert_string(
    NOTE_PARSER,
    """
A.2.e:1.0 C.3.e:1.0 E.3.e:1.0 G.3.e:1.0
A.3.e:1.0 Ed.3.e:1.0 Eb.3.e:1.0 C.3.e:1.0
Bdb.2.e:1.0 C.3.e:1.0 Bb.2.e:1.0 G.2.e:1.0
A.2.e:1.0 E.2.e:1.0 Dt.2.e:1.0 E.2.e:1.0
""",
)

# === NEW: Harmony + Melody content (fills were previously empty) ===

harmony_clip_1 = Clip().insert_string(
    NOTE_PARSER,
    """
A.3.q:0.8 E.4.q:0.8 G.4.q:0.8 C.5.q:0.8
A.3.h:0.8 E.4.h:0.8

A.3.q:0.8 Ed.4.q:0.8 G.4.q:0.8 C.5.q:0.8
A.3.h:0.8 Ed.4.h:0.8

Bdb.3.q:0.8 F.4.q:0.8 Bb.4.q:0.8 D.5.q:0.8
Bdb.3.h:0.8 F.4.h:0.8

A.3.q:0.8 E.4.q:0.8 Dt.4.q:0.8 E.4.q:0.8
A.3.h:0.8 E.4.h:0.8
""",
)

melody_clip_1 = Clip().insert_string(
    NOTE_PARSER,
    """
E.5.e:1.0 G.5.e:1.0 A.5.q:1.0
C.6.e:1.0 Bdb.5.e:1.0 A.5.q:1.0

G.5.e:1.0 A.5.e:1.0 Ed.6.q:1.0
E.6.e:1.0 Dt.6.e:1.0 C.6.q:1.0

Bb.5.e:1.0 D.6.e:1.0 F.6.q:1.0
G.6.e:1.0 F.6.e:1.0 D.6.q:1.0

E.5.e:1.0 G.5.e:1.0 A.5.e:1.0 C.6.e:1.0
Dt.6.q:1.0 A.5.q:1.0
""",
)

DRUM_CLIP_A = Clip().insert_string(
    DRUM_NOTE_PARSER,
    """

# Swing drums

# Intro
K.0.e:1.0 HC.0.e:1.0 K.0.e:1.0 HC.0.e:1.0 K.0.e:1.0 HC.0.e*2/3:1.0
K.0.e*(1/3+2/3):1.0 K.0.e*1/3:1.0 HC.0.e:1.0

K.0.e:1.0 HC.0.e:1.0 K.0.e:1.0 HC.0.e:1.0 K.0.e:1.0 HC.0.e*2/3:1.0
K.0.e*(1/3+2/3):1.0 K.0.e*1/3:1.0 HC.0.e:1.0

K.0.e:1.0 HC.0.e:1.0 K.0.e:1.0 HC.0.e:1.0 K.0.e:1.0 HC.0.e*2/3:1.0
K.0.e*(1/3+2/3):1.0 K.0.e*1/3:1.0 HC.0.e:1.0

TH.0.e/3:1.0 TL.0.e/3:1.0 TM.0.e/3:1.0
TM.0.e/3:1.0 TM.0.e/3:1.0 TM.0.e/3:1.0

TH.0.e/3:1.0 TL.0.e/3:1.0 TM.0.e/3:1.0
TM.0.e/3:1.0 TM.0.e/3:1.0 TM.0.e/3:1.0

TH.0.e/3:1.0 TM.0.e/3:1.0 TM.0.e/3:1.0
TH.0.e/3:1.0 TM.0.e/3:1.0 TM.0.e/3:1.0
TH.0.e/3:1.0 TM.0.e/3:1.0 TM.0.e/3:1.0
TH.0.e/3:1.0 TM.0.e/3:1.0 TM.0.e/3:1.0

""",
)

DRUM_CLIP_B = Clip().insert_string(
    DRUM_NOTE_PARSER,
    """

RS.0.q:1.0 RS.0.q:1.0 RS.0.q:1.0 RS.0.e:1.0 HO.0.e:1.0
RS.0.q:1.0 RS.0.q:1.0 RS.0.q:1.0 RS.0.e:1.0 HO.0.e:1.0
RS.0.q:1.0 RS.0.q:1.0 RS.0.q:1.0 RS.0.e:1.0 HO.0.e:1.0
RS.0.q:1.0 RS.0.q:1.0 RS.0.q:1.0 RS.0.e:1.0 RS.0.e/2:1.0 RS.0.e/2:1.0

""",
)

# === Arrangement-y clip commands: layering subclips ===
drum_clip_1 = Clip().add_subclip_at(DRUM_CLIP_A, 0.0).add_subclip_at(DRUM_CLIP_B, 0.0)

# ============================================================
# SCHEDULE / RENDER
# ============================================================

# === Arrangement-y clip commands: placing clips on root timelines ===
# Your bass enters after one drum_clip_1 duration (i.e., drums do an "intro" first).
bass_track_root_clip.add_subclip_at(bass_clip_1, drum_clip_1.duration)

# Harmony + melody start at the beginning (you can offset if you want them to wait for the intro too).
harmony_track_root_clip.add_subclip_at(harmony_clip_1, 0.0)
melody_track_root_clip.add_subclip_at(melody_clip_1, 0.0)

# Drums: two back-to-back repeats of the combined A+B layer
drum_track_root_clip.add_subclip_next(drum_clip_1).add_subclip_next(drum_clip_1)

# === Event scheduling for each track ===
bass_track.schedule_own_root_clip(NOTE_PARSER.note_name, make_state)
harmony_track.schedule_own_root_clip(NOTE_PARSER.note_name, make_state)
melody_track.schedule_own_root_clip(NOTE_PARSER.note_name, make_state)
drum_track.schedule_own_root_clip(DRUM_NOTE_PARSER.note_name, make_drum_state)

# === Mixdown/render ===
master = Master([bass_track, harmony_track, melody_track, drum_track])
frames = master.render_collect()

print("Playing...")
se = StereoAudio(frames, SAMPLE_RATE)
se.play(blocking=True)
se.export("output_files/31_edo_song.wav", 24)
