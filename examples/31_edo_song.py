import json
import os

from src.audio.stereo_audio import StereoAudio
from src.audio.note_parsing import build_note_parser
from src.audio.sampler_synth import SamplerState, build_sampler_synth_factory
from src.audio.synth_factory import CustomState, build_synth_factories
from src.audio.arrangement import Clip, Master, Track
from src.audio.event_scheduler import RetriggerMode

SAMPLE_RATE = 48_000

EVENT_BIN_WIDTH=1

BPM=136

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

clip1 = Clip()
clip2 = Clip()
clip3 = Clip()
clip4 = Clip()
clip5 = Clip()

track1 = Track(
    "bass",
    10 ** (-3 / 20),
    clip1,
    sample_rate=SAMPLE_RATE,
    polyphony=16,
    synth_factory=synth_track1,
    adsr_factory=adsr_track1,
    event_bin_width=EVENT_BIN_WIDTH,
    block_size=512,
    retrigger_mode=RetriggerMode.ATTACK_FROM_CURRENT_LEVEL,
)

track2 = Track(
    "harmony",
    10 ** (-16.5 / 20),
    clip2,
    sample_rate=SAMPLE_RATE,
    polyphony=16,
    synth_factory=synth_track2,
    adsr_factory=adsr_track2,
    event_bin_width=EVENT_BIN_WIDTH,
    block_size=512,
    retrigger_mode=RetriggerMode.ATTACK_FROM_CURRENT_LEVEL,
)

track3 = Track(
    "melody",
    10 ** (-16.5 / 20),
    clip3,
    sample_rate=SAMPLE_RATE,
    polyphony=16,
    synth_factory=synth_track3,
    adsr_factory=adsr_track3,
    event_bin_width=EVENT_BIN_WIDTH,
    block_size=512,
    retrigger_mode=RetriggerMode.ATTACK_FROM_CURRENT_LEVEL,
)

track4 = Track(
    "counterpoint",
    10 ** (-16.5 / 20),
    clip4,
    sample_rate=SAMPLE_RATE,
    polyphony=16,
    synth_factory=synth_track4,
    adsr_factory=adsr_track4,
    event_bin_width=EVENT_BIN_WIDTH,
    block_size=512,
    retrigger_mode=RetriggerMode.ATTACK_FROM_CURRENT_LEVEL,
)

track5 = Track(
    "drums",
    10 ** (-3 / 20),
    clip5,
    sample_rate=SAMPLE_RATE,
    polyphony=8,
    synth_factory=drum_synth,
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

BASS_CLIP = Clip().insert_string(
    NOTE_PARSER,
    """

""",
)

HARMONY_CLIP = Clip().insert_string(
    NOTE_PARSER,
    """
""",
)

MELODY_CLIP = Clip().insert_string(
    NOTE_PARSER,
    """
""",
)

COUNTER_CLIP = Clip().insert_string(
    NOTE_PARSER,
    """
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



# ============================================================
# SCHEDULE / RENDER
# ============================================================

clip1.add_subclip_at(BASS_CLIP, 0.0)
clip2.add_subclip_at(HARMONY_CLIP, 0.0)
clip3.add_subclip_at(MELODY_CLIP, 0.0)
clip4.add_subclip_at(COUNTER_CLIP, 0.0)
clip5.add_subclip_at(DRUM_CLIP_A, 0.0).add_subclip_at(DRUM_CLIP_B, 0.0)

track1.schedule_clip(NOTE_PARSER.note_name, make_state)
track2.schedule_clip(NOTE_PARSER.note_name, make_state)
track3.schedule_clip(NOTE_PARSER.note_name, make_state)
track4.schedule_clip(NOTE_PARSER.note_name, make_state)
track5.schedule_clip(DRUM_NOTE_PARSER.note_name, make_drum_state)

master = Master([track1, track2, track3, track4, track5])
frames = master.render_collect()

print("Playing...")
se = StereoAudio(frames, SAMPLE_RATE)
se.play(blocking=True)
se.export("output_files/31_edo_song.wav", 24)
