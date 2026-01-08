from src.audio.mixing import mix
from src.audio.stereo_audio import StereoAudio
from src.audio.event_scheduler import EventScheduler, RetriggerMode
from src.audio.note_parsing import build_note_parser
from src.audio.note_sequence import schedule_parsed_notes
from src.audio.synth_factory import CustomState, build_synth_factories

SAMPLE_RATE = 48_000

# C5 is 3 semitones above A4
# The -1 makes it go to the octave below
REFERENCE_A_FREQ = 440
REFERENCE_C_FREQ = REFERENCE_A_FREQ * 2 ** (3 / 12 - 1)

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

NOTE_PARSER = build_note_parser(NAMES_TO_INDEX, SCALE, REFERENCE_C_FREQ, bpm=110)


# ============================================================
# SYNTHS
# ============================================================

create_new_synth_track1, create_new_adsr_track1 = build_synth_factories(
    SAMPLE_RATE,
    "triangle",
    (0.030, 0.080, 0.7, 0.220),
)

create_new_synth_track2, create_new_adsr_track2 = build_synth_factories(
    SAMPLE_RATE,
    "triangle",
    (0.020, 0.090, 0.6, 0.200),
)

create_new_synth_track3, create_new_adsr_track3 = build_synth_factories(
    SAMPLE_RATE,
    "saw",
    (0.010, 0.070, 0.55, 0.160),
)


# ============================================================
# TRACKS / SCHEDULERS
# ============================================================

track1 = EventScheduler(
    SAMPLE_RATE,
    12,
    create_new_synth_track1,
    create_new_adsr_track1,
    1,
    512,
    RetriggerMode.ATTACK_FROM_CURRENT_LEVEL,
)

track2 = EventScheduler(
    SAMPLE_RATE,
    12,
    create_new_synth_track2,
    create_new_adsr_track2,
    1,
    512,
    RetriggerMode.ATTACK_FROM_CURRENT_LEVEL,
)

track3 = EventScheduler(
    SAMPLE_RATE,
    12,
    create_new_synth_track3,
    create_new_adsr_track3,
    1,
    512,
    RetriggerMode.ATTACK_FROM_CURRENT_LEVEL,
)


def make_state(note_name, note):
    return CustomState(
        pitch=note_name.get_pitch(),
        note_id=note_name.get_note_id(),
        volume=note.volume,
    )


# ============================================================
# MUSIC (dot-joined notation, one note per token)
# ============================================================

BASS_LINES = """
# Rooted progression with subtle 31-EDO spice
C.2.q:0.95 G.1.q:0.8 C.2.q:0.9 G.1.q:0.8
F.1.q:0.9 C.2.q:0.75 F.1.q:0.85 C.2.q:0.75
Bb.1.q:0.9 F.1.q:0.75 Bb.1.q:0.85 F.1.q:0.75
G.1.q:0.9 D.2.q:0.75 G.1.q:0.85 D.2.q:0.75

# Color the cadence with sharp/flat inflections
C.2.q:0.95 G.1.q:0.8 C.2.q:0.9 G.1.q:0.8
F.1.q:0.9 C.2.q:0.75 F.1.q:0.85 C.2.q:0.75
Bb.1.q:0.9 F.1.q:0.75 Bb.1.q:0.85 F.1.q:0.75
G.1.q:0.9 Dt.2.q:0.75 G.1.q:0.85 D.2.q:0.75
""".splitlines()

HARMONY_LINES = """
C.3.e:0.48 E.3.e:0.45 G.3.e:0.42 C.4.e:0.4
F.3.e:0.46 A.3.e:0.44 C.4.e:0.42 F.4.e:0.4
Bb.2.e:0.46 D.3.e:0.44 F.3.e:0.42 Bb.3.e:0.4
G.2.e:0.46 B.2.e:0.44 D.3.e:0.42 G.3.e:0.4
C.3.e:0.48 Et.3.e:0.45 G.3.e:0.42 C.4.e:0.4
F.3.e:0.46 A.3.e:0.44 Ct.4.e:0.42 F.4.e:0.4
Bb.2.e:0.46 D.3.e:0.44 F.3.e:0.42 Bb.3.e:0.4
G.2.e:0.46 Bt.2.e:0.44 D.3.e:0.42 G.3.e:0.4
""".splitlines()

MELODY_LINES = """
# Singable motif with occasional microtonal colors
C.4.q:0.85 E.4.e:0.8 G.4.e:0.85 A.4.q:0.85 G.4.e:0.8 E.4.e:0.8
F.4.q:0.85 A.4.e:0.8 C.5.e:0.85 D.5.q:0.85 C.5.e:0.8 A.4.e:0.8
Bb.4.q:0.85 D.5.e:0.8 F.5.e:0.85 G.5.q:0.85 F.5.e:0.8 D.5.e:0.8
G.4.q:0.85 B.4.e:0.8 D.5.e:0.85 E.5.q:0.85 D.5.e:0.8 B.4.e:0.8

# Short enharmonic inflections
C.5.e:0.85 Ct.5.e:0.8 D.5.q:0.85 E.5.e:0.8 Et.5.e:0.8 F.5.q:0.85
Bb.4.e:0.85 Bdb.4.e:0.8 C.5.q:0.85 D.5.e:0.8 Dt.5.e:0.8 Eb.5.q:0.85

# Flatward cadence
Ab.4.q:0.85 C.5.e:0.8 Eb.5.e:0.85 F.5.q:0.85 Eb.5.e:0.8 C.5.e:0.8
G.4.q:0.85 Bt.4.e:0.8 D.5.e:0.85 C.5.q:0.85 Bb.4.e:0.8 Ab.4.e:0.8
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

frames1 = track1.render_collect()
frames2 = track2.render_collect()
frames3 = track3.render_collect()

frames = mix(
    (
        (10 ** (-4 / 20), frames1),
        (10 ** (-12 / 20), frames2),
        (10 ** (-9 / 20), frames3),
    )
)

print("Playing...")
se = StereoAudio(frames, SAMPLE_RATE)
se.play(blocking=True)
se.export("output_files/aurora_31edo.wav", 24)
