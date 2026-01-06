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

SCALE = tuple(2 ** (i / 12) for i in range(12))

NAMES_TO_INDEX = {
    "C": 0,
    "C#": 1,
    "Db": 1,
    "D": 2,
    "D#": 3,
    "Eb": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "Gb": 6,
    "G": 7,
    "G#": 8,
    "Ab": 8,
    "A": 9,
    "A#": 10,
    "Bb": 10,
    "B": 11,
}

NOTE_PARSER = build_note_parser(NAMES_TO_INDEX, SCALE, REFERENCE_C_FREQ, bpm=96)


# ============================================================
# SYNTHS
# ============================================================

create_new_synth_track1, create_new_adsr_track1 = build_synth_factories(
    SAMPLE_RATE,
    "sine",
    (0.020, 0.080, 0.7, 0.180),
)

create_new_synth_track2, create_new_adsr_track2 = build_synth_factories(
    SAMPLE_RATE,
    "triangle",
    (0.040, 0.120, 0.6, 0.240),
)

create_new_synth_track3, create_new_adsr_track3 = build_synth_factories(
    SAMPLE_RATE,
    "saw",
    (0.010, 0.090, 0.65, 0.180),
)

create_new_synth_track4, create_new_adsr_track4 = build_synth_factories(
    SAMPLE_RATE,
    "square",
    (0.015, 0.070, 0.55, 0.160),
)


# ============================================================
# TRACKS / SCHEDULERS
# ============================================================

track1 = EventScheduler(
    SAMPLE_RATE, 16, create_new_synth_track1, create_new_adsr_track1, 1, 512, RetriggerMode.ATTACK_FROM_CURRENT_LEVEL
)

track2 = EventScheduler(
    SAMPLE_RATE, 16, create_new_synth_track2, create_new_adsr_track2, 1, 512, RetriggerMode.ATTACK_FROM_CURRENT_LEVEL
)

track3 = EventScheduler(
    SAMPLE_RATE, 16, create_new_synth_track3, create_new_adsr_track3, 1, 512, RetriggerMode.ATTACK_FROM_CURRENT_LEVEL
)

track4 = EventScheduler(
    SAMPLE_RATE, 16, create_new_synth_track4, create_new_adsr_track4, 1, 512, RetriggerMode.ATTACK_FROM_CURRENT_LEVEL
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
# D minor -> Bb -> F -> C -> D minor -> G minor -> Bb -> A
D.2.q:0.9 A.1.q:0.75 D.2.q:0.85 A.1.q:0.75
Bb.1.q:0.85 F.1.q:0.7 Bb.1.q:0.8 F.1.q:0.7
F.2.q:0.85 C.2.q:0.7 F.2.q:0.8 C.2.q:0.7
C.2.q:0.85 G.1.q:0.7 C.2.q:0.8 G.1.q:0.7
D.2.q:0.9 A.1.q:0.75 D.2.q:0.85 A.1.q:0.75
G.1.q:0.85 D.2.q:0.7 G.1.q:0.8 D.2.q:0.7
Bb.1.q:0.85 F.1.q:0.7 Bb.1.q:0.8 F.1.q:0.7
A.1.q:0.9 E.2.q:0.7 A.1.q:0.85 E.2.q:0.7
""".splitlines()

HARMONY_LINES = """
F.3.e:0.45 A.3.e:0.45 D.4.e:0.45 A.3.e:0.4 F.3.e:0.4 A.3.e:0.4 D.4.e:0.4 A.3.e:0.4
F.3.e:0.42 Bb.3.e:0.42 D.4.e:0.42 F.4.e:0.4 D.4.e:0.4 Bb.3.e:0.4 F.3.e:0.4 D.4.e:0.4
A.3.e:0.42 C.4.e:0.42 F.4.e:0.42 A.4.e:0.4 F.4.e:0.4 C.4.e:0.4 A.3.e:0.4 C.4.e:0.4
G.3.e:0.42 C.4.e:0.42 E.4.e:0.42 G.4.e:0.4 E.4.e:0.4 C.4.e:0.4 G.3.e:0.4 C.4.e:0.4
F.3.e:0.45 A.3.e:0.45 D.4.e:0.45 A.3.e:0.4 F.3.e:0.4 A.3.e:0.4 D.4.e:0.4 A.3.e:0.4
Bb.2.e:0.42 D.3.e:0.42 G.3.e:0.42 Bb.3.e:0.4 G.3.e:0.4 D.3.e:0.4 Bb.2.e:0.4 D.3.e:0.4
F.3.e:0.42 Bb.3.e:0.42 D.4.e:0.42 F.4.e:0.4 D.4.e:0.4 Bb.3.e:0.4 F.3.e:0.4 D.4.e:0.4
C#.3.e:0.42 E.3.e:0.42 A.3.e:0.42 C#.4.e:0.4 A.3.e:0.4 E.3.e:0.4 C#.3.e:0.4 E.3.e:0.4
""".splitlines()

MELODY_LINES = """
A.4.q:0.7 D.5.e:0.85 F.5.e:0.9 A.5.q:0.95 G.5.e:0.85 F.5.e:0.8
F.5.q:0.85 Eb.5.e:0.75 D.5.e:0.7 C.5.q:0.75 D.5.q:0.85
A.4.q:0.7 C.5.e:0.8 F.5.e:0.9 A.5.q:0.95 G.5.e:0.85 F.5.e:0.8
E.5.q:0.8 D.5.e:0.75 C.5.e:0.7 B.4.q:0.7 C.5.q:0.8
A.4.q:0.7 D.5.e:0.85 F.5.e:0.9 A.5.q:0.95 G.5.e:0.85 F.5.e:0.8
G.5.q:0.85 Bb.5.e:0.9 A.5.e:0.85 G.5.q:0.8 F.5.q:0.75
D.5.q:0.8 F.5.e:0.85 G.5.e:0.9 Bb.5.q:0.9 A.5.q:0.85
C#.5.q:0.8 E.5.e:0.85 A.5.e:0.9 G.5.q:0.85 F.5.q:0.75
""".splitlines()

COUNTER_LINES = """
D.4.q:0.5 R.e A.4.e:0.55 F.4.q:0.5 R.e D.4.e:0.5
Bb.3.q:0.5 R.e D.4.e:0.55 F.4.q:0.5 R.e Bb.3.e:0.5
F.4.q:0.5 R.e A.4.e:0.55 C.5.q:0.5 R.e A.4.e:0.5
C.4.q:0.5 R.e E.4.e:0.55 G.4.q:0.5 R.e E.4.e:0.5
D.4.q:0.5 R.e A.4.e:0.55 F.4.q:0.5 R.e D.4.e:0.5
G.3.q:0.5 R.e D.4.e:0.55 Bb.4.q:0.5 R.e G.3.e:0.5
Bb.3.q:0.5 R.e D.4.e:0.55 F.4.q:0.5 R.e Bb.3.e:0.5
A.3.q:0.5 R.e C#.4.e:0.55 E.4.q:0.5 R.e A.3.e:0.5
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

frames1 = track1.render_collect()
frames2 = track2.render_collect()
frames3 = track3.render_collect()
frames4 = track4.render_collect()

frames = mix(
    (
        (10 ** (-7 / 20), frames1),
        (10 ** (-15 / 20), frames2),
        (10 ** (-10 / 20), frames3),
        (10 ** (-13 / 20), frames4),
    )
)

print("Playing...")
se = StereoAudio(frames, SAMPLE_RATE)
se.play(blocking=True)
se.export("output_files/starfield_waltz.wav", 24)
