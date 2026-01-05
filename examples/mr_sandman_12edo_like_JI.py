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
REFERENCE_C_FREQ = REFERENCE_A_FREQ * 9 / 16

#  ./__inenv python experiments/pick_ji.py --primes=2,3,5 --max-int=64 --edo=12 --per-step=1 --format ratios
SCALE = [
    1 / 1,
    16 / 15,
    9 / 8,
    32 / 27,
    5 / 4,
    4 / 3,
    45 / 32,
    3 / 2,
    8 / 5,
    27 / 16,
    16 / 9,
    15 / 8,
]

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

NOTE_PARSER = build_note_parser(NAMES_TO_INDEX, SCALE, REFERENCE_C_FREQ)


# ============================================================
# SYNTHS
# ============================================================

create_new_synth_track1, create_new_adsr_track1 = build_synth_factories(
    SAMPLE_RATE,
    "triangle",
    (0.050, 0.050, 0.5, 0.200),
)

create_new_synth_track2, create_new_adsr_track2 = build_synth_factories(
    SAMPLE_RATE,
    "saw",
    (0.015, 0.100, 0.7, 0.200),
)


# ============================================================
# TRACKS / SCHEDULERS
# ============================================================

# For now: track1 = bassline, track2 = melody

track1 = EventScheduler(
    SAMPLE_RATE, 16, create_new_synth_track1, create_new_adsr_track1, 1, 512, RetriggerMode.ATTACK_FROM_CURRENT_LEVEL
)

track2 = EventScheduler(
    SAMPLE_RATE, 16, create_new_synth_track2, create_new_adsr_track2, 1, 512, RetriggerMode.ATTACK_FROM_CURRENT_LEVEL
)


def make_state(note_name, note):
    return CustomState(
        pitch=note_name.get_pitch(),
        note_id=note_name.get_note_id(),
        volume=note.volume,
    )


# ============================================================
# MUSIC (dot-joined notation, one or more notes per line)
# ============================================================

# Credit:
# https://sheetsfree.com/sheets/C/The%20Chordettes%20-%20Mister%20Sandman%20%28Lead%20Sheet%29.pdf
# https://tabs.ultimate-guitar.com/tab/the-chordettes/mr-sandman-chords-847467

BASSLINE_LINES = """

# intro bar 1
C.3.e*6 C.3.e*2
F.3.e*4 G.2.e*4

# intro bar 2
C.3.e*2 C.3.e*2
C.3.e*2 C.3.e*2
F.3.e*2 F.3.e*2
G.2.e*2 G.2.e*2

# Mr Sandman
C.3.e G.3.e E.3.e G.3.e
C.3.e G.3.e E.3.e G.3.e

# Give me a dream

B.2.e F#.3.e D#.3.e F#.3.e
B.2.e B.2.e C#.3.e D#.3.e


# Make him the cutest that I've ever seen
E.3.e B.3.e G#.3.e B.3.e
E.3.e B.3.e G#.3.e B.3.e

A.2.e E.3.e C#.3.e E.3.e
A.2.e A.2.e B.2.e C#.3.e

# Give him two lips
D.3.e A.3.e F#.3.e A.3.e
C.4.e A.3.e F#.3.e D.3.e

# like roses and clovers
B.2.e G.2.e B.2.e D.3.e
F.3.e G.2.e A.2.e B.2.e

# and tell me that my lonesome nights are over
C.3.e G.3.e E.3.e G.3.e
C.3.e G.3.e E.3.e G.3.e
Ab.3.e Eb.3.e C.3.e Ab.2.e
G.2.e B.2.e D.3.e F.3.e

# sandman (sandman)
C.3.e G.3.e E.3.e G.3.e
C.3.e G.3.e E.3.e G.3.e

# I'm so alone

B.2.e F#.3.e D#.3.e F#.3.e
B.2.e B.2.e C#.3.e D#.3.e


# I've got nobody to call my own
E.3.e B.3.e G#.3.e B.3.e
E.3.e B.3.e G#.3.e B.3.e

A.2.e E.3.e C#.3.e E.3.e
A.2.e A.2.e B.2.e C#.3.e

# please turn on your magic
D.3.e A.3.e F.3.e A.3.e 
D.3.e A.3.e F.3.e A.3.e

# beam ... mr
Ab.3.e F.3.e Ab.3.e C.4.e
Ab.3.e F.3.e C.3.e Ab.2.e

# sand man bring me a
C.3.e G.3.e E.3.e G.3.e
D.3.e F.3.e G.3.e B.3.e

# dream
C.4.e G.3.e E.3.e C.3.e

C.2.e*4


""".splitlines()

MELODY_LINES = """
# Intro bar 1
C.4.e E.4.e G.4.e B.4.e
A.4.e G.4.e E.4.e C.4.e
D.4.e F.4.e A.4.e C.5.e
B.4.e*4

# Intro bar 2
C.4.e E.4.e G.4.e B.4.e
A.4.e G.4.e E.4.e C.4.e
D.4.e F.4.e A.4.e C.5.e
B.4.e*2 G.4.e A.4.e

# Mr sandman 
B.4.q
A.4.q*3

# Give me a dream
B.4.e B.4.e A.4.e B.4.e+e*4

# Make him the cutest that I've ever seen
R.e
C.5.e C.5.e B.4.e C.5.e B.4.e
R.e B.4.e
F.4.e F.4.e E.4.e F.4.e+q*2

# Give him two lips
R.e
B.4.e B.4.e A.4.e B.4.e
R.e*2

# Like roses and clovers
A.4.e E.4.e E.4.e D.4.e
E.4.q G.4.q+e

# and tell me that my lonesome nights are over
R.e
D.5.e D.5.e C.5.e D.5.e C.5.e D.5.e C.5.e Eb.5.q Eb.5.q E.5.e D.5.q R.e

# sandman (sandman)

B.4.q A.4.q B.5.q A.5.q

# i'm so alone
B.4.e B.4.e A.4.e B.4.e+e*4

# I've got no-bo-dy to call my own
R.e
C.5.e C.5.e B.4.e C.5.e B.4.e
R.e B.4.e
F.4.e+e  E.4.e F.4.e+q*2

# Please turn on your magic
D.4.q F.4.e A.4.q C.5.e C.5.e C.5.e

# beam ... mr
D.5.e*6  C.5.e D.5.e

# sand man
E.5.q E.5.q 

# give me a 
E.5.e C.5.e D.5.e*2

# dream
C.5.e*4
R.e*4

""".splitlines()

# ============================================================
# SCHEDULE / RENDER
# ============================================================

acc = 0.0
for line in BASSLINE_LINES:
    acc += schedule_parsed_notes(
        track1,
        acc,
        NOTE_PARSER.parse_lines(line),
        NOTE_PARSER.note_name,
        make_state,
    )

acc = 0.0
for line in MELODY_LINES:
    acc += schedule_parsed_notes(
        track2,
        acc,
        NOTE_PARSER.parse_lines(line),
        NOTE_PARSER.note_name,
        make_state,
    )

frames1 = track1.render_collect()
frames2 = track2.render_collect()

frames = mix(((10 ** (-3 / 20), frames1), (10 ** (-19.5 / 20), frames2)))


print("Playing...")
se = StereoAudio(frames, SAMPLE_RATE)
se.play(blocking=True)
se.export("output_files/mr_sandman_ji.wav", 24)
