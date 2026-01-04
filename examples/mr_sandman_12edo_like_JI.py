from src.audio.mixing import mix
from src.audio.stereo_audio import StereoAudio
from src.audio.event_scheduler import EventScheduler, RetriggerMode

from examples.note_parser import NoteName, parse_lines
from examples.synth_factory import CustomState, build_synth_factories

SAMPLE_RATE = 48_000


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


def write_notes(track: EventScheduler, start: float, notes):
    acc = start
    for (text, octave, dur) in notes:
        if text is None:
            acc += dur
            continue
        note_name = NoteName(text, octave)
        track.add_note(
            acc,
            dur,
            CustomState(
                pitch=note_name.get_pitch(),
                note_id=note_name.get_note_id(),
                volume=1,
            ),
        )
        acc += dur
    return acc - start


# ============================================================
# MUSIC (dot-joined notation, one or more notes per line)
# ============================================================

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
C#.3.e A.3.e E.3.e A.3.e
A.2.e E.3.e G.3.e E.3.e

# Make him the cutest that I've ever seen
D.3.e F.3.e A.3.e F.3.e
D.3.e F.3.e A.3.e F.3.e
G.2.e B.2.e D.3.e F.3.e A.3.e F.3.e D.3.e G.2.e

# Give him two lips
C.3.e E.3.e G.3.e E.3.e
A.2.e E.3.e C.3.e E.3.e

# like roses and clovers
G.2.e B.2.e D.3.e
C.3.e E.3.e C.3.e G.2.e E.2.e
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
B.4.e*2 A.4.e B.4.e

# Mr sandman 
B.4.q
A.4.q*3

# Give me a dream
B.4.e B.4.e A.4.e B.4.e+e*4

# Make him the cutest that I've ever seen
R.e
C.5.e C.5.e B.4.e C.5.e B.4.e
R.e B.4.e
F.4.e E.4.e E.4.e F.4.e+q*2

# Give him two lips
R.e
A.4.e A.4.e G.4.e A.4.e
R.e*2

# Like roses and clovers
G.4.e E.4.e E.4.e D.4.e
E.4.q G.4.q+e
""".splitlines()

# ============================================================
# SCHEDULE / RENDER
# ============================================================

acc = 0.0
for line in BASSLINE_LINES:
    acc += write_notes(track1, acc, parse_lines(line))

acc = 0.0
for line in MELODY_LINES:
    acc += write_notes(track2, acc, parse_lines(line))

frames1 = track1.render_collect()
frames2 = track2.render_collect()

frames = mix(((10 ** (-3 / 20), frames1), (10 ** (-28.5 / 20), frames2)))


print("Playing...")
se = StereoAudio(frames, SAMPLE_RATE)
se.play(blocking=True)
se.export("output_files/mr_sandman_ji.wav", 24)
