from dataclasses import dataclass
from src.audio.mixing import mix
from src.audio.stereo_audio import StereoAudio
from src.audio.core import AudioBuffer
from src.audio.exp_adsr import ExpADSR
import math
from src.audio.helpers.create_synth import create_synth
from src.audio.event_scheduler import EventScheduler
import re

SAMPLE_RATE = 48_000

REFERENCE_A_FREQ = 440


# C5 is 3 semitones above A4
# The -1 makes it go to the octave below
REFERENCE_C_FREQ = REFERENCE_A_FREQ * 9/16


@dataclass(slots=False)
class CustomState:
    volume: float
    pitch: float
    note_id: int


#  ./__inenv python experiments/pick_ji.py --primes=2,3,5 --max-int=64 --edo=12 --per-step=1 --format ratios
SCALE = [
1/1,
16/15,
9/8,
32/27,
5/4,
4/3,
45/32,
3/2,
8/5,
27/16,
16/9,
15/8,
]

NAMES_TO_INDEX= {
    "C":0,
    "C#": 1, "Db": 1,
    "D": 2,
    "D#": 3, "Eb": 3,
    "E": 4,
    "F": 5,
    "F#": 6, "Gb": 6,
    "G": 7,
    "G#": 8, "Ab": 8,
    "A": 9,
    "A#": 10, "Bb": 10,
    "B": 11
}

@dataclass(slots=True)
class NoteName:
    text: str
    octave: int

    def __init__(self, text: str, octave: int):
        if text not in NAMES_TO_INDEX:
            raise ValueError(f"Invalid note name: {text}")
        self.text = text
        self.octave = octave

    def get_index_in_octave(self) -> int:
        return NAMES_TO_INDEX[self.text]

    def get_note_id(self):
        return self.octave * 12 + self.get_index_in_octave()

    def get_pitch(self):
        return SCALE[self.get_index_in_octave()] * 2**(self.octave-4) * REFERENCE_C_FREQ

    def __repr__(self):
        return f"{self.text}{self.octave}"



# Just the waveform part
# ADSR is handeled automatically
# The create_synth_with_adsr function takes care of the multiplying internally
def process_callback_track1(
    sample_rate: int, n: int, num_samples: int, state: CustomState  # unused
) -> AudioBuffer:
    buffer = []
    for i in range(num_samples):
        t = (n + i) / sample_rate
        value = state.volume * math.sin(state.pitch * 2 * math.pi * t)
        buffer.append((value, value))  # stereo
    return buffer


def create_new_synth_track1():
    return create_synth(
        SAMPLE_RATE,
        CustomState(pitch=0.0, volume=1.0, note_id=-1),
        process_callback_track1,
        reset_callback=None,
    )


def create_new_adsr_track1():
    return ExpADSR(SAMPLE_RATE, 0.015, 0.100, 1, 0.200, num_tau=5.0)


# Just the waveform part
# ADSR is handeled automatically
# The create_synth_with_adsr function takes care of the multiplying internally
def process_callback_track2(
    sample_rate: int, n: int, num_samples: int, state: CustomState  # unused
) -> AudioBuffer:
    buffer = []
    for i in range(num_samples):
        t = (n + i) / sample_rate
        # value = state.volume*math.sin(state.pitch *2 * math.pi*t)
        phase = (state.pitch * 2 * math.pi * t) % (2 * math.pi)
        # sawtooth wave
        value_unscaled = phase / math.pi
        if phase > math.pi:
            value_unscaled = -1 + (phase - math.pi) / math.pi
        value = value_unscaled * state.volume
        buffer.append((value, value))  # stereo
    return buffer


def create_new_synth_track2():
    return create_synth(
        SAMPLE_RATE,
        CustomState(pitch=0.0, volume=1.0, note_id=-1),
        process_callback_track2,
        reset_callback=None,
    )


def create_new_adsr_track2():
    return ExpADSR(SAMPLE_RATE, 0.015, 0.100, 1, 0.200, num_tau=5.0)


# Tracks have polyphony

# We could put melody and accompanyment on the same track

# But this gives us more flexibility

# For now: track1 = bassline, track2 = melody

track1 = EventScheduler(
    SAMPLE_RATE, 16, create_new_synth_track1, create_new_adsr_track1, 1, 512
)

track2 = EventScheduler(
    SAMPLE_RATE, 16, create_new_synth_track2, create_new_adsr_track2, 1, 512
)

bassline = [
    ("C",3,0.25 * 6),
    ("C",3,0.25 * 2),
    ("F",3,0.25 * 4),
    ("G",2,0.25 * 4),
]


bassline2 = [
    ("C",3,0.25 * 2),
    ("C",3,0.25 * 2),
    ("C",3,0.25 * 2),
    ("C",3,0.25 * 2),

    ("F",3,0.25 * 2),
    ("F",3,0.25 * 2),

    ("G",2,0.25 * 2),
    ("G",2,0.25 * 2),
]

acc=0
for (text, octave, dur) in bassline:
    if text is None: # Rest, octave will also be None
        acc += dur
        continue
    note_name = NoteName(text, octave)
    track1.add_note(
        acc,
        dur,
        CustomState(
            pitch=note_name.get_pitch(), note_id=note_name.get_note_id(), volume= 10**(-6/20) if octave >=3 else 10**(-0/20)
        ),
    )
    acc+=dur

for (text, octave, dur) in bassline2:
    if text is None: # Rest, octave will also be None
        acc += dur
        continue
    note_name = NoteName(text, octave)
    track1.add_note(
        acc,
        dur,
        CustomState(
            pitch=note_name.get_pitch(), note_id=note_name.get_note_id(), volume= 10**(-6/20) if octave >=3 else 10**(-0/20)
        ),
    )
    acc+=dur

melody = [
    ("C", 4, 0.25),
    ("E", 4, 0.25),
    ("G", 4, 0.25),
    ("B", 4, 0.25),
    ("A", 4, 0.25),
    ("G", 4, 0.25),
    ("E", 4, 0.25),
    ("C", 4, 0.25),
    ("D", 4, 0.25),   
    ("F", 4, 0.25),
    ("A", 4, 0.25),
    ("C", 5, 0.25),
    ("B", 4, 0.25*4),
]


acc=0
for (text, octave, dur) in melody:
    if text is None: # Rest, octave will also be None
        acc += dur
        continue
    note_name = NoteName(text, octave)
    track2.add_note(
        acc,
        dur,
        CustomState(
            pitch=note_name.get_pitch(), note_id=note_name.get_note_id(), volume=1
        ),
    )
    acc+=dur

for (text, octave, dur) in melody:
    if text is None: # Rest, octave will also be None
        acc += dur
        continue
    note_name = NoteName(text, octave)
    track2.add_note(
        acc,
        dur,
        CustomState(
            pitch=note_name.get_pitch(), note_id=note_name.get_note_id(), volume=1
        ),
    )
    acc+=dur

frames1 = track1.render_collect()
frames2 = track2.render_collect()

frames = mix(((10**(-3/20), frames1), (10**(-45/20), frames2)))

se = StereoAudio(tuple(frames), SAMPLE_RATE)
se.play(blocking=True)
