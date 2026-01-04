from dataclasses import dataclass
from src.audio.stereo_audio import StereoAudio
from src.audio.core import AudioBuffer
from src.audio.exp_adsr import ExpADSR
import math
from src.audio.helpers.create_synth import create_synth
from src.audio.event_scheduler import EventScheduler

SAMPLE_RATE = 48_000

REFERENCE_A_FREQ = 440

# C5 is 3 semitones above A4
# The -1 makes it go to the octave below
REFERENCE_C_FREQ = REFERENCE_A_FREQ * 2 ** (3 / 12 - 1)

# GPT helped me choose some decent starting values
LIMIT_5_JI_12 = [
    1/1,    # C
    16/15,  # C#
    9/8,    # D
    6/5,    # D#
    5/4,    # E
    4/3,    # F
    45/32,  # F# (alt: 64/45)
    3/2,    # G
    8/5,    # G#
    5/3,    # A
    9/5,    # A#
    15/8,   # B
]

EDO_12 = []

for n in range(12):
    ratio = 2 ** (n / 12)
    EDO_12.append(ratio)

@dataclass(slots=False)
class CustomState:
    volume: float
    pitch: float
    note_id: int

# Just the waveform part
# ADSR is handeled automatically
# The create_synth_with_adsr function takes care of the multiplying internally
def process_callback(
        sample_rate: int,
        n: int,
        num_samples: int,
        state: CustomState # unused
) -> AudioBuffer:
    buffer = []
    for i in range(num_samples):
        t = (n + i) / sample_rate
        value = state.volume*math.sin(state.pitch *2 * math.pi*t)
        buffer.append((value, value))  # stereo
    return buffer


def create_new_synth():
    return create_synth(
        SAMPLE_RATE,
        CustomState(pitch=0.0, volume=1.0, note_id=-1),
        process_callback,
        reset_callback=None,
    )

def create_new_adsr():
    return ExpADSR(SAMPLE_RATE, 0.015, 0.100, 0.1, 0.200, num_tau=5.0)

track1 = EventScheduler(
    SAMPLE_RATE,
    16,
    create_new_synth,
    create_new_adsr,
    4,
    512
)

# Compose the song here

melody = [

    # C E G E C

    0, 4, 7, 4, 0,

]

# Compare JI to 12-tone
for i,note_index in enumerate(melody):
    track1.add_note(
        0+i * 0.5,
        0.5,
        CustomState(
            pitch=REFERENCE_C_FREQ * LIMIT_5_JI_12[note_index],
            volume=0.3,
            note_id=note_index
        )
    )

for i, note_index in enumerate(melody):
    track1.add_note(
        3 + i * 0.5,
        0.5,
        CustomState(
            pitch=REFERENCE_C_FREQ * EDO_12[note_index],
            volume=0.3,
            note_id=note_index
        )
    )


render_gen = track1.render()

frames = []

while True:
    try:
        block = next(render_gen)
        frames.extend(block)
    except StopIteration:
        break

se = StereoAudio(tuple(frames), SAMPLE_RATE)
se.play(blocking=True)