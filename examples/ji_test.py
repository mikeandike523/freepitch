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


# ./
SCALE = [
1/1,
10/9,
9/8,
5/4,
4/3,
25/18,
36/25,
3/2,
5/3,
16/9,
9/5,
48/25
]

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

# Compare JI to 12-tone
for i in range(len(SCALE)):
    track1.add_note(
        0+i * 0.5,
        0.5,
        CustomState(
            pitch=REFERENCE_C_FREQ * SCALE[i],
            volume=0.3,
            note_id=i
        )
    )

frames = track1.render_collect()

se = StereoAudio(tuple(frames), SAMPLE_RATE)
se.play(blocking=True)