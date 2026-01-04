from dataclasses import dataclass
from src.audio.stereo_audio import StereoAudio
from src.audio.core import AudioBuffer
from src.audio.exp_adsr import ExpADSR
import math
from src.audio.helpers.create_synth import create_synth
from src.audio.event_scheduler import EventKind, EventScheduler

SAMPLE_RATE = 48_000

frames=[]


@dataclass(slots=False)
class CustomState:
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
        value = math.sin(state.pitch *2 * math.pi*t)
        buffer.append((value, value))  # stereo
    return buffer


def create_new_synth():
    return create_synth(
        SAMPLE_RATE,
        CustomState(pitch=0.0, note_id=-1),
        process_callback,
        reset_callback=None,
    )

def create_new_adsr():
    return ExpADSR(SAMPLE_RATE, 0.015, 0.100, 0.1, 0.200, num_tau=5.0)

event_scheduler = EventScheduler(
    SAMPLE_RATE,
    16,
    create_new_synth,
    create_new_adsr,
    4,
    512
)

event_scheduler.add_event(
    0,
    EventKind.NOTE_ON,
    CustomState(pitch=440.0, note_id=1)
)

event_scheduler.add_event(
    1,
    EventKind.NOTE_OFF,
    CustomState(pitch=440.0, note_id=1)
)

render_gen = event_scheduler.render()

frames = []

while True:
    try:
        block = next(render_gen)
        frames.extend(block)
    except StopIteration:
        break

se = StereoAudio(tuple(frames), SAMPLE_RATE)
se.play(blocking=True)