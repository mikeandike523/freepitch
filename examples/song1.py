from dataclasses import dataclass
from src.audio.mixing import mix
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


@dataclass(slots=False)
class CustomState:
    volume: float
    pitch: float
    note_id: int


SCALE = sorted((1, 6 / 5, 5 / 4, 9 / 8, 3 / 4, 4 / 5, 5/3))

print(SCALE)


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
    return ExpADSR(SAMPLE_RATE, 0.015, 0.100, 0.1, 0.200, num_tau=5.0)


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
    return ExpADSR(SAMPLE_RATE, 0.015, 0.100, 0.1, 0.200, num_tau=5.0)


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

for index, ratio in enumerate(SCALE):
    pitch = REFERENCE_C_FREQ * ratio
    track1.add_note(
        index * 0.25, 0.25, CustomState(pitch=pitch, note_id=index, volume=1)
    )


frames1 = track1.render_collect()
frames2 = track2.render_collect()



frames = mix(((0.5, frames1), (0.5, frames2)))

se = StereoAudio(tuple(frames1), SAMPLE_RATE)
se.play(blocking=True)
