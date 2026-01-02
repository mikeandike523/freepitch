from dataclasses import dataclass
from src.audio.stereo_audio import StereoAudio
from src.audio.core import AudioBuffer
from src.audio.exp_adsr import ExpADSR
import math
from src.audio.helpers.create_synth import create_synth_with_adsr

SAMPLE_RATE = 48_000

frames=[]

adsr = ExpADSR(SAMPLE_RATE, 0.015, 0.100, 0.1, 0.200, num_tau=5.0)

@dataclass(slots=True)
class CustomState:
    ... # no custom state needed for this example

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
        value = math.sin(440 *2 * math.pi*t)
        buffer.append((value, value))  # stereo
    return buffer

synth = create_synth_with_adsr(
    SAMPLE_RATE,
    CustomState(),
    adsr,
    process_callback,
    reset_callback=None,
)

adsr.note_on()

while len(frames) < SAMPLE_RATE * 1:  # 1 seconds

    block = synth.process(1024)
    frames.extend(block)

    if len(frames) >= SAMPLE_RATE/2:  # after 0.5 second
        adsr.note_off()



se = StereoAudio(tuple(frames), SAMPLE_RATE)
se.play(blocking=True)