from dataclasses import dataclass
import math

from src.audio.core import AudioBuffer, CallbackSynth
from src.audio.exp_adsr import ExpADSR
from src.audio.helpers.create_synth import create_synth


@dataclass(slots=False)
class CustomState:
    volume: float
    pitch: float
    note_id: int


def _triangle_wave(phase: float) -> float:
    if phase < math.pi / 2:
        return phase / (math.pi / 2)
    if phase < math.pi:
        p = (phase - math.pi / 2) / (math.pi / 2)
        return 1 - p
    if phase < 3 * math.pi / 2:
        p = (phase - math.pi) / (math.pi / 2)
        return -p
    p = (phase - 3 * math.pi / 2) / (math.pi / 2)
    return -1 + p


def _saw_wave(phase: float) -> float:
    value_unscaled = phase / math.pi
    if phase > math.pi:
        value_unscaled = -1 + (phase - math.pi) / math.pi
    return value_unscaled


def _square_wave(phase: float) -> float:
    return 1.0 if phase < math.pi else -1.0


def _sine_wave(phase: float) -> float:
    return math.sin(phase)


_WAVE_SHAPES = {
    "triangle": _triangle_wave,
    "saw": _saw_wave,
    "square": _square_wave,
    "sine": _sine_wave,
}


def build_synth_factories(
    sample_rate: int,
    wave_shape: str,
    adsr: tuple[float, float, float, float],
    *,
    num_tau: float = 5.0,
) -> tuple[CallbackSynth[CustomState, None], ExpADSR]:
    wave_fn = _WAVE_SHAPES.get(wave_shape)
    if wave_fn is None:
        raise ValueError(f"Unsupported wave shape: {wave_shape}")

    attack, decay, sustain, release = adsr

    def process_callback(
        sample_rate: int,
        n: int,
        num_samples: int,
        state: CustomState,
        config: None,
    ) -> AudioBuffer:
        buffer = []
        for i in range(num_samples):
            t = (n + i) / sample_rate
            phase = (state.pitch * 2 * math.pi * t) % (2 * math.pi)
            value = wave_fn(phase) * state.volume
            buffer.append((value, value))
        return buffer

    synth = create_synth(
        sample_rate,
        CustomState(pitch=0.0, volume=1.0, note_id=-1),
        None,
        process_callback,
        reset_callback=None,
    )

    adsr_envelope = ExpADSR(sample_rate, attack, decay, sustain, release, num_tau=num_tau)

    return synth, adsr_envelope
