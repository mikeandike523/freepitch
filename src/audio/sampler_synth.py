from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np
import soundfile as sf

from src.audio.core import AudioBuffer
from src.audio.helpers.create_synth import create_synth


@dataclass(slots=True)
class SamplerState:
    buffer: AudioBuffer
    volume: float
    note_id: int


def _to_stereo(data: np.ndarray) -> np.ndarray:
    if data.ndim == 1:
        data = data[:, np.newaxis]
    if data.shape[1] == 1:
        return np.repeat(data, 2, axis=1)
    if data.shape[1] >= 2:
        return data[:, :2]
    raise ValueError("Audio data must have at least one channel.")


def _resample_audio(
    data: np.ndarray,
    source_rate: int,
    target_rate: int,
) -> np.ndarray:
    if source_rate == target_rate:
        return data
    if data.size == 0:
        return data
    ratio = target_rate / source_rate
    target_length = int(round(data.shape[0] * ratio))
    if target_length <= 1:
        return data[:target_length]
    old_indices = np.linspace(0, data.shape[0] - 1, num=data.shape[0])
    new_indices = np.linspace(0, data.shape[0] - 1, num=target_length)
    channels = []
    for ch in range(data.shape[1]):
        channels.append(np.interp(new_indices, old_indices, data[:, ch]))
    return np.stack(channels, axis=1)


def load_audio_buffer(path: str, target_sample_rate: int) -> AudioBuffer:
    print(f"[sampler] Loading sample: {path}")
    data, source_rate = sf.read(path, always_2d=True, dtype="float32")
    data = _to_stereo(data)
    if source_rate != target_sample_rate:
        print(
            "[sampler] Resampling sample:"
            f" {source_rate} Hz -> {target_sample_rate} Hz"
        )
    data = _resample_audio(data, source_rate, target_sample_rate)
    print(f"[sampler] Loaded {path} ({len(data)} frames).")
    return tuple((float(l), float(r)) for l, r in data.tolist())


def build_sampler_synth_factory(
    sample_rate: int,
    sample_paths: Mapping[str, str],
) -> tuple[Callable[[], object], dict[str, AudioBuffer]]:
    print(f"[sampler] Building sampler buffers ({len(sample_paths)} samples).")
    sample_buffers = {
        name: load_audio_buffer(path, sample_rate)
        for name, path in sample_paths.items()
    }
    print("[sampler] Sampler buffers ready.")

    def process_callback(
        sample_rate: int, n: int, num_samples: int, state: SamplerState
    ) -> AudioBuffer:
        frames = []
        for i in range(num_samples):
            idx = n + i
            if idx < len(state.buffer):
                left, right = state.buffer[idx]
                frames.append((left * state.volume, right * state.volume))
            else:
                frames.append((0.0, 0.0))
        return tuple(frames)

    def create_new_synth():
        return create_synth(
            sample_rate,
            SamplerState(buffer=tuple(), volume=1.0, note_id=-1),
            process_callback,
            reset_callback=None,
        )

    return create_new_synth, sample_buffers
