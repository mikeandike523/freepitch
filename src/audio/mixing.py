from typing import List, Tuple
import numpy as np
from src.audio.core import AudioBuffer


def mix(setup: List[Tuple[float, AudioBuffer]]) -> AudioBuffer:
    """Mix multiple stereo `AudioBuffer`s with given volumes using NumPy.

    Returns an `AudioBuffer` (tuple of stereo frames).
    """
    if not setup:
        return tuple()

    length = max(len(frames) for _, frames in setup)
    result = np.zeros((length, 2), dtype=float)

    for volume, frames in setup:
        if not frames:
            continue
        arr = np.asarray(frames, dtype=float)
        n = arr.shape[0]
        result[:n] += volume * arr

    # Convert to AudioBuffer: tuple of (left, right) float pairs
    return tuple((float(l), float(r)) for l, r in result.tolist())
