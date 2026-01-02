import numpy as np
import soundfile as sf
import sounddevice as sd


class StereoAudio:
    """
    frames: tuple of (L, R) samples
            samples should be floats in [-1, 1] for best fidelity
    """

    def __init__(self, frames, sample_rate=48000):
        self.sample_rate = sample_rate
        self.data = np.asarray(frames, dtype=np.float32)
        if self.data.ndim != 2 or self.data.shape[1] != 2:
            raise ValueError("frames must be a tuple of (L, R) samples")
        self.data = np.clip(self.data, -1.0, 1.0)

    def export(self, path, bit_depth=24):
        """
        Export to a Windows-native WAV file.
        bit_depth: 16 or 24 (24 = higher fidelity, still widely supported)
        """
        subtype = {16: "PCM_16", 24: "PCM_24"}[bit_depth]
        sf.write(path, self.data, self.sample_rate,
                 format="WAV", subtype=subtype)

    def play(self, blocking=False):
        """
        Play audio immediately without writing a file.
        """
        sd.play(self.data, self.sample_rate, blocking=blocking)

    def stop(self):
        """
        Stop playback.
        """
        sd.stop()
