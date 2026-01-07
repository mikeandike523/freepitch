import os
from dataclasses import dataclass

import numpy as np
import soundfile as sf


SR = 44100
OUTPUT_DIR = os.path.join(
    "output_files",
    "generate_samples",
    "drumkits",
    "preset_1",
)


@dataclass(frozen=True)
class DrumSample:
    name: str
    audio: np.ndarray


def exp_env(length, decay):
    t = np.linspace(0.0, 1.0, length, endpoint=False)
    return np.exp(-t * decay)


def sine_osc(freq, length):
    t = np.arange(length) / SR
    return np.sin(2 * np.pi * freq * t)


def sine_sweep(f_start, f_end, length):
    t = np.linspace(0.0, 1.0, length, endpoint=False)
    freqs = f_start * (f_end / f_start) ** t
    phase = 2 * np.pi * np.cumsum(freqs) / SR
    return np.sin(phase)


def soft_clip(signal, amount=1.5):
    return np.tanh(signal * amount)


def normalize(signal, peak=0.95):
    max_val = np.max(np.abs(signal))
    if max_val == 0:
        return signal
    return signal * (peak / max_val)


def fft_bandpass(signal, low, high):
    spectrum = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(signal.size, 1.0 / SR)
    mask = (freqs >= low) & (freqs <= high)
    spectrum *= mask
    return np.fft.irfft(spectrum, n=signal.size)


def fft_highpass(signal, cutoff):
    spectrum = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(signal.size, 1.0 / SR)
    spectrum[freqs < cutoff] = 0
    return np.fft.irfft(spectrum, n=signal.size)


def make_kick():
    length = int(0.6 * SR)
    body = sine_sweep(160, 45, length)
    body *= exp_env(length, 7.0)
    sub = sine_osc(45, length) * exp_env(length, 5.0)
    click = np.random.randn(int(0.015 * SR)) * exp_env(int(0.015 * SR), 30.0)
    click = fft_highpass(click, 3000)
    click = np.pad(click, (0, length - click.size))
    kick = body * 0.9 + sub * 0.4 + click * 0.3
    kick = soft_clip(kick, 2.2)
    return normalize(kick)


def make_snare():
    length = int(0.4 * SR)
    noise = np.random.randn(length)
    noise = fft_bandpass(noise, 800, 8000)
    noise *= exp_env(length, 10.0)
    tone = sine_osc(190, length) * exp_env(length, 12.0)
    snap = np.random.randn(int(0.03 * SR)) * exp_env(int(0.03 * SR), 25.0)
    snap = fft_bandpass(snap, 2000, 12000)
    snap = np.pad(snap, (0, length - snap.size))
    snare = noise * 0.8 + tone * 0.5 + snap * 0.4
    snare = soft_clip(snare, 1.8)
    return normalize(snare)


def make_hat_closed():
    length = int(0.15 * SR)
    noise = np.random.randn(length)
    noise = fft_highpass(noise, 6000)
    noise *= exp_env(length, 25.0)
    metallic = sine_osc(9500, length) * exp_env(length, 30.0)
    hat = noise * 0.9 + metallic * 0.2
    hat = soft_clip(hat, 2.0)
    return normalize(hat)


def make_hat_open():
    length = int(0.6 * SR)
    noise = np.random.randn(length)
    noise = fft_highpass(noise, 5000)
    noise *= exp_env(length, 8.0)
    metallic = sine_osc(8500, length) * exp_env(length, 10.0)
    hat = noise * 0.85 + metallic * 0.25
    hat = soft_clip(hat, 1.8)
    return normalize(hat)


def make_clap():
    length = int(0.5 * SR)
    noise = np.random.randn(length)
    noise = fft_bandpass(noise, 900, 12000)
    env = exp_env(length, 6.0)
    pulses = np.zeros(length)
    for delay_ms in (0, 20, 40, 60):
        start = int((delay_ms / 1000) * SR)
        pulse_len = int(0.04 * SR)
        pulse = np.random.randn(pulse_len) * exp_env(pulse_len, 18.0)
        pulse = fft_bandpass(pulse, 1200, 12000)
        end = min(start + pulse_len, length)
        pulses[start:end] += pulse[: end - start]
    clap = noise * env * 0.3 + pulses * 0.9
    clap = soft_clip(clap, 2.0)
    return normalize(clap)


def make_tom(frequency, decay):
    length = int(0.5 * SR)
    tone = sine_sweep(frequency * 1.2, frequency, length)
    tone *= exp_env(length, decay)
    thump = sine_osc(frequency / 2, length) * exp_env(length, decay * 0.9)
    attack = np.random.randn(int(0.02 * SR)) * exp_env(int(0.02 * SR), 20.0)
    attack = fft_bandpass(attack, 300, 3000)
    attack = np.pad(attack, (0, length - attack.size))
    tom = tone * 0.8 + thump * 0.4 + attack * 0.25
    tom = soft_clip(tom, 1.8)
    return normalize(tom)


def make_rimshot():
    length = int(0.12 * SR)
    noise = np.random.randn(length)
    noise = fft_highpass(noise, 2000)
    noise *= exp_env(length, 18.0)
    click = sine_osc(2100, length) * exp_env(length, 20.0)
    rim = noise * 0.7 + click * 0.4
    rim = soft_clip(rim, 2.2)
    return normalize(rim)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    samples = [
        DrumSample("kick.wav", make_kick()),
        DrumSample("snare.wav", make_snare()),
        DrumSample("hat_closed.wav", make_hat_closed()),
        DrumSample("hat_open.wav", make_hat_open()),
        DrumSample("clap.wav", make_clap()),
        DrumSample("tom_low.wav", make_tom(110, 6.5)),
        DrumSample("tom_mid.wav", make_tom(160, 7.5)),
        DrumSample("tom_high.wav", make_tom(220, 8.0)),
        DrumSample("rimshot.wav", make_rimshot()),
    ]

    for sample in samples:
        path = os.path.join(OUTPUT_DIR, sample.name)
        sf.write(path, sample.audio.astype(np.float32), SR)


if __name__ == "__main__":
    main()
