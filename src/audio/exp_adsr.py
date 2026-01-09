from __future__ import annotations

from dataclasses import dataclass, field
from math import exp
from typing import Callable, List, Optional, Tuple
from src.audio.adsr_types import ADSRStage




@dataclass(slots=False)
class ExpADSR:
    sample_rate: int
    attack_s: float
    decay_s: float
    sustain_level: float
    release_s: float
    num_tau: float = 5.0

    enter_idle_handlers: List[Callable[[],None]] = field(default_factory=list)

    stage: ADSRStage = ADSRStage.IDLE
    value: float = 0.0

    # per-stage state (for the current exponential segment)
    _i: int = 0           # sample index within segment
    _n: int = 0           # segment length in samples
    _start: float = 0.0   # captured start value at segment entry
    _target: float = 0.0  # segment target
    _tau: float = 1.0     # segment time constant in samples

    def register_enter_idle_handler(self,handler: Callable[[],None]):
        self.enter_idle_handlers.append(handler)

    def clone(self) -> "ExpADSR":
        return ExpADSR(
            sample_rate=self.sample_rate,
            attack_s=self.attack_s,
            decay_s=self.decay_s,
            sustain_level=self.sustain_level,
            release_s=self.release_s,
            num_tau=self.num_tau,
        )

    def reset(self) -> None:
        self.stage = ADSRStage.IDLE
        self.value = 0.0
        self._i = 0
        self._n = 0
        self._start = 0.0
        self._target = 0.0
        self._tau = 1.0

    def note_on(self) -> None:
        self._enter_attack()

    def note_off(self) -> None:
        if self.stage is not ADSRStage.IDLE:
            self._enter_release()

    def get_stage(self) -> ADSRStage:
        return self.stage

    def generate(self, num_samples: int) -> Tuple[float, ...]:
        out = [0.0] * num_samples

        for k in range(num_samples):
            if self.stage is ADSRStage.IDLE:
                # Only forced value: silence in IDLE.
                self.value = 0.0

            elif self.stage is ADSRStage.SUSTAIN:
                # Sustain holds whatever value we reached (typically very close to sustain_level),
                # and does NOT force-jump to sustain_level.
                pass

            else:
                # ATTACK / DECAY / RELEASE are exponential segments.
                self._exp_step()


                # Transition when the segment duration elapses.
                if self._i >= self._n:
                    if self.stage is ADSRStage.ATTACK:
                        self._enter_decay()
                    elif self.stage is ADSRStage.DECAY:
                        self._enter_sustain()
                    elif self.stage is ADSRStage.RELEASE:
                        self.stage = ADSRStage.IDLE  # next sample will become 0.0 (no snap here)
                        for handler in self.enter_idle_handlers:
                            handler()
            out[k] = self.value

        return tuple(out)

    # ---------- internal helpers ----------

    def _secs_to_samples(self, s: float) -> int:
        return int(s * self.sample_rate)

    def _enter_segment(self, stage: ADSRStage, target: float, seconds: float) -> None:
        self.stage = stage

        self._i = 0
        self._n = self._secs_to_samples(seconds)
        self._start = self.value          # capture CURRENT value: guarantees continuity
        self._target = target
        self._tau = (self._n / self.num_tau) if self._n > 0 else 1.0

    def _exp_step(self) -> None:
        if self._n <= 0:
            # Zero-length segment: do not force value; just mark segment as "done".
            self._i = 1
            return

        # y = target + (start - target) * exp(-i / tau)
        self.value = self._target + (self._start - self._target) * exp(-self._i / self._tau)
        self._i += 1

    def _enter_attack(self) -> None:
        self._enter_segment(ADSRStage.ATTACK, target=1.0, seconds=self.attack_s)

    def _enter_decay(self) -> None:
        self._enter_segment(ADSRStage.DECAY, target=self.sustain_level, seconds=self.decay_s)

    def _enter_sustain(self) -> None:
        self.stage = ADSRStage.SUSTAIN
        # IMPORTANT: do not touch self.value here (no jump)

    def _enter_release(self) -> None:
        self._enter_segment(ADSRStage.RELEASE, target=0.0, seconds=self.release_s)
