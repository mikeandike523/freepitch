from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Generic, Optional, Protocol, Tuple, TypeVar

StereoFrame = Tuple[float, float]
AudioBuffer = Tuple[StereoFrame, ...]
S = TypeVar("S")


class ProcessCallback(Protocol[S]):
    def __call__(
        self,
        sample_rate: int,
        n: int,
        num_samples: int,
        state: S,
    ) -> AudioBuffer: ...


class ResetCallback(Protocol[S]):
    def __call__(self, state: S) -> None: ...


class Synth(Protocol[S]):
    n: int
    def set_state(self, new_state: S) -> None: ...
    def process(self, num_samples: int) -> AudioBuffer: ...
    def reset(self) -> None: ...


@dataclass(slots=True)
class CallbackSynth(Generic[S]):
    _sample_rate: int
    state: S = field()   # user state kept separate
    _process_cb: ProcessCallback[S]
    _reset_cb: Optional[ResetCallback[S]] = None
    n: int = 0           # internal counter since first process() or reset()

    @property
    def sample_rate(self) -> int:
        return self._sample_rate
    
    def set_state(self, new_state: S):
        self.state = new_state

    def process(self, num_samples: int) -> AudioBuffer:
        out = self._process_cb(
            sample_rate=self._sample_rate,
            n=self.n,
            num_samples=num_samples,
            state=self.state,
        )
        self.n += num_samples
        return out

    def reset(self) -> None:
        self.n = 0
        if self._reset_cb is not None:
            self._reset_cb(state=self.state)

    @classmethod
    def create(
        cls,
        sample_rate: int,
        initial_state: S,
        process_callback: ProcessCallback[S],
        reset_callback: Optional[ResetCallback[S]] = None
    ) -> "CallbackSynth[S]":
        return cls(
            _sample_rate=sample_rate,
            state=initial_state,
            _process_cb=process_callback,
            _reset_cb=reset_callback,
        )
