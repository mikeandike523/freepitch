from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Callable, Generic, Optional, Protocol, Tuple, TypeVar

StereoFrame = Tuple[float, float]
AudioBuffer = Tuple[StereoFrame, ...]
S = TypeVar("S")
C = TypeVar("C")


class ProcessCallback(Protocol[S, C]):
    def __call__(
        self,
        sample_rate: int,
        n: int,
        num_samples: int,
        state: S,
        config: C,
    ) -> AudioBuffer: ...


class ResetCallback(Protocol[S, C]):
    def __call__(self, state: S, config: C) -> None: ...


class Synth(Protocol[S, C]):
    n: int
    def set_state(self, new_state: S) -> None: ...
    def process(self, num_samples: int) -> AudioBuffer: ...
    def reset(self) -> None: ...
    def clone(self) -> "Synth[S, C]": ...


@dataclass(slots=False)
class CallbackSynth(Generic[S, C]):
    _sample_rate: int
    state: S = field()   # user state kept separate
    config: C
    _process_cb: ProcessCallback[S, C]
    _clone_state_cb: Callable[[S], S] = field(repr=False)
    _clone_config_cb: Callable[[C], C] = field(repr=False)
    _initial_state: S = field(repr=False)
    _reset_cb: Optional[ResetCallback[S, C]] = None
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
            config=self.config,
        )
        self.n += num_samples
        return out

    def reset(self) -> None:
        self.n = 0
        if self._reset_cb is not None:
            self._reset_cb(state=self.state, config=self.config)

    def clone(self) -> "CallbackSynth[S, C]":
        return CallbackSynth.create(
            sample_rate=self._sample_rate,
            initial_state=self._initial_state,
            config=self.config,
            process_callback=self._process_cb,
            reset_callback=self._reset_cb,
            clone_state=self._clone_state_cb,
            clone_config=self._clone_config_cb,
        )

    @staticmethod
    def _default_clone_state(state: S) -> S:
        clone_method = getattr(state, "clone", None)
        if callable(clone_method):
            return clone_method()
        return copy.deepcopy(state)

    @staticmethod
    def _default_clone_config(config: C) -> C:
        clone_method = getattr(config, "clone", None)
        if callable(clone_method):
            return clone_method()
        return config

    @classmethod
    def create(
        cls,
        sample_rate: int,
        initial_state: S,
        config: C,
        process_callback: ProcessCallback[S, C],
        reset_callback: Optional[ResetCallback[S, C]] = None,
        clone_state: Optional[Callable[[S], S]] = None,
        clone_config: Optional[Callable[[C], C]] = None,
    ) -> "CallbackSynth[S, C]":
        clone_state_cb = clone_state or cls._default_clone_state
        clone_config_cb = clone_config or cls._default_clone_config
        initial_state_copy = clone_state_cb(initial_state)
        return cls(
            _sample_rate=sample_rate,
            state=clone_state_cb(initial_state_copy),
            config=clone_config_cb(config),
            _process_cb=process_callback,
            _reset_cb=reset_callback,
            _clone_state_cb=clone_state_cb,
            _clone_config_cb=clone_config_cb,
            _initial_state=initial_state_copy,
        )
