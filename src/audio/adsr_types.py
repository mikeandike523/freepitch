from __future__ import annotations
from typing import Callable, Protocol, Tuple


Envelope = Tuple[float, ...]
from enum import Enum, auto


class ADSRStage(Enum):
    IDLE = auto()
    ATTACK = auto()
    DECAY = auto()
    SUSTAIN = auto()
    RELEASE = auto()


class ADSR(Protocol):
    def generate(self, num_samples: int) -> Envelope: ...
    def reset(self) -> None: ...
    def note_on(self) -> None: ...
    def note_off(self) -> None: ...
    def register_enter_idle_handler(self, handler:Callable[[],None]) -> None: ...
    def get_stage(self) -> ADSRStage: ...
