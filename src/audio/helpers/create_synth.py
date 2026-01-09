from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

from src.audio.adsr_types import ADSR  # Protocol / abstract interface
from src.audio.core import AudioBuffer, CallbackSynth, ProcessCallback, ResetCallback

S = TypeVar("S")
C = TypeVar("C")


@dataclass(slots=False)
class ADSRAugmentedState(Generic[S]):
    """Internal synth state: user state + ADSR instance (stored by reference)."""
    custom: S
    adsr: ADSR


def create_synth(
    sample_rate: int,
    initial_custom_state: S,
    config: C,
    process_callback: ProcessCallback[S, C],
    reset_callback: ResetCallback[S, C] | None,
    *,
    clone_state: Callable[[S], S] | None = None,
    clone_config: Callable[[C], C] | None = None,
) -> CallbackSynth[S, C]:
    """Thin wrapper around CallbackSynth.create for a consistent API surface."""
    return CallbackSynth.create(
        sample_rate,
        initial_custom_state,
        config,
        process_callback,
        reset_callback,
        clone_state=clone_state,
        clone_config=clone_config,
    )


# Not used in real world examples
# event_scheduler will take in separate factories
# for creating synth and ADSR instances
def create_synth_with_adsr(
    sample_rate: int,
    initial_custom_state: S,
    config: C,
    adsr: ADSR,
    process_callback: ProcessCallback[S, C],
    reset_callback: ResetCallback[S, C] | None,
    *,
    clone_state: Callable[[S], S] | None = None,
    clone_config: Callable[[C], C] | None = None,
) -> CallbackSynth[ADSRAugmentedState[S], C]:
    """
    Create a synth whose internal state is ADSRAugmentedState[S], while keeping
    user callbacks typed against S.

    Note: This function does not force envelope note_on/note_off behavior; it
    only stores the ADSR instance and resets it on synth reset. Gate control is
    up to the user (e.g., calling adsr.note_on/off externally or via custom state).
    """

    def wrapped_process(
        sample_rate: int,
        n: int,
        num_samples: int,
        state: ADSRAugmentedState[S],
        config: C,
    ) -> AudioBuffer:
        
        frames = process_callback(
            sample_rate=sample_rate,
            n=n,
            num_samples=num_samples,
            state=state.custom,
            config=config,
        )

        adsr_values = state.adsr.generate(num_samples)

        return tuple(map(lambda d: (d[0][0]*d[1], d[0][1]*d[1]), zip(frames, adsr_values)))


    def wrapped_reset(state: ADSRAugmentedState[S], config: C) -> None:
        if reset_callback is not None:
            reset_callback(state=state.custom, config=config)
        state.adsr.reset()

    augmented_state = ADSRAugmentedState(custom=initial_custom_state, adsr=adsr)

    return CallbackSynth.create(
        sample_rate,
        augmented_state,
        config,
        wrapped_process,
        wrapped_reset,
        clone_state=clone_state,
        clone_config=clone_config,
    )
