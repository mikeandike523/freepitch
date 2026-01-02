from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

from src.audio.adsr_types import ADSR  # Protocol / abstract interface
from src.audio.core import AudioBuffer, CallbackSynth, ProcessCallback, ResetCallback

S = TypeVar("S")


@dataclass(slots=True)
class ADSRAugmentedState(Generic[S]):
    """Internal synth state: user state + ADSR instance (stored by reference)."""
    custom: S
    adsr: ADSR


def create_synth(
    sample_rate: int,
    initial_custom_state: S,
    process_callback: ProcessCallback[S],
    reset_callback: ResetCallback[S] | None,
) -> CallbackSynth[S]:
    """Thin wrapper around CallbackSynth.create for a consistent API surface."""
    return CallbackSynth.create(
        sample_rate,
        initial_custom_state,
        process_callback,
        reset_callback,
    )


# Not used in real world examples
# event_scheduler will take in separate factories
# for creating synth and ADSR instances
def create_synth_with_adsr(
    sample_rate: int,
    initial_custom_state: S,
    adsr: ADSR,
    process_callback: ProcessCallback[S],
    reset_callback: ResetCallback[S] | None,
) -> CallbackSynth[ADSRAugmentedState[S]]:
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
    ) -> AudioBuffer:
        
        frames = process_callback(
            sample_rate=sample_rate,
            n=n,
            num_samples=num_samples,
            state=state.custom,
        )

        adsr_values = state.adsr.generate(num_samples)

        return tuple(map(lambda d: (d[0][0]*d[1], d[0][1]*d[1]), zip(frames, adsr_values)))


    def wrapped_reset(state: ADSRAugmentedState[S]) -> None:
        if reset_callback is not None:
            reset_callback(state=state.custom)
        state.adsr.reset()

    augmented_state = ADSRAugmentedState(custom=initial_custom_state, adsr=adsr)

    return CallbackSynth.create(
        sample_rate,
        augmented_state,
        wrapped_process,
        wrapped_reset,
    )