from collections import defaultdict
import copy
from dataclasses import dataclass, field
from enum import Enum, auto
import json
import math
from typing import Dict, List, Optional, Tuple, TypeVar
import warnings

from tqdm import tqdm
from src.audio.adsr_types import ADSR, ADSRStage
from src.audio.core import CallbackSynth, AudioBuffer


@dataclass(slots=False)
class Voice:
    _running: bool
    _synth: CallbackSynth
    _adsr: Optional[ADSR] = None
    _last_note_on_sample_index: int = 0
    _last_note_off_sample_index: int = 0
    _current_note_id: Optional[int] = None

    def __init__(self, synth: CallbackSynth, adsr: Optional[ADSR] = None):
        self._synth = synth
        self._adsr = adsr
        self._running = False

        def make_self_not_running():
            self._running = False

        if self._adsr:
            self._adsr.register_enter_idle_handler(make_self_not_running)

    def note_on(
        self,
        note_id,
        data,
        global_sample_index: Optional[int] = None,
        *,
        reset_adsr: bool = True,
        reset_synth: bool = True,
        trigger_adsr: bool = True,
    ):
        if self._adsr:
            if reset_adsr:
                self._adsr.reset()
            if trigger_adsr:
                self._adsr.note_on()
        self._synth.set_state(data)
        if reset_synth:
            self._synth.reset()
        self._current_note_id = note_id

        self._running = True

        if global_sample_index is not None:
            self._last_note_on_sample_index = global_sample_index

    def note_off(self, global_sample_index: Optional[int] = None):
        if self._adsr:
            self._adsr.note_off()
        else:
            self._running = False
        if global_sample_index is not None:
            self._last_note_off_sample_index = global_sample_index

    def process(self, num_samples: int):
        frames = self._synth.process(num_samples)
        envelope = tuple(1.0 for _ in range(len(frames)))
        if self._adsr:
            envelope = self._adsr.generate(num_samples)
        multiplied = tuple((v * l, v * r) for (v, (l, r)) in zip(envelope, frames))
        return multiplied

    def is_running(self) -> bool:
        return self._running

    def get_current_note_id(self) -> Optional[int]:
        return self._current_note_id

    def get_last_note_on_sample_index(self) -> int:
        return self._last_note_on_sample_index

    def get_last_note_off_sample_index(self) -> int:
        return self._last_note_off_sample_index

    def get_adsr_stage(self) -> Optional[ADSRStage]:
        if not self._adsr:
            return None
        return self._adsr.get_stage()


class EventKind(Enum):
    NOTE_ON = auto()
    NOTE_OFF = auto()


class RetriggerMode(Enum):
    ALLOW_TAILS = auto()
    CUT_TAILS = auto()
    ATTACK_FROM_CURRENT_LEVEL = auto()


D = TypeVar("D")


@dataclass(slots=False)
class Event[D]:
    kind: EventKind
    data: D

    @property
    def note_id(self) -> int:
        return self.data.note_id
    


@dataclass(slots=False)
class EventBin:
    events: Dict[int, List[Event]]  # sort events by note id

    def __init__(self, events: Optional[Dict[int, List[Event]]] = None):
        self.events = {}
        if events:
            self.events = copy.deepcopy(events)

    def add_event(self, event: Event):
        note_id = event.note_id

        if note_id not in self.events:
            self.events[note_id] = [event]
        else:
            self.events[note_id].append(event)

    def get_simplified(self) -> "EventBin":
        """
        Generate a new, simplified bin that adheres to our simultaneity disambiguation rules
        """

        simple_events: Dict[int, List[Event]] = defaultdict(list)

        # under the assumption that note_id uniquely identifies a value of "data"
        # This is up to library user to uphold

        for note_id, events_for_note_id in self.events.items():
            offs_for_note_id = tuple(
                filter(lambda evt: evt.kind == EventKind.NOTE_OFF, events_for_note_id)
            )
            ons_for_note_id = tuple(
                filter(lambda evt: evt.kind == EventKind.NOTE_ON, events_for_note_id)
            )
            if offs_for_note_id:
                simple_events[note_id].append(
                    copy.deepcopy(offs_for_note_id[0])
                )  # Once again, assume that an ID uniquely corresponds to unique data
            if ons_for_note_id:
                simple_events[note_id].append(copy.deepcopy(ons_for_note_id[0]))

        return EventBin(simple_events)
        


@dataclass(slots=False)
class EventScheduler:
    """

    A scheduler for a specific track

    Handles note on and off events
    Can handle up to the specified number of simultaneous voices
    You provide a prototype synth (no ADSR): `synth`
    You provide a prototype ADSR instance, if desired: `adsr`
    The scheduler will clone and manage the synths and ADSRs as needed
    Under the assumption that most tracks have some significant ADSR envelope
    The voice-stealing rule is simple:
        If utilizing ADSR:
            If there are any notes in the release phase:
                Steal the oldest one (earliest note off time)
            if not:
                Steal the oldest note on (earliest note on time)
        else:
            note_off events immediately free the voice
            So a steal will occur only to note_on when all voices are active
            It will simply steal the oldest note on

    Retrigger handling is configurable via retrigger_mode:
        ALLOW_TAILS: do not special-case; allow overlapping voices
        CUT_TAILS: reuse the running voice for the same note id
        ATTACK_FROM_CURRENT_LEVEL: if ADSR is enabled, restart attack from
            the current envelope level without resetting synth state


    Event simultaneity:

    If two events occur on the same sample index (i=floor(time*sampleRate))

    We do the following logic:

    Collect the set of events at that time and group by id:

    For each id "bin":

    Determine if there is at least one note_off
    Determine if there is at least one note_on


    If no note_on, just process one note_off event

    If at least one note_on, process one note_off, followed by one note_on and no more

    Note:

    We bin only a single, precise sample index by default

    It is recommend, when rounding, to round down note_on and round_up note_off events

    note_on:  i= floor(t*sample_rate)
    note_off: i=ceil(t*sample_rate)

    This is essentially a binning width of 1

    But we will optionally allow an argument of a large binning width (e.g. 4 samples)

    so we will still ceil note_offs, and floor note_ons, but quantizing to a grid of more samples (e.g. 4)

    This can also be done at composition time as an extra processing layer, but
    might as well just include the functionality here and avoid needing to do it upwards
    in the pipeline even though this is non-traditional

    More notes about simultaneity:

    Because the scheduler is offline, we have the luxury of not needing to buffer

    We can go sample-by-sample and process events as they occur


    """

    _synth: CallbackSynth
    _adsr: Optional[ADSR]
    _is_using_adsr: bool
    _sample_rate: int
    _max_voices: int
    _tick_size: int
    _buffer_size: int
    _retrigger_mode: RetriggerMode

    voices: Tuple[Voice, ...]

    event_bins: Dict[int, EventBin] = field(
        default_factory=lambda: defaultdict(lambda: EventBin(None))
    )

    def __init__(
        self,
        sample_rate: int,
        max_voices: int,
        synth: CallbackSynth,
        adsr: Optional[ADSR] = None,
        tick_size: Optional[
            int
        ] = 4,  # Default to a small value > 1 to avoid floating point issues
        buffer_size: Optional[int] = 512,
        retrigger_mode: RetriggerMode = RetriggerMode.ALLOW_TAILS,
    ):
        if buffer_size % tick_size != 0:
            raise ValueError(
                """
We enforce condition that buffer_size must be a multiple of tick_size.
It may not be mathematically necessary, but it is recommended enough
to throw error if not true.
"""
            )

        self._synth = synth
        self._adsr = adsr
        self._sample_rate = sample_rate
        self._max_voices = max_voices
        self._buffer_size = buffer_size
        self._tick_size = tick_size
        self._is_using_adsr = adsr is not None
        self._retrigger_mode = retrigger_mode

        self.voices = tuple(
            Voice(
                synth=self._synth.clone(),
                adsr=None if self._adsr is None else self._adsr.clone(),
            )
            for _ in range(self._max_voices)
        )

        self.event_bins = defaultdict(lambda: EventBin(None))

    def add_event[D](self, time: float, kind: EventKind, data: D):
        quantized_sample_index = None
        if kind == EventKind.NOTE_ON:
            quantized_sample_index = (
                math.floor(time * self._sample_rate / self._tick_size) * self._tick_size
            )
        else:
            quantized_sample_index = (
                math.ceil(time * self._sample_rate / self._tick_size) * self._tick_size
            )
        self.event_bins[quantized_sample_index].add_event(Event(kind=kind, data=data))

    def add_note[D](self, time: float, duration: float, data: D):
        self.add_event(time, EventKind.NOTE_ON, data)
        self.add_event(time + duration, EventKind.NOTE_OFF, data)

    def _get_running_voice_count(self):
        return len(tuple(True for v in self.voices if v.is_running()))

    def _get_free_voice_count(self):
        return self._max_voices - self._get_running_voice_count()

    def _get_free_voices(self):
        return tuple(filter(lambda x: not x.is_running(), self.voices))

    def _get_retrigger_voice(self, note_id: Optional[int]) -> Optional[Voice]:
        return next(
            (
                voice
                for voice in self.voices
                if voice.is_running() and voice.get_current_note_id() == note_id
            ),
            None,
        )

    def _get_or_steal_voice(self) -> Voice:
        free_voices = self._get_free_voices()
        if free_voices:
            return free_voices[0]
        if not self._is_using_adsr:
            # Simply use the oldest
            voices_sorted = sorted(
                self.voices, key=lambda voice: voice.get_last_note_on_sample_index()
            )
            return voices_sorted[0]
        else:
            release_voices: Tuple[Voice, ...] = tuple(
                filter(
                    lambda voice: voice.get_adsr_stage() == ADSRStage.RELEASE,
                    self.voices,
                )
            )
            if release_voices:
                release_voices_sorted = sorted(
                    release_voices,
                    key=lambda voice: voice.get_last_note_off_sample_index(),
                )
                return release_voices_sorted[0]
            # Simply use the oldest
            voices_sorted = sorted(
                self.voices, key=lambda voice: voice.get_last_note_on_sample_index()
            )
            return voices_sorted[0]

    def render_collect(
        self,
        silence_db: float = -60.0,
        max_seconds_after_last_note_off: float = 4.0,
    ):
        gen = self.render(
            silence_db=silence_db,
            max_seconds_after_last_note_off=max_seconds_after_last_note_off,
        )
        frames = []
        while True:
            try:
                block = next(gen)
                frames.extend(block)
            except StopIteration:
                break
        return tuple(frames)

    # Naive approach
    # go tick by tick
    # In the future, will want to write algorithms
    # to mark regions of silence and skip
    def render(
        self,
        silence_db: float = -60.0,
        max_seconds_after_last_note_off: float = 4.0,
    ):
        silence_amplitude = 10 ** (silence_db / 20)

        # Step 1. Perform disambiguation rules

        remaining_bins = {i: b.get_simplified() for (i, b) in self.event_bins.items()}

        # Sort by key, ascending

        remaining_bins: Dict[int, EventBin] = dict(
            sorted(remaining_bins.items(), key=lambda item: item[0])
        )


        if not remaining_bins:
            warnings.warn(
                "EventScheduler.render received no events after simplification; "
                "rendering an empty AudioBuffer.",
                RuntimeWarning,
            )
            return
        
        print("Remaining Bins: ", remaining_bins)

        last_note_off_sample_index: Optional[int] = None
        last_event_sample_index = max(remaining_bins.keys())

        for bin_index, event_bin in remaining_bins.items():
            for events_for_note_id in event_bin.events.values():
                for event in events_for_note_id:
                    if event.kind == EventKind.NOTE_OFF:
                        if (
                            last_note_off_sample_index is None
                            or bin_index > last_note_off_sample_index
                        ):
                            last_note_off_sample_index = bin_index

        if last_note_off_sample_index is None:
            warnings.warn(
                "EventScheduler.render detected events but no NOTE_OFF events after "
                "simplification; invalid track. Rendering an empty AudioBuffer.",
                RuntimeWarning,
            )
            return

        max_samples_after_last_note_off = int(
            max_seconds_after_last_note_off * self._sample_rate
        )
        estimated_end_sample = (
            max(last_event_sample_index, last_note_off_sample_index)
            + max_samples_after_last_note_off
        )
        estimated_total_blocks = max(
            1, math.ceil(estimated_end_sample / self._buffer_size)
        )

        progress = tqdm(
            total=estimated_total_blocks,
            desc="Rendering",
            unit="block",
        )

        block_frames = None

        num_processed_blocks = 0

        try:
            while (
                block_frames is None
                or any(voice.is_running() for voice in self.voices)
                or remaining_bins
            ):
                cursor = num_processed_blocks * self._buffer_size
                block_event_bins = []
                while True:
                    if remaining_bins:
                        next_bin_index, next_bin = next(iter(remaining_bins.items()))
                        if next_bin_index < cursor + self._buffer_size:
                            block_event_bins.append((next_bin_index, next_bin))
                            del remaining_bins[next_bin_index]
                        else:
                            break
                    else:
                        break

                block_frames = []
                segment_start = 0

                def render_segment(num_samples: int) -> List[Tuple[float, float]]:
                    if num_samples <= 0:
                        return []
                    mix_left = [0.0] * num_samples
                    mix_right = [0.0] * num_samples
                    for voice in self.voices:
                        if not voice.is_running():
                            continue
                        voice_frames = voice.process(num_samples)
                        for i, (l, r) in enumerate(voice_frames):
                            mix_left[i] += l
                            mix_right[i] += r
                    return list(zip(mix_left, mix_right))

                for bin_index, event_bin in block_event_bins:
                    # sample index, even bin
                    offset_in_block = bin_index - cursor
                    segment_length = offset_in_block - segment_start
                    if segment_length > 0:
                        block_frames.extend(render_segment(segment_length))
                        segment_start = offset_in_block
                    for note_id, events_for_note_id in event_bin.events.items():
                        for event in events_for_note_id:
                            if event.kind == EventKind.NOTE_ON:
                                retrigger_voice = None
                                if (
                                    self._retrigger_mode
                                    is not RetriggerMode.ALLOW_TAILS
                                ):
                                    retrigger_voice = self._get_retrigger_voice(note_id)
                                if (
                                    retrigger_voice
                                    and self._retrigger_mode
                                    is RetriggerMode.ATTACK_FROM_CURRENT_LEVEL
                                    and self._is_using_adsr
                                ):
                                    retrigger_voice.note_on(
                                        note_id,
                                        event.data,
                                        global_sample_index=bin_index,
                                        reset_adsr=False,
                                        reset_synth=False,
                                        trigger_adsr=True,
                                    )
                                else:
                                    voice = retrigger_voice or self._get_or_steal_voice()
                                    voice.note_on(
                                        note_id,
                                        event.data,
                                        global_sample_index=bin_index,
                                    )
                            elif event.kind == EventKind.NOTE_OFF:
                                target_voice = next(
                                    (
                                        voice
                                        for voice in self.voices
                                        if voice.is_running()
                                        and voice.get_current_note_id() == note_id
                                    ),
                                    None,
                                )
                                if target_voice is not None:
                                    target_voice.note_off(global_sample_index=bin_index)

                remaining_samples = self._buffer_size - segment_start
                if remaining_samples > 0:
                    block_frames.extend(render_segment(remaining_samples))

                num_processed_blocks += 1
                progress.update(1)

                cursor_end = cursor + self._buffer_size
                past_last_event = cursor_end >= last_event_sample_index

                should_stop = False
                if past_last_event and cursor_end >= (
                    last_note_off_sample_index + max_samples_after_last_note_off
                ):
                    should_stop = True

                if past_last_event and not remaining_bins:
                    if not any(voice.is_running() for voice in self.voices):
                        should_stop = True
                    if all(
                        abs(l) <= silence_amplitude and abs(r) <= silence_amplitude
                        for l, r in block_frames
                    ):
                        should_stop = True

                yield block_frames
                if should_stop:
                    break
        finally:
            if progress.n < progress.total:
                progress.update(progress.total - progress.n)
            progress.close()
