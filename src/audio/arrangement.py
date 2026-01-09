from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from src.audio.event_scheduler import EventScheduler, RetriggerMode
from src.audio.mixing import mix
from src.audio.note_parsing import NoteParser, ParsedNote


@dataclass(slots=False)
class ClipNote:
    start: float
    note: ParsedNote


@dataclass(slots=False)
class Clip:
    start_time: float = 0.0
    duration: float = 0.0
    notes: list[ClipNote] = field(default_factory=list)
    _cursor: float = 0.0

    def parse_and_add_notes(
        self,
        parser: NoteParser,
        text: str,
        *,
        dur_env: dict[str, float] | None = None,
    ) -> "Clip":
        acc = self._cursor
        for note in parser.parse_lines(text, dur_env=dur_env):
            self.notes.append(ClipNote(acc, note))
            acc += note.duration
        self._cursor = acc
        self._sync_duration(acc)
        return self

    def insert_string(
        self,
        parser: NoteParser,
        text: str,
        *,
        dur_env: dict[str, float] | None = None,
    ) -> "Clip":
        return self.parse_and_add_notes(parser, text, dur_env=dur_env)

    def insert_lines(
        self,
        parser: NoteParser,
        lines: Iterable[str],
        *,
        dur_env: dict[str, float] | None = None,
    ) -> "Clip":
        for line in lines:
            self.parse_and_add_notes(parser, line, dur_env=dur_env)
        return self

    def add_subclip_at(self, clip: "Clip", start_time: float) -> "Clip":
        for clip_note in clip.notes:
            self.notes.append(ClipNote(start_time + clip_note.start, clip_note.note))
        self._sync_duration(start_time + clip.duration)
        return self

    def add_subclip_next(self, clip: "Clip") -> "Clip":
        self.add_subclip_at(clip, self._cursor)
        self._cursor = self.get_end_time()
        return self

    def seek(self, position: float = 0.0) -> "Clip":
        self._cursor = position
        return self

    def _sync_duration(self, candidate_end: float) -> None:
        if candidate_end > self.duration:
            self.duration = candidate_end

    def compute_duration(self) -> float:
        self.duration = 0.0
        for clip_note in self.notes:
            end = clip_note.start + clip_note.note.duration
            if end > self.duration:
                self.duration = end
        return self.duration

    def get_start_time(self) -> float:
        return self.start_time

    def get_end_time(self) -> float:
        return self.start_time + self.duration


class Track:
    def __init__(
        self,
        name: str,
        volume: float,
        clip: Clip,
        *,
        sample_rate: int,
        polyphony: int,
        synth_factory,
        adsr_factory,
        event_bin_width: int,
        block_size: int,
        retrigger_mode: RetriggerMode,
    ) -> None:
        self.name = name
        self.volume = volume
        self.clip = clip
        self._event_scheduler = EventScheduler(
            sample_rate,
            polyphony,
            synth_factory,
            adsr_factory,
            event_bin_width,
            block_size,
            retrigger_mode,
        )

    def add_note(self, start: float, duration: float, state: object) -> None:
        self._event_scheduler.add_note(start, duration, state)

    def schedule_clip(
        self,
        note_name_factory,
        state_factory,
    ) -> None:
        for clip_note in self.clip.notes:
            note = clip_note.note
            if note.name is None:
                continue
            note_name = note_name_factory(note.name, note.octave)
            self.add_note(
                self.clip.start_time + clip_note.start,
                note.duration,
                state_factory(note_name, note),
            )

    def render_collect(self):
        return self._event_scheduler.render_collect()


class Master:
    def __init__(self, tracks: Iterable[Track]) -> None:
        self._tracks = list(tracks)

    def render_collect(self):
        # rendered = ((track.volume, track.render_collect()) for track in self._tracks)
        rendered = []
        for track in self._tracks:
            print(f"Rendering track: {track.name}")
            rendered.append((track.volume, track.render_collect()))
        return mix(rendered)
