from collections.abc import Callable, Iterable

from src.audio.note_parsing import NoteName, ParsedNote


def schedule_parsed_notes(
    track,
    start: float,
    notes: Iterable[ParsedNote],
    note_name_factory: Callable[[str, int], NoteName],
    state_factory: Callable[[NoteName, ParsedNote], object],
) -> float:
    acc = start
    for note in notes:
        if note.name is None:
            acc += note.duration
            continue
        note_name = note_name_factory(note.name, note.octave)
        track.add_note(acc, note.duration, state_factory(note_name, note))
        acc += note.duration
    return acc - start
