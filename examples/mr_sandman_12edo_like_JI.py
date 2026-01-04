from dataclasses import dataclass

import numpy as np
from src.audio.mixing import mix
from src.audio.stereo_audio import StereoAudio
from src.audio.core import AudioBuffer
from src.audio.exp_adsr import ExpADSR
import math
from src.audio.helpers.create_synth import create_synth
from src.audio.event_scheduler import EventScheduler, RetriggerMode
import re

SAMPLE_RATE = 48_000

REFERENCE_A_FREQ = 440

# C5 is 3 semitones above A4
# The -1 makes it go to the octave below
REFERENCE_C_FREQ = REFERENCE_A_FREQ * 9/16


@dataclass(slots=False)
class CustomState:
    volume: float
    pitch: float
    note_id: int


#  ./__inenv python experiments/pick_ji.py --primes=2,3,5 --max-int=64 --edo=12 --per-step=1 --format ratios
SCALE = [
1/1,
16/15,
9/8,
32/27,
5/4,
4/3,
45/32,
3/2,
8/5,
27/16,
16/9,
15/8,
]

NAMES_TO_INDEX= {
    "C":0,
    "C#": 1, "Db": 1,
    "D": 2,
    "D#": 3, "Eb": 3,
    "E": 4,
    "F": 5,
    "F#": 6, "Gb": 6,
    "G": 7,
    "G#": 8, "Ab": 8,
    "A": 9,
    "A#": 10, "Bb": 10,
    "B": 11
}

@dataclass(slots=True)
class NoteName:
    text: str
    octave: int

    def __init__(self, text: str, octave: int):
        if text not in NAMES_TO_INDEX:
            raise ValueError(f"Invalid note name: {text}")
        self.text = text
        self.octave = octave

    def get_index_in_octave(self) -> int:
        return NAMES_TO_INDEX[self.text]

    def get_note_id(self):
        return self.octave * 12 + self.get_index_in_octave()

    def get_pitch(self):
        return SCALE[self.get_index_in_octave()] * 2**(self.octave-4) * REFERENCE_C_FREQ

    def __repr__(self):
        return f"{self.text}{self.octave}"


# ============================================================
# SYNTHS
# ============================================================

# Just the waveform part
# ADSR is handled automatically
# The create_synth_with_adsr function takes care of the multiplying internally

def process_callback_track1(
    sample_rate: int, n: int, num_samples: int, state: CustomState  # unused
) -> AudioBuffer:
    buffer = []
    for i in range(num_samples):
        t = (n + i) / sample_rate
        # triangle wave
        phase = (state.pitch * 2 * math.pi * t) % (2*math.pi)
        if phase < math.pi/2:
            value = phase / (math.pi/2)
        elif phase < math.pi:
            p = (phase-math.pi/2)/(math.pi/2)
            value = 1 - p
        elif phase < 3 * math.pi/2:
            p = (phase-math.pi)/(math.pi/2)
            value = -p
        else:
            p = (phase-3*math.pi/2)/(math.pi/2)
            value = -1 + p
        value *= state.volume
        buffer.append((value, value))  # stereo
    return buffer


def create_new_synth_track1():
    return create_synth(
        SAMPLE_RATE,
        CustomState(pitch=0.0, volume=1.0, note_id=-1),
        process_callback_track1,
        reset_callback=None,
    )


def create_new_adsr_track1():
    return ExpADSR(SAMPLE_RATE, 0.050, 0.050, 0.5, 0.200, num_tau=5.0)


# Just the waveform part
# ADSR is handled automatically
# The create_synth_with_adsr function takes care of the multiplying internally

def process_callback_track2(
    sample_rate: int, n: int, num_samples: int, state: CustomState  # unused
) -> AudioBuffer:
    buffer = []
    for i in range(num_samples):
        t = (n + i) / sample_rate
        phase = (state.pitch * 2 * math.pi * t) % (2 * math.pi)
        # sawtooth wave
        value_unscaled = phase / math.pi
        if phase > math.pi:
            value_unscaled = -1 + (phase - math.pi) / math.pi
        value = value_unscaled * state.volume
        buffer.append((value, value))  # stereo
    return buffer


def create_new_synth_track2():
    return create_synth(
        SAMPLE_RATE,
        CustomState(pitch=0.0, volume=1.0, note_id=-1),
        process_callback_track2,
        reset_callback=None,
    )


def create_new_adsr_track2():
    return ExpADSR(SAMPLE_RATE, 0.015, 0.100, 0.7, 0.200, num_tau=5.0)


# ============================================================
# TRACKS / SCHEDULERS
# ============================================================

# For now: track1 = bassline, track2 = melody

track1 = EventScheduler(
    SAMPLE_RATE, 16, create_new_synth_track1, create_new_adsr_track1, 1, 512, RetriggerMode.ATTACK_FROM_CURRENT_LEVEL
)

track2 = EventScheduler(
    SAMPLE_RATE, 16, create_new_synth_track2, create_new_adsr_track2, 1, 512, RetriggerMode.ATTACK_FROM_CURRENT_LEVEL
)


import ast
import operator as op

# ============================================================
# TEMPO
# ============================================================

BPM = 120
QUARTER = 60 / BPM
EIGHTH = QUARTER / 2

# Base symbols available inside expressions
_DURATION_ENV_BASE = {
    "q": QUARTER,
    "e": EIGHTH,
    "QUARTER": QUARTER,
    "EIGHTH": EIGHTH,
    "BPM": BPM,
}


class _SafeExprEval(ast.NodeVisitor):
    """Safely evaluate a restricted arithmetic expression with variables."""

    _bin_ops = {
        ast.Add: op.add,
        ast.Sub: op.sub,
        ast.Mult: op.mul,
        ast.Div: op.truediv,
        ast.FloorDiv: op.floordiv,
        ast.Mod: op.mod,
        ast.Pow: op.pow,
    }

    _unary_ops = {
        ast.UAdd: op.pos,
        ast.USub: op.neg,
    }

    def __init__(self, env: dict[str, float]):
        self.env = env

    def visit_Expression(self, node: ast.Expression):
        return self.visit(node.body)

    def visit_BinOp(self, node: ast.BinOp):
        left = self.visit(node.left)
        right = self.visit(node.right)
        fn = self._bin_ops.get(type(node.op))
        if fn is None:
            raise ValueError(f"Operator not allowed: {ast.dump(node.op)}")
        return fn(left, right)

    def visit_UnaryOp(self, node: ast.UnaryOp):
        val = self.visit(node.operand)
        fn = self._unary_ops.get(type(node.op))
        if fn is None:
            raise ValueError(f"Unary operator not allowed: {ast.dump(node.op)}")
        return fn(val)

    def visit_Name(self, node: ast.Name):
        if node.id not in self.env:
            raise ValueError(f"Unknown variable in duration expression: {node.id!r}")
        return float(self.env[node.id])

    def visit_Constant(self, node: ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError("Only numeric constants are allowed")

    # Python <3.8 compatibility (probably not needed, but harmless)
    def visit_Num(self, node: ast.Num):  # pragma: no cover
        return float(node.n)

    # Disallow everything else (calls, attributes, subscripts, lambdas, comprehensions, etc.)
    def generic_visit(self, node):
        raise ValueError(f"Disallowed syntax in duration expression: {type(node).__name__}")


def eval_duration_expr(expr: str, *, env: dict[str, float] | None = None) -> float:
    """Evaluate a safe arithmetic expression for durations."""
    expr = expr.strip()
    if not expr:
        raise ValueError("Empty duration")

    merged_env = dict(_DURATION_ENV_BASE)
    if env:
        # user env overrides base env if same names
        merged_env.update(env)

    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"Bad duration expression: {expr!r}") from e

    value = _SafeExprEval(merged_env).visit(tree)
    if not (value >= 0):
        raise ValueError(f"Duration must be >= 0, got {value}")
    return float(value)


# ============================================================
# SIMPLE PARSER (updated)
# ============================================================

def parse_lines(text: str, *, dur_env: dict[str, float] | None = None):
    """Return list of tuples (note_text|None, octave|None, duration_seconds).

    Duration token can now be an expression and may include parentheses.
    Examples:
      C 3 e*6
      C 3 (e*3)/2
      R bar - e

    If your duration expression contains spaces, everything after octave (or note for rests)
    is treated as duration text.
    """

    out = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith('#'):
            continue

        parts = line.split()

        if len(parts) < 2:
            raise ValueError(f"Bad line: {raw!r}")

        # rest form: "R <dur expr>" or "- <dur expr>"
        if parts[0] in ("R", "-", "rest", "REST"):
            dur_txt = " ".join(parts[1:])
            out.append((None, None, eval_duration_expr(dur_txt, env=dur_env)))
            continue

        # note form: "<note> <octave> <dur expr>"
        if len(parts) < 3:
            raise ValueError(f"Bad line: {raw!r}")

        note_txt = parts[0]
        octave = int(parts[1])
        dur_txt = " ".join(parts[2:])

        out.append((note_txt, octave, eval_duration_expr(dur_txt, env=dur_env)))

    return out


def write_notes(track: EventScheduler, start: float, notes):
    acc = start
    for (text, octave, dur) in notes:
        if text is None:
            acc += dur
            continue
        note_name = NoteName(text, octave)
        track.add_note(
            acc,
            dur,
            CustomState(
                pitch=note_name.get_pitch(),
                note_id=note_name.get_note_id(),
                volume=1,
            ),
        )
        acc += dur
    return acc - start


# ============================================================
# MUSIC (your existing material, reformatted into parser syntax)
# ============================================================

BASSLINE_1 = """
# bassline
C 3 e*6
C 3 e*2
F 3 e*4
G 2 e*4
"""

BASSLINE_2 = """
# bassline2
C 3 e*2
C 3 e*2
C 3 e*2
C 3 e*2
F 3 e*2
F 3 e*2
G 2 e*2
G 2 e*2
"""

BASELINE_3 = """
# baseline3: pattern1 x2
C 3 e
G 3 e
E 3 e
G 3 e
C 3 e
G 3 e
E 3 e
G 3 e


C# 3 e
A 3 e
E 3 e
A 3 e

A 2 e
E 3 e
G 3 e
E 3 e
"""

BASSLINE_4="""


"""

MELODY_1 = """
# melody
C 4 e
E 4 e
G 4 e
B 4 e
A 4 e
G 4 e
E 4 e
C 4 e
D 4 e
F 4 e
A 4 e
C 5 e
B 4 e*4
"""

MELODY_2 = """
# melody2
C 4 e
E 4 e
G 4 e
B 4 e
A 4 e
G 4 e
E 4 e
C 4 e
D 4 e
F 4 e
A 4 e
C 5 e
B 4 e*2
A 4 e
B 4 e
"""

MELODY_3 = """
# melody3
B 4 q
A 4 q*3
B 4 e
B 4 e
A 4 e
B 4 e+e*4  
"""

MELODY_4="""
# melody4
R e
C 5 e
C 5 e
B 4 e
C 5 e
B 4 e
R e
B 4 e

F 4 e*2
E 4 e
F 4 e+q*2
"""


# ============================================================
# SCHEDULE / RENDER
# ============================================================

acc = 0.0
acc += write_notes(track1, acc, parse_lines(BASSLINE_1))
acc += write_notes(track1, acc, parse_lines(BASSLINE_2))
acc += write_notes(track1, acc, parse_lines(BASELINE_3))
acc += write_notes(track1, acc, parse_lines(BASSLINE_4))

acc = 0.0
acc += write_notes(track2, acc, parse_lines(MELODY_1))
acc += write_notes(track2, acc, parse_lines(MELODY_2))
acc += write_notes(track2, acc, parse_lines(MELODY_3))
acc += write_notes(track2, acc, parse_lines(MELODY_4))

frames1 = track1.render_collect()
frames2 = track2.render_collect()

frames = mix(((10**(-3/20), frames1), (10**(-30/20), frames2)))

se = StereoAudio(frames, SAMPLE_RATE)
se.play(blocking=True)
