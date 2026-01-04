import ast
import operator as op
from dataclasses import dataclass

REFERENCE_A_FREQ = 440

# C5 is 3 semitones above A4
# The -1 makes it go to the octave below
REFERENCE_C_FREQ = REFERENCE_A_FREQ * 9 / 16

#  ./__inenv python experiments/pick_ji.py --primes=2,3,5 --max-int=64 --edo=12 --per-step=1 --format ratios
SCALE = [
    1 / 1,
    16 / 15,
    9 / 8,
    32 / 27,
    5 / 4,
    4 / 3,
    45 / 32,
    3 / 2,
    8 / 5,
    27 / 16,
    16 / 9,
    15 / 8,
]

NAMES_TO_INDEX = {
    "C": 0,
    "C#": 1,
    "Db": 1,
    "D": 2,
    "D#": 3,
    "Eb": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "Gb": 6,
    "G": 7,
    "G#": 8,
    "Ab": 8,
    "A": 9,
    "A#": 10,
    "Bb": 10,
    "B": 11,
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
        return SCALE[self.get_index_in_octave()] * 2 ** (self.octave - 4) * REFERENCE_C_FREQ

    def __repr__(self):
        return f"{self.text}{self.octave}"


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
# SIMPLE PARSER (dot-joined notation only)
# ============================================================

def parse_lines(text: str, *, dur_env: dict[str, float] | None = None):
    """Return list of tuples (note_text|None, octave|None, duration_seconds).

    Dot-joined notation allows multiple notes per line:
      C.3.e*2 D.3.e*2 R.e

    Rest tokens use a dot-joined duration:
      R.e
      -.e
    """

    rest_tokens = {"R", "-", "rest", "REST"}

    def parse_dot_token(token: str):
        if "." not in token:
            raise ValueError(f"Bad token (missing dots): {token!r}")
        head = token.split(".", 1)[0]
        if head in rest_tokens:
            _, dur_txt = token.split(".", 1)
            if not dur_txt:
                raise ValueError(f"Bad rest token: {token!r}")
            return (None, None, eval_duration_expr(dur_txt, env=dur_env))
        if head not in NAMES_TO_INDEX:
            raise ValueError(f"Bad note token: {token!r}")
        parts = token.split(".", 2)
        if len(parts) != 3:
            raise ValueError(f"Bad note token: {token!r}")
        _, octave_txt, dur_txt = parts
        if not octave_txt or not dur_txt:
            raise ValueError(f"Bad note token: {token!r}")
        octave = int(octave_txt)
        return (head, octave, eval_duration_expr(dur_txt, env=dur_env))

    out = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line == "#":
            continue
        if line.startswith("#"):
            raise ValueError(f"Comments must be a lone '#': {raw!r}")

        parts = line.split()
        for token in parts:
            out.append(parse_dot_token(token))

    return out
