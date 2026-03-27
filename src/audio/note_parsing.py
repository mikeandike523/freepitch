import ast
import operator as op
import re
from dataclasses import dataclass
from typing import Iterable

_INTERVAL_RE = re.compile(r'^([+-]\d+)$')


@dataclass(slots=False)
class ParsedNote:
    name: str | None
    octave: int | None
    duration: float
    volume: float = 1.0


@dataclass(slots=False)
class NoteName:
    text: str
    octave: int
    _names_to_index: dict[str, int]
    _scale: list[float]
    _reference_c_freq: float

    def __init__(
        self,
        text: str,
        octave: int,
        names_to_index: dict[str, int],
        scale: list[float],
        reference_c_freq: float,
    ):
        if text not in names_to_index:
            raise ValueError(f"Invalid note name: {text}")
        self.text = text
        self.octave = octave
        self._names_to_index = names_to_index
        self._scale = scale
        self._reference_c_freq = reference_c_freq

    def get_index_in_octave(self) -> int:
        return self._names_to_index[self.text]

    def get_note_id(self) -> int:
        return self.octave * len(self._scale) + self.get_index_in_octave()

    def get_pitch(self) -> float:
        return (
            self._scale[self.get_index_in_octave()]
            * 2 ** (self.octave - 4)
            * self._reference_c_freq
        )

    def __repr__(self) -> str:
        return f"{self.text}{self.octave}"


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

    def visit_Num(self, node: ast.Num):  # pragma: no cover
        return float(node.n)

    def generic_visit(self, node):
        raise ValueError(f"Disallowed syntax in duration expression: {type(node).__name__}")


class NoteParser:
    def __init__(
        self,
        names_to_index: dict[str, int],
        scale: list[float],
        reference_c_freq: float,
        *,
        bpm: float = 120,
        sharpwards_names_to_index: dict[str, int] | None = None,
        flatwards_names_to_index: dict[str, int] | None = None,
    ):
        self._names_to_index = names_to_index
        self._scale = scale
        self._reference_c_freq = reference_c_freq
        quarter = 60 / bpm
        eighth = quarter / 2
        half= quarter * 2

        # Reverse mappings for interval resolution
        if sharpwards_names_to_index is not None:
            self._index_to_sharp_name: dict[int, str] = {
                idx: name for name, idx in sharpwards_names_to_index.items()
            }
        else:
            self._index_to_sharp_name = {}
            for name, idx in names_to_index.items():
                if idx not in self._index_to_sharp_name:
                    self._index_to_sharp_name[idx] = name

        if flatwards_names_to_index is not None:
            self._index_to_flat_name: dict[int, str] = {
                idx: name for name, idx in flatwards_names_to_index.items()
            }
        else:
            self._index_to_flat_name = self._index_to_sharp_name

        self._duration_env_base = {
            "q": quarter,
            "e": eighth,
            "h": half,
            "QUARTER": quarter,
            "EIGHTH": eighth,
            "HALF": half,
            "BPM": bpm,
        }

    def note_name(self, text: str, octave: int) -> NoteName:
        return NoteName(
            text,
            octave,
            self._names_to_index,
            self._scale,
            self._reference_c_freq,
        )

    def eval_duration_expr(self, expr: str, *, env: dict[str, float] | None = None) -> float:
        expr = expr.strip()
        if not expr:
            raise ValueError("Empty duration")

        merged_env = dict(self._duration_env_base)
        if env:
            merged_env.update(env)

        try:
            tree = ast.parse(expr, mode="eval")
        except SyntaxError as exc:
            raise ValueError(f"Bad duration expression: {expr!r}") from exc

        value = _SafeExprEval(merged_env).visit(tree)
        if not (value >= 0):
            raise ValueError(f"Duration must be >= 0, got {value}")
        return float(value)

    def parse_lines(
        self, text: str, *, dur_env: dict[str, float] | None = None
    ) -> list[ParsedNote]:
        """Return list of ParsedNote entries for dot-joined notation."""

        rest_tokens = {"R", "-", "rest", "REST"}
        scale_len = len(self._scale)
        prev_step: list[int | None] = [None]
        prev_name: list[str | None] = [None]

        def _parse_dur_vol(dur_txt: str, token: str) -> tuple[float, float]:
            if ":" in dur_txt:
                dur_expr, vol_txt = dur_txt.split(":", 1)
                if not vol_txt:
                    raise ValueError(f"Bad volume in token: {token!r}")
                volume = float(vol_txt)
                if volume < 0:
                    raise ValueError(f"Volume must be >= 0, got {volume}")
            else:
                dur_expr = dur_txt
                volume = 1.0
            return self.eval_duration_expr(dur_expr, env=dur_env), volume

        def parse_dot_token(token: str) -> ParsedNote:
            if "." not in token:
                raise ValueError(f"Bad token (missing dots): {token!r}")
            head, rest = token.split(".", 1)
            if head in rest_tokens:
                if not rest:
                    raise ValueError(f"Bad rest token: {token!r}")
                if ":" in rest:
                    raise ValueError(f"Rest tokens cannot include volume: {token!r}")
                return ParsedNote(
                    None,
                    None,
                    self.eval_duration_expr(rest, env=dur_env),
                )

            # Interval token: +N or -N relative to previous note in EDO steps
            if _INTERVAL_RE.match(head):
                if prev_step[0] is None:
                    raise ValueError(
                        f"Interval token {token!r} used before any absolute note"
                    )
                offset = int(head)
                new_step = prev_step[0] + offset
                octave, index_in_octave = divmod(new_step, scale_len)
                if offset == 0:
                    name = prev_name[0]
                elif offset > 0:
                    name = self._index_to_sharp_name[index_in_octave]
                else:
                    name = self._index_to_flat_name[index_in_octave]
                duration, volume = _parse_dur_vol(rest, token)
                prev_step[0] = new_step
                prev_name[0] = name
                return ParsedNote(name, octave, duration, volume)

            if head not in self._names_to_index:
                raise ValueError(f"Bad note token: {token!r}")
            parts = token.split(".", 2)
            if len(parts) != 3:
                raise ValueError(f"Bad note token: {token!r}")
            _, octave_txt, dur_txt = parts
            if not octave_txt or not dur_txt:
                raise ValueError(f"Bad note token: {token!r}")
            octave = int(octave_txt)
            duration, volume = _parse_dur_vol(dur_txt, token)
            prev_step[0] = octave * scale_len + self._names_to_index[head]
            prev_name[0] = head
            return ParsedNote(head, octave, duration, volume)

        out: list[ParsedNote] = []
        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#"):
                continue

            parts = line.split()
            for token in parts:
                out.append(parse_dot_token(token))

        return out


def build_note_parser(
    names_to_index: dict[str, int],
    scale: Iterable[float],
    reference_c_freq: float,
    *,
    bpm: float = 120,
    sharpwards_names_to_index: dict[str, int] | None = None,
    flatwards_names_to_index: dict[str, int] | None = None,
) -> NoteParser:
    return NoteParser(
        names_to_index,
        list(scale),
        reference_c_freq,
        bpm=bpm,
        sharpwards_names_to_index=sharpwards_names_to_index,
        flatwards_names_to_index=flatwards_names_to_index,
    )
