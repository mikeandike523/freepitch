#!/usr/bin/env python3
"""ratio_bins.py

Prime-limit ratio explorer.

Simplified pipeline (per user request):
- Build all "smooth" integers <= max_int using only the given primes.
- Form all reduced ratios n/d from those integers.
- **Immediately octave-fold each ratio into [1, 2)** by dividing/multiplying by 2.
  (It’s okay if this increases "complexity" via a larger denominator.)
- **Deduplicate after folding**.
- Sort globally by cents (equivalently numeric value, monotone on [1,2)).
- Bin by delta cents (default 5c).
- Within each bin, sort by (N + D) so cleaner fractions appear first.

Examples:
  python ratio_bins.py --primes 2,3,5 --max-int 128
  python ratio_bins.py --primes 2,3,5,7 --max-int 200 --delta-cents 3
  python ratio_bins.py --primes 3,5,7 --max-int 180 --format json
  python ratio_bins.py --primes 2,3,5 --max-int 128 --format csv > ratios.csv
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
from fractions import Fraction
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class RatioItem:
    ratio: str     # e.g. "3/2"
    num: int
    den: int
    value: float   # numeric ratio in [1,2)
    cents: float   # cents in [0,1200)
    complexity: int  # num + den


def parse_primes(primes_s: str) -> List[int]:
    parts: List[int] = []
    for chunk in primes_s.replace(",", " ").split():
        if chunk.strip():
            parts.append(int(chunk))
    if not parts:
        raise ValueError("No primes provided.")

    uniq = sorted(set(parts))
    for p in uniq:
        if p <= 1:
            raise ValueError(f"Invalid prime value: {p}")
    return uniq


def gen_smooth_numbers(primes: List[int], max_int: int) -> List[int]:
    """Generate all integers <= max_int whose prime factors are subset of primes. Includes 1."""
    smooth = {1}
    changed = True
    while changed:
        changed = False
        current = list(smooth)
        for x in current:
            for p in primes:
                y = x * p
                if y <= max_int and y not in smooth:
                    smooth.add(y)
                    changed = True
    return sorted(smooth)


def fold_to_octave(fr: Fraction) -> Fraction:
    """Fold a ratio into [1, 2) using octave equivalence."""
    r = fr
    two = Fraction(2, 1)
    one = Fraction(1, 1)

    # User explicitly asked for divide-by-2 when > 2.0.
    while r >= two:
        r /= 2

    # Keeping this makes the set truly "in the octave" [1,2).
    while r < one:
        r *= 2

    return r


def ratio_to_cents(fr: Fraction) -> float:
    # cents = 1200 * log2(ratio), and fr is in [1,2) so cents in [0,1200)
    return 1200.0 * math.log2(fr.numerator / fr.denominator)


def compute_item(fr_folded: Fraction) -> RatioItem:
    fr_reduced = fr_folded  # Fraction is always reduced
    val = fr_reduced.numerator / fr_reduced.denominator
    cents = ratio_to_cents(fr_reduced)
    # Guard for any tiny floating wobble:
    if cents < 0:
        cents += 1200.0
    elif cents >= 1200:
        cents -= 1200.0

    return RatioItem(
        ratio=f"{fr_reduced.numerator}/{fr_reduced.denominator}",
        num=fr_reduced.numerator,
        den=fr_reduced.denominator,
        value=val,
        cents=cents,
        complexity=fr_reduced.numerator + fr_reduced.denominator,
    )


def cents_distance(a: float, b: float) -> float:
    """Circular distance in cents on the octave [0,1200)."""
    d = abs(a - b)
    if d > 600.0:
        d = 1200.0 - d
    return d


def find_closest_per_edo(items: List[RatioItem], edo: int, per_step: int = 1, tolerance: float | None = None, optimize: str = "max") -> Dict[int, List[Tuple[RatioItem, float]]]:
    """For each EDO step, find the closest `per_step` RatioItem(s).

    Returns a dict mapping step index -> list of tuples (RatioItem, distance_cents).
    """
    if edo < 1:
        raise ValueError("edo must be >= 1")

    # Choose tie-breaker key consistent with binning optimize behavior
    if optimize == "sum":
        tie_key = lambda x: (x.complexity, x.cents, x.num, x.den)
    else:
        tie_key = lambda x: (max(x.num, x.den), x.complexity, x.cents, x.num, x.den)

    # Pre-sort items by the tie-breaker so stable tie resolution can be applied
    items_sorted_for_ties = sorted(items, key=tie_key)

    results: Dict[int, List[Tuple[RatioItem, float]]] = {}
    step_size = 1200.0 / edo
    for s in range(edo):
        target = s * step_size

        # compute distances
        scored = []
        for it in items_sorted_for_ties:
            dist = cents_distance(it.cents, target)
            if tolerance is None or dist <= tolerance:
                scored.append((dist, it))

        # pick best per_step by distance then tie_key
        scored.sort(key=lambda x: (x[0],) + tie_key(x[1]))
        chosen: List[Tuple[RatioItem, float]] = [(it, d) for (d, it) in scored[:per_step]]
        results[s] = chosen

    return results


def generate_folded_unique_ratios(primes: List[int], max_int: int) -> List[Fraction]:
    """Generate all unique ratios, octave-folded into [1,2) and deduped after folding."""
    smooth = gen_smooth_numbers(primes, max_int)
    folded: set[Fraction] = set()

    for n in smooth:
        for d in smooth:
            r = Fraction(n, d)
            r_fold = fold_to_octave(r)
            folded.add(r_fold)

    # Sort by numeric value in [1,2)
    return sorted(folded)


def bin_items(items: List[RatioItem], delta_cents: float, optimize: str = "max") -> Dict[str, List[RatioItem]]:
    """Bin by cents, then sort within each bin.

    optimize: 'sum' -> sort by N+D (complexity) first;
              'max' -> sort by max(N,D) first (default).
    """
    bins: Dict[int, List[RatioItem]] = {}
    for it in items:
        k = int(math.floor(it.cents / delta_cents))
        bins.setdefault(k, []).append(it)

    def sort_key(x: RatioItem):
        if optimize == "sum":
            return (x.complexity, x.cents, x.num, x.den)
        # default: minimize max(N,D), then N+D, then cents
        return (max(x.num, x.den), x.complexity, x.cents, x.num, x.den)

    labeled: Dict[str, List[RatioItem]] = {}
    for k in sorted(bins.keys()):
        lo = k * delta_cents
        hi = (k + 1) * delta_cents
        label = f"{lo:.1f}–{hi:.1f}"
        labeled[label] = sorted(bins[k], key=sort_key)

    return labeled


def _fmt_item_line(it: RatioItem, indent: str = "  ") -> str:
    return (
        f"{indent}- {it.ratio}"
        f"  (N+D={it.complexity}, cents={it.cents:0.3f}c, value={it.value:.10f})"
    )


def format_text(items: List[RatioItem], binned: Dict[str, List[RatioItem]], max_per_bin: int | None, delta_cents: float, optimize: str = "max") -> str:
    lines: List[str] = []

    # Small summary header
    if items:
        lines.append("• Summary")
        lines.append(f"  - total unique folded ratios: {len(items)}")
        lines.append(f"  - min: {items[0].ratio} ({items[0].cents:0.3f}c)")
        lines.append(f"  - max: {items[-1].ratio} ({items[-1].cents:0.3f}c)")
        lines.append("")

    for label, bin_list in binned.items():
        lines.append(f"• Bin {label} cents")
        shown = bin_list if max_per_bin is None else bin_list[:max_per_bin]
        for it in shown:
            lines.append(_fmt_item_line(it, indent="  "))
        if max_per_bin is not None and len(bin_list) > max_per_bin:
            lines.append(f"  - … ({len(bin_list) - max_per_bin} more)")
        lines.append("")

    # Coverage summary across full octave (0..1200 cents)
    num_bins = int(math.ceil(1200.0 / delta_cents))
    # Rebuild raw bins keyed by integer index so we can detect missing bins
    raw_bins: Dict[int, List[RatioItem]] = {}
    for it in items:
        k = int(math.floor(it.cents / delta_cents))
        raw_bins.setdefault(k, []).append(it)

    missing = [k for k in range(num_bins) if k not in raw_bins or len(raw_bins[k]) == 0]

    lines.append("• Bin coverage summary")
    if missing:
        lines.append(f"  - missing bins: {len(missing)} of {num_bins} total bins ({delta_cents}c steps)")
        for k in missing:
            lo = k * delta_cents
            hi = (k + 1) * delta_cents
            lines.append(f"  - missing: {lo:.1f}–{hi:.1f}c")
    else:
        if optimize == "sum":
            lines.append(f"  - all {num_bins} bins covered ({delta_cents}c steps). Lowest N+D ratio per bin:")
            keyfn = lambda x: (x.complexity, x.cents, x.num, x.den)
        else:
            lines.append(f"  - all {num_bins} bins covered ({delta_cents}c steps). Lowest max(N,D) ratio per bin:")
            keyfn = lambda x: (max(x.num, x.den), x.complexity, x.cents, x.num, x.den)

        for k in range(num_bins):
            bin_list = raw_bins[k]
            best = sorted(bin_list, key=keyfn)[0]
            lo = k * delta_cents
            hi = (k + 1) * delta_cents
            lines.append(f"  - {lo:.1f}–{hi:.1f}c: {best.ratio} (N+D={best.complexity}, {best.cents:0.3f}c)")
    lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def format_csv(items: List[RatioItem]) -> str:
    header = ["ratio", "num", "den", "value", "cents", "complexity"]
    out = [",".join(header)]
    for it in items:
        out.append(",".join([
            it.ratio,
            str(it.num),
            str(it.den),
            f"{it.value:.12g}",
            f"{it.cents:.6f}",
            str(it.complexity),
        ]))
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate prime-limit ratios, fold to an octave, sort, and bin by cents.")
    ap.add_argument("--primes", required=True, help="Comma/space-separated primes, e.g. 2,3,5 or '2 3 5'")
    ap.add_argument(
        "--max-int",
        type=int,
        default=128,
        help="Smooth-number limit: numerator/denominator factors built from primes, each <= max-int (default 128).",
    )
    ap.add_argument("--delta-cents", type=float, default=5.0, help="Bin size in cents (default 5).")
    ap.add_argument("--max-per-bin", type=int, default=None, help="Limit ratios printed per bin in text mode.")
    ap.add_argument("--edo", type=int, default=None, help="If set, find closest JI ratios to each EDO step (e.g. 12 for 12-edo).")
    ap.add_argument("--per-step", type=int, default=1, help="How many closest ratios to show per EDO step (default 1).")
    ap.add_argument("--tolerance-cents", type=float, default=None, help="Optional maximum distance in cents to accept a match for EDO mode.")
    ap.add_argument(
        "--optimize",
        choices=["max", "sum"],
        default="max",
        help="Optimization target when choosing representative ratios for bins: 'max' => minimize max(N,D) (default), 'sum' => minimize N+D.",
    )
    ap.add_argument("--format", choices=["text", "json", "csv"], default="text", help="Output format.")
    ap.add_argument(
        "--limit-output",
        type=int,
        default=None,
        help="Limit total printed items (after global cents sort, before binning).",
    )
    ap.add_argument(
        "--ratios-only",
        action="store_true",
        help="Only print one ratio per line (newline-delimited). Error if not all bins covered.",
    )

    args = ap.parse_args()

    primes = parse_primes(args.primes)
    if args.max_int < 1:
        raise SystemExit("--max-int must be >= 1")
    if args.delta_cents <= 0:
        raise SystemExit("--delta-cents must be > 0")
    if args.edo is not None and args.edo < 1:
        raise SystemExit("--edo must be >= 1 if specified")

    folded = generate_folded_unique_ratios(primes, args.max_int)
    items = [compute_item(fr) for fr in folded]

    # Global sort by cents/value (monotone on [1,2))
    items.sort(key=lambda x: (x.cents, x.complexity, x.num, x.den))

    if args.limit_output is not None:
        items = items[: args.limit_output]

    # If user requested EDO-centred matching, handle that mode and exit
    if args.edo is not None:
        edo_results = find_closest_per_edo(items, args.edo, per_step=args.per_step, tolerance=args.tolerance_cents, optimize=args.optimize)

        # CSV mode: emit one line per match: step, target_cents, ratio, cents, distance, complexity
        if args.format == "csv":
            header = ["step", "target_cents", "ratio", "num", "den", "cents", "distance_cents", "complexity"]
            out = [",".join(header)]
            step_size = 1200.0 / args.edo
            for s in range(args.edo):
                target = s * step_size
                matches = edo_results.get(s, [])
                if not matches:
                    out.append(f"{s},{target:.6f},,, , , ,")
                    continue
                for (it, dist) in matches:
                    out.append(",".join([
                        str(s),
                        f"{target:.6f}",
                        it.ratio,
                        str(it.num),
                        str(it.den),
                        f"{it.cents:.6f}",
                        f"{dist:.6f}",
                        str(it.complexity),
                    ]))
            print("\n".join(out))
            return

        # JSON: serialize per-step matches
        if args.format == "json":
            step_size = 1200.0 / args.edo
            payload = {
                "primes": primes,
                "max_int": args.max_int,
                "edo": args.edo,
                "per_step": args.per_step,
                "tolerance_cents": args.tolerance_cents,
                "optimize": args.optimize,
                "total_unique_folded": len(items),
                "steps": {},
            }
            for s in range(args.edo):
                target = s * step_size
                matches = edo_results.get(s, [])
                payload["steps"][str(s)] = {
                    "target_cents": target,
                    "matches": [
                        {**asdict(it), "distance_cents": d} for (it, d) in matches
                    ],
                }
            print(json.dumps(payload, indent=2))
            return

        # Text mode: pretty print per-step results
        step_size = 1200.0 / args.edo
        lines: List[str] = []
        lines.append(f"EDO: {args.edo} (step = {step_size:.6f} cents), primes={primes}, max_int={args.max_int}")
        for s in range(args.edo):
            target = s * step_size
            matches = edo_results.get(s, [])
            if not matches:
                lines.append(f"- Step {s}: target={target:.3f}c -> (no match within tolerance)")
                continue
            for i, (it, dist) in enumerate(matches):
                prefix = f"- Step {s}" if i == 0 else "  "
                lines.append(f"{prefix}: target={target:.3f}c -> {it.ratio} {it.cents:0.3f}c (Δ={dist:0.3f}c, N+D={it.complexity})")
        print("\n".join(lines))
        return

    if args.format == "csv":
        print(format_csv(items), end="")
        return

    # ratios-only: print one representative ratio per bin (newline-delimited), require full coverage
    if args.ratios_only:
        if args.delta_cents <= 0:
            raise SystemExit("--delta-cents must be > 0")
        num_bins = int(math.ceil(1200.0 / args.delta_cents))
        raw_bins: Dict[int, List[RatioItem]] = {}
        for it in items:
            k = int(math.floor(it.cents / args.delta_cents))
            raw_bins.setdefault(k, []).append(it)

        missing = [k for k in range(num_bins) if k not in raw_bins or len(raw_bins[k]) == 0]
        if missing:
            raise SystemExit(f"--ratios-only requires full coverage: {len(missing)} missing bins of {num_bins}")

        # choose one representative per bin according to optimization target
        if args.optimize == "sum":
            keyfn = lambda x: (x.complexity, x.cents, x.num, x.den)
        else:
            keyfn = lambda x: (max(x.num, x.den), x.complexity, x.cents, x.num, x.den)

        reps: List[RatioItem] = [sorted(raw_bins[k], key=keyfn)[0] for k in range(num_bins)]

        # Print representatives in ascending cents order (bin order)
        for it in reps:
            print(it.ratio)
        return

    if args.format == "json":
        binned = bin_items(items, args.delta_cents, args.optimize)
        payload = {
            "primes": primes,
            "max_int": args.max_int,
            "delta_cents": args.delta_cents,
            "optimize": args.optimize,
            "total_unique_folded": len(items),
            "bins": {label: [asdict(x) for x in bin_list] for label, bin_list in binned.items()},
        }
        print(json.dumps(payload, indent=2))
        return

    # text
    binned = bin_items(items, args.delta_cents, args.optimize)
    print(format_text(items, binned, args.max_per_bin, args.delta_cents, args.optimize), end="")


if __name__ == "__main__":
    main()
