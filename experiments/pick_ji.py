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


def bin_items(items: List[RatioItem], delta_cents: float) -> Dict[str, List[RatioItem]]:
    """Bin by cents, then sort within each bin by (N+D) (cleaner first)."""
    bins: Dict[int, List[RatioItem]] = {}
    for it in items:
        k = int(math.floor(it.cents / delta_cents))
        bins.setdefault(k, []).append(it)

    labeled: Dict[str, List[RatioItem]] = {}
    for k in sorted(bins.keys()):
        lo = k * delta_cents
        hi = (k + 1) * delta_cents
        label = f"{lo:.1f}–{hi:.1f}"
        labeled[label] = sorted(bins[k], key=lambda x: (x.complexity, x.cents, x.num, x.den))

    return labeled


def _fmt_item_line(it: RatioItem, indent: str = "  ") -> str:
    return (
        f"{indent}- {it.ratio}"
        f"  (N+D={it.complexity}, cents={it.cents:0.3f}c, value={it.value:.10f})"
    )


def format_text(items: List[RatioItem], binned: Dict[str, List[RatioItem]], max_per_bin: int | None, delta_cents: float) -> str:
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
        lines.append(f"  - all {num_bins} bins covered ({delta_cents}c steps). Lowest N+D ratio per bin:")
        for k in range(num_bins):
            bin_list = raw_bins[k]
            best = sorted(bin_list, key=lambda x: (x.complexity, x.cents, x.num, x.den))[0]
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
    ap.add_argument("--format", choices=["text", "json", "csv"], default="text", help="Output format.")
    ap.add_argument(
        "--limit-output",
        type=int,
        default=None,
        help="Limit total printed items (after global cents sort, before binning).",
    )

    args = ap.parse_args()

    primes = parse_primes(args.primes)
    if args.max_int < 1:
        raise SystemExit("--max-int must be >= 1")
    if args.delta_cents <= 0:
        raise SystemExit("--delta-cents must be > 0")

    folded = generate_folded_unique_ratios(primes, args.max_int)
    items = [compute_item(fr) for fr in folded]

    # Global sort by cents/value (monotone on [1,2))
    items.sort(key=lambda x: (x.cents, x.complexity, x.num, x.den))

    if args.limit_output is not None:
        items = items[: args.limit_output]

    if args.format == "csv":
        print(format_csv(items), end="")
        return

    if args.format == "json":
        binned = bin_items(items, args.delta_cents)
        payload = {
            "primes": primes,
            "max_int": args.max_int,
            "delta_cents": args.delta_cents,
            "total_unique_folded": len(items),
            "bins": {label: [asdict(x) for x in bin_list] for label, bin_list in binned.items()},
        }
        print(json.dumps(payload, indent=2))
        return

    # text
    binned = bin_items(items, args.delta_cents)
    print(format_text(items, binned, args.max_per_bin, args.delta_cents), end="")


if __name__ == "__main__":
    main()
