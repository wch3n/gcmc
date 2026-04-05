#!/usr/bin/env python3
"""Summarize adsorbate GCMC scan outputs from one output directory."""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import Counter
from pathlib import Path
from typing import Iterable


KB_EV_PER_K = 8.617333262145e-5


def _parse_mu_from_name(name: str) -> float:
    token = name
    if token.startswith("mu_"):
        token = token[3:]
    if token.startswith("m"):
        sign = -1.0
        token = token[1:]
    else:
        sign = 1.0
    token = token.replace("p", ".")
    return sign * float(token)


def _parse_seed_from_name(path: Path) -> int | None:
    match = re.search(r"seed_(\d+)", path.stem)
    return int(match.group(1)) if match else None


def _read_dat(
    path: Path, equilibration_sweeps: int = 0
) -> tuple[list[int], list[float]]:
    occupancies: list[int] = []
    energies: list[float] = []
    with path.open() as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            sweep = int(float(parts[0]))
            if sweep <= equilibration_sweeps:
                continue
            energies.append(float(parts[1]))
            occupancies.append(int(float(parts[2])))
    return occupancies, energies


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def _switch_count(values: list[int]) -> int:
    return int(sum(a != b for a, b in zip(values, values[1:])))


def _histogram(values: list[int]) -> dict[int, int]:
    return dict(sorted(Counter(int(v) for v in values).items()))


def _half_means(values: list[int]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    mid = len(values) // 2
    if mid == 0:
        val = float(values[0])
        return val, val
    return _mean(values[:mid]), _mean(values[mid:])


def _delta_f_01(mu: float, mean_n: float, temperature: float) -> float:
    if not (0.0 < mean_n < 1.0):
        return float("nan")
    p1 = mean_n
    p0 = 1.0 - p1
    return float(mu + KB_EV_PER_K * temperature * math.log(p0 / p1))


def _format_hist(hist: dict[int, int]) -> str:
    return "{" + ", ".join(f"{k}:{v}" for k, v in hist.items()) + "}"


def analyze_directory(
    scan_dir: Path, temperature: float, equilibration_sweeps: int = 0
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    per_seed_rows: list[dict[str, object]] = []
    pooled_rows: list[dict[str, object]] = []

    mu_dirs = sorted(
        [path for path in scan_dir.iterdir() if path.is_dir() and path.name.startswith("mu_")],
        key=lambda path: _parse_mu_from_name(path.name),
    )
    for mu_dir in mu_dirs:
        mu = _parse_mu_from_name(mu_dir.name)
        pooled_occ: list[int] = []
        pooled_energy: list[float] = []
        for dat_path in sorted(mu_dir.glob("*.dat")):
            occupancies, energies = _read_dat(
                dat_path, equilibration_sweeps=equilibration_sweeps
            )
            seed = _parse_seed_from_name(dat_path)
            mean_n = _mean(occupancies)
            mean_e = _mean(energies)
            first_half_n, second_half_n = _half_means(occupancies)
            row = {
                "mu": mu,
                "mu_label": mu_dir.name,
                "seed": seed,
                "samples": len(occupancies),
                "mean_n": mean_n,
                "mean_e_eV": mean_e,
                "switches": _switch_count(occupancies),
                "final_n": occupancies[-1] if occupancies else None,
                "first_half_mean_n": first_half_n,
                "second_half_mean_n": second_half_n,
                "delta_f_01_eV": _delta_f_01(mu, mean_n, temperature),
                "histogram": _format_hist(_histogram(occupancies)),
                "dat_file": str(dat_path),
            }
            per_seed_rows.append(row)
            pooled_occ.extend(occupancies)
            pooled_energy.extend(energies)

        if pooled_occ:
            pooled_mean_n = _mean(pooled_occ)
            pooled_rows.append(
                {
                    "mu": mu,
                    "mu_label": mu_dir.name,
                    "samples": len(pooled_occ),
                    "mean_n": pooled_mean_n,
                    "mean_e_eV": _mean(pooled_energy),
                    "switches": _switch_count(pooled_occ),
                    "first_half_mean_n": _half_means(pooled_occ)[0],
                    "second_half_mean_n": _half_means(pooled_occ)[1],
                    "delta_f_01_eV": _delta_f_01(mu, pooled_mean_n, temperature),
                    "histogram": _format_hist(_histogram(pooled_occ)),
                }
            )

    return per_seed_rows, pooled_rows


def _write_csv(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scan-dir",
        type=Path,
        required=True,
        help="GCMC scan output directory containing mu_* subdirectories.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=300.0,
        help="Temperature in K used for DeltaF_01 estimation (default: 300).",
    )
    parser.add_argument(
        "--equilibration-sweeps",
        type=int,
        default=0,
        help="Discard .dat entries with recorded sweep <= this value (default: 0).",
    )
    parser.add_argument(
        "--out-prefix",
        type=Path,
        default=None,
        help="Optional CSV output prefix. Writes '<prefix>_per_seed.csv' and '<prefix>_pooled.csv'.",
    )
    args = parser.parse_args()

    per_seed_rows, pooled_rows = analyze_directory(
        args.scan_dir,
        args.temperature,
        equilibration_sweeps=args.equilibration_sweeps,
    )
    if not per_seed_rows:
        raise SystemExit(f"No .dat files found under {args.scan_dir}")

    print("Per-mu pooled summary:")
    for row in pooled_rows:
        delta_f = row["delta_f_01_eV"]
        delta_f_str = "nan" if math.isnan(delta_f) else f"{delta_f: .4f}"
        print(
            f"  {row['mu_label']:>10s} | mu={row['mu']: .4f} eV | "
            f"P1={row['mean_n']:.3f} | switches={row['switches']:3d} | "
            f"half=({row['first_half_mean_n']:.3f}, {row['second_half_mean_n']:.3f}) | "
            f"DeltaF01={delta_f_str} eV | hist={row['histogram']}"
        )

    print("\nPer-seed summary:")
    for row in per_seed_rows:
        delta_f = row["delta_f_01_eV"]
        delta_f_str = "nan" if math.isnan(delta_f) else f"{delta_f: .4f}"
        print(
            f"  {row['mu_label']:>10s} seed={row['seed']:>3} | "
            f"P1={row['mean_n']:.3f} | switches={row['switches']:3d} | "
            f"final_N={row['final_n']} | half=({row['first_half_mean_n']:.3f}, {row['second_half_mean_n']:.3f}) | "
            f"DeltaF01={delta_f_str} eV"
        )

    if args.out_prefix is not None:
        prefix = Path(args.out_prefix)
        _write_csv(per_seed_rows, prefix.with_name(prefix.name + "_per_seed.csv"))
        _write_csv(pooled_rows, prefix.with_name(prefix.name + "_pooled.csv"))
        print(
            f"\nWrote CSV files:\n"
            f"  {prefix.with_name(prefix.name + '_per_seed.csv')}\n"
            f"  {prefix.with_name(prefix.name + '_pooled.csv')}"
        )


if __name__ == "__main__":
    main()
