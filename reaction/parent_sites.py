"""Parent-site aggregation utilities for reaction post-processing."""

from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np


def temperature_label(value: object) -> str:
    try:
        temp = float(value)
    except (TypeError, ValueError):
        return "unknown"
    if not np.isfinite(temp):
        return "unknown"
    if abs(temp - round(temp)) < 1.0e-6:
        return f"{int(round(temp))}K"
    return f"{temp:g}K"


def support_tuple(row: dict[str, object]) -> tuple[int, ...]:
    value = str(row.get("support_indices", "")).strip()
    if not value:
        return ()
    return tuple(sorted(int(part) for part in value.split()))


def canonical_parent_site_id(row: dict[str, object]) -> str:
    """Return a stable site id independent of support-index ordering."""

    support = support_tuple(row)
    support_key = "-".join(str(idx) for idx in support) if support else "none"
    return f"{row.get('site_type', 'site')}:{support_key}"


def sanitize_path_component(value: object) -> str:
    """Return a filesystem-safe path component for generated output dirs."""

    text = str(value).strip().replace(":", "_")
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_.")
    return text or "unknown"


def site_directory_name(site_id: object) -> str:
    """Return the stable directory name for a canonical parent site id."""

    return sanitize_path_component(site_id)


def safe_float(value: object, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def mean(rows: Sequence[dict[str, object]], key: str) -> float:
    values = [safe_float(row.get(key)) for row in rows]
    values = [value for value in values if np.isfinite(value)]
    if not values:
        return float("nan")
    return float(np.mean(values))


def aggregate_parent_site_rows(
    rows: Sequence[dict[str, object]],
    *,
    temperature_labels: Sequence[str] | None = None,
) -> list[dict[str, object]]:
    """Aggregate per-frame motif rows into canonical parent-site populations."""

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[canonical_parent_site_id(row)].append(dict(row))

    total = len(rows)
    if temperature_labels is None:
        temperature_labels = sorted(
            {
                str(
                    row.get(
                        "_temperature_label",
                        temperature_label(row.get("temperature_K")),
                    )
                )
                for row in rows
            }
        )

    counts_by_temperature = Counter(
        str(
            row.get(
                "_temperature_label",
                temperature_label(row.get("temperature_K")),
            )
        )
        for row in rows
    )

    summary: list[dict[str, object]] = []
    for site_id, site_rows in grouped.items():
        first = site_rows[0]
        site_temperature_counts = Counter(
            str(
                row.get(
                    "_temperature_label",
                    temperature_label(row.get("temperature_K")),
                )
            )
            for row in site_rows
        )
        support = support_tuple(first)
        support_keys = Counter(str(row.get("support_key", "")) for row in site_rows)
        shell1_keys = Counter(str(row.get("shell1_key", "")) for row in site_rows)
        shell2_keys = Counter(str(row.get("shell2_key", "")) for row in site_rows)
        functionals = Counter(str(row.get("functional_count", "")) for row in site_rows)
        motif_keys = Counter(str(row.get("motif_key", "")) for row in site_rows)

        row: dict[str, object] = {
            "site_id": site_id,
            "site_type": str(first.get("site_type", "")),
            "support_indices_sorted": " ".join(str(idx) for idx in support),
            "support_key_mode": support_keys.most_common(1)[0][0],
            "samples_total": int(len(site_rows)),
            "population_total": float(len(site_rows) / total) if total else float("nan"),
            "shell1_key_mode": shell1_keys.most_common(1)[0][0],
            "shell2_key_mode": shell2_keys.most_common(1)[0][0],
            "functional_count_mode": functionals.most_common(1)[0][0],
            "motif_key_mode": motif_keys.most_common(1)[0][0],
            "anchor_site_xy_dist_A_mean": mean(site_rows, "anchor_site_xy_dist_A"),
            "anchor_support_min_dist_A_mean": mean(
                site_rows,
                "anchor_support_min_dist_A",
            ),
            "anchor_z_offset_A_mean": mean(site_rows, "anchor_z_offset_A"),
        }
        for label in temperature_labels:
            count = site_temperature_counts.get(str(label), 0)
            denom = counts_by_temperature.get(str(label), 0)
            row[f"samples_{label}"] = int(count)
            row[f"population_{label}"] = float(count / denom) if denom else float("nan")
        summary.append(row)

    summary.sort(
        key=lambda row: (
            -int(row["samples_total"]),
            str(row["site_type"]),
            str(row["support_indices_sorted"]),
        )
    )
    return summary


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def representative_score(
    row: dict[str, object],
    rows: Sequence[dict[str, object]],
) -> float:
    fields = (
        "anchor_site_xy_dist_A",
        "anchor_support_min_dist_A",
        "anchor_z_offset_A",
    )
    score = 0.0
    for field in fields:
        values = np.asarray([safe_float(item.get(field)) for item in rows], dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        field_mean = float(np.mean(values))
        field_std = float(np.std(values))
        if field_std < 1.0e-12:
            field_std = 1.0
        score += ((safe_float(row.get(field)) - field_mean) / field_std) ** 2
    return float(score)
