#!/usr/bin/env python3
"""Plot layer polarization and WC-SRO curves from an SRO phase summary."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np


THIS_DIR = Path(__file__).resolve().parent
CONFIG = SimpleNamespace(
    summary_csv=THIS_DIR / "sro_analysis" / "sro_phase_summary.csv",
    shell_id=1,
    use_unordered_pairs=True,
    use_absolute_polarization=False,
    use_canonical_polarization=True,
    output_png=THIS_DIR / "sro_analysis" / "sro_overview.png",
    output_pdf=THIS_DIR / "sro_analysis" / "sro_overview.pdf",
    figsize=(13.0, 5.0),
    dpi=200,
)


def _read_summary(path: Path) -> list[dict[str, float | str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            parsed: dict[str, float | str] = {}
            for key, value in row.items():
                if value is None or value == "":
                    parsed[key] = np.nan
                    continue
                try:
                    parsed[key] = float(value)
                except ValueError:
                    parsed[key] = value
            rows.append(parsed)
    if not rows:
        raise ValueError(f"No rows found in {path}.")
    rows.sort(key=lambda r: float(r.get("temperature_K", np.nan)))
    return rows


def _detect_species(fieldnames: list[str]) -> list[str]:
    pattern = re.compile(r"^global_frac_(.+)_mean$")
    species = []
    for field in fieldnames:
        match = pattern.match(field)
        if match:
            species.append(match.group(1))
    if not species:
        raise ValueError("Could not infer species from global_frac_* columns.")
    return sorted(species)


def _collect_wc_pairs(fieldnames: list[str], shell_id: int) -> list[tuple[str, str]]:
    pattern = re.compile(rf"^wc_shell{shell_id}_global_(.+)_(.+)_mean$")
    pairs = []
    for field in fieldnames:
        match = pattern.match(field)
        if match:
            pairs.append((match.group(1), match.group(2)))
    if not pairs:
        raise ValueError(f"No wc_shell{shell_id}_global_* columns found.")
    return sorted(pairs)


def _polarization_series(
    rows: list[dict[str, float | str]],
    species: list[str],
    use_absolute: bool,
    use_canonical: bool,
) -> dict[str, np.ndarray]:
    base = "layer_abs_top_minus_bottom_frac" if use_absolute else "layer_top_minus_bottom_frac"
    prefix = "canonical_" if use_canonical else ""

    def _series_for(prefix_value: str) -> dict[str, np.ndarray]:
        return {
            sp: np.array(
                [float(row.get(f"{prefix_value}{base}_{sp}_mean", np.nan)) for row in rows],
                dtype=float,
            )
            for sp in species
        }

    data = _series_for(prefix)
    if use_canonical and all(np.all(np.isnan(vals)) for vals in data.values()):
        data = _series_for("")
    return data


def _wc_series(
    rows: list[dict[str, float | str]],
    ordered_pairs: list[tuple[str, str]],
    shell_id: int,
    use_unordered_pairs: bool,
):
    if not use_unordered_pairs:
        labels = [(i, j, f"{i}-{j}") for (i, j) in ordered_pairs]
        data = {
            label: np.array(
                [float(row.get(f"wc_shell{shell_id}_global_{i}_{j}_mean", np.nan)) for row in rows],
                dtype=float,
            )
            for i, j, label in labels
        }
        return labels, data

    grouped: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for i, j in ordered_pairs:
        key = tuple(sorted((i, j)))
        grouped.setdefault(key, []).append((i, j))

    labels = []
    data: dict[str, np.ndarray] = {}
    for key in sorted(grouped):
        members = grouped[key]
        label = f"{key[0]}-{key[1]}"
        values = []
        for row in rows:
            row_vals = [
                float(row.get(f"wc_shell{shell_id}_global_{i}_{j}_mean", np.nan))
                for i, j in members
            ]
            values.append(float(np.nanmean(row_vals)))
        labels.append((key[0], key[1], label))
        data[label] = np.array(values, dtype=float)
    return labels, data


def main() -> None:
    rows = _read_summary(Path(CONFIG.summary_csv))
    fieldnames = list(rows[0].keys())
    species = _detect_species(fieldnames)
    ordered_pairs = _collect_wc_pairs(fieldnames, CONFIG.shell_id)

    temps = np.array([float(row["temperature_K"]) for row in rows], dtype=float)
    polarization = _polarization_series(
        rows,
        species,
        CONFIG.use_absolute_polarization,
        CONFIG.use_canonical_polarization,
    )
    pair_labels, wc_data = _wc_series(rows, ordered_pairs, CONFIG.shell_id, CONFIG.use_unordered_pairs)

    fig, (ax_pol, ax_wc) = plt.subplots(1, 2, figsize=CONFIG.figsize, dpi=CONFIG.dpi)
    cmap = plt.get_cmap("tab10")

    for idx, sp in enumerate(species):
        ax_pol.plot(
            temps,
            polarization[sp],
            label=sp,
            color=cmap(idx % 10),
            marker="o",
            markersize=3,
            linewidth=1.8,
        )
    ax_pol.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    ax_pol.set_xlabel("Temperature (K)")
    ax_pol.set_ylabel("|layer top - bottom|" if CONFIG.use_absolute_polarization else "Layer polarization")
    ax_pol.set_title("Absolute layer polarization" if CONFIG.use_absolute_polarization else "Layer polarization")
    ax_pol.grid(True, alpha=0.25)
    ax_pol.legend(fontsize=8)

    for idx, (_, _, label) in enumerate(pair_labels):
        ax_wc.plot(
            temps,
            wc_data[label],
            label=label,
            color=cmap(idx % 10),
            marker="o",
            markersize=3,
            linewidth=1.8,
        )
    ax_wc.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    ax_wc.set_xlabel("Temperature (K)")
    ax_wc.set_ylabel(r"WC-SRO $\alpha_{ij}$")
    ax_wc.set_title(f"Shell {CONFIG.shell_id} WC-SRO")
    ax_wc.grid(True, alpha=0.25)
    ax_wc.legend(fontsize=8, ncol=max(1, min(3, len(pair_labels))))

    fig.tight_layout()
    CONFIG.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(CONFIG.output_png, bbox_inches="tight")
    fig.savefig(CONFIG.output_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"Read summary: {CONFIG.summary_csv}")
    print(f"Detected species: {species}")
    print(f"Plotted WC pairs: {[label for _, _, label in pair_labels]}")
    print("Polarization mode:", "absolute" if CONFIG.use_absolute_polarization else "signed")
    print("Canonical polarization:", CONFIG.use_canonical_polarization)
    print(f"Wrote: {CONFIG.output_png}")
    print(f"Wrote: {CONFIG.output_pdf}")


if __name__ == "__main__":
    main()
