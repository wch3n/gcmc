#!/usr/bin/env python3
"""Plot layer polarization and layer-resolved WC-SRO from an SRO summary."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np


THIS_DIR = Path(__file__).resolve().parent
EXAMPLE_DIR = THIS_DIR.parent
CONFIG = SimpleNamespace(
    summary_csv=EXAMPLE_DIR / "outputs" / "sro_analysis" / "sro_phase_summary.csv",
    same_layer_shell_id=1,
    cross_layer_shell_id=1,
    use_unordered_pairs=True,
    use_absolute_polarization=False,
    use_canonical_polarization=True,
    polarization_png=EXAMPLE_DIR / "outputs" / "sro_analysis" / "layer_polarization.png",
    polarization_pdf=EXAMPLE_DIR / "outputs" / "sro_analysis" / "layer_polarization.pdf",
    wc_png=EXAMPLE_DIR / "outputs" / "sro_analysis" / "wc_sro_split.png",
    wc_pdf=EXAMPLE_DIR / "outputs" / "sro_analysis" / "wc_sro_split.pdf",
    polarization_figsize=(6.8, 5.0),
    wc_figsize=(13.0, 5.0),
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


def _collect_wc_pairs(
    fieldnames: list[str],
    relation: str,
    relation_shell_id: int,
) -> list[tuple[str, str]]:
    pattern = re.compile(
        rf"^wc_{relation}_shell{relation_shell_id}_(.+)_(.+)_mean$"
    )
    pairs = []
    for field in fieldnames:
        match = pattern.match(field)
        if match:
            pairs.append((match.group(1), match.group(2)))
    if not pairs:
        legacy_pattern = re.compile(
            rf"^wc_shell{relation_shell_id}_{relation}_(.+)_(.+)_mean$"
        )
        for field in fieldnames:
            match = legacy_pattern.match(field)
            if match:
                pairs.append((match.group(1), match.group(2)))
    if not pairs:
        raise ValueError(
            f"No anisotropic WC-SRO columns found for {relation} shell {relation_shell_id}. "
            "Rerun the SRO analysis with the updated analyzer."
        )
    return sorted(set(pairs))


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


def _wc_relation_series(
    rows: list[dict[str, float | str]],
    ordered_pairs: list[tuple[str, str]],
    relation: str,
    relation_shell_id: int,
    use_unordered_pairs: bool,
):
    relation_key = f"wc_{relation}_shell{relation_shell_id}"
    legacy_key = f"wc_shell{relation_shell_id}_{relation}"

    def _value(row: dict[str, float | str], i: str, j: str) -> float:
        key = f"{relation_key}_{i}_{j}_mean"
        if key in row:
            return float(row.get(key, np.nan))
        return float(row.get(f"{legacy_key}_{i}_{j}_mean", np.nan))

    if not use_unordered_pairs:
        labels = [(i, j, f"{i}-{j}") for (i, j) in ordered_pairs]
        data = {
            label: np.array(
                [_value(row, i, j) for row in rows],
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
            row_vals = [_value(row, i, j) for i, j in members]
            values.append(float(np.nanmean(row_vals)))
        labels.append((key[0], key[1], label))
        data[label] = np.array(values, dtype=float)
    return labels, data


def _plot_polarization(
    temps: np.ndarray,
    polarization: dict[str, np.ndarray],
) -> plt.Figure:
    fig, ax = plt.subplots(1, 1, figsize=CONFIG.polarization_figsize, dpi=CONFIG.dpi)
    cmap = plt.get_cmap("tab10")

    for idx, (sp, values) in enumerate(polarization.items()):
        ax.plot(
            temps,
            values,
            label=sp,
            color=cmap(idx % 10),
            marker="o",
            markersize=3,
            linewidth=1.8,
        )

    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("|layer top - bottom|" if CONFIG.use_absolute_polarization else "Layer polarization")
    ax.set_title("Absolute layer polarization" if CONFIG.use_absolute_polarization else "Layer polarization")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def _plot_wc(
    temps: np.ndarray,
    same_labels,
    same_data,
    cross_labels,
    cross_data,
) -> plt.Figure:
    fig, (ax_cross, ax_same) = plt.subplots(1, 2, figsize=CONFIG.wc_figsize, dpi=CONFIG.dpi)
    cmap = plt.get_cmap("tab10")

    def _plot_one(ax, labels, data, title):
        for idx, (_, _, label) in enumerate(labels):
            ax.plot(
                temps,
                data[label],
                label=label,
                color=cmap(idx % 10),
                marker="o",
                markersize=3,
                linewidth=1.8,
            )
        ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
        ax.set_xlabel("Temperature (K)")
        ax.set_ylabel(r"WC-SRO $\alpha_{ij}$")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, ncol=max(1, min(3, len(labels))))

    _plot_one(
        ax_cross,
        cross_labels,
        cross_data,
        f"Interlayer shell {CONFIG.cross_layer_shell_id} WC-SRO",
    )
    _plot_one(
        ax_same,
        same_labels,
        same_data,
        f"Intralayer shell {CONFIG.same_layer_shell_id} WC-SRO",
    )
    fig.tight_layout()
    return fig


def main() -> None:
    rows = _read_summary(Path(CONFIG.summary_csv))
    fieldnames = list(rows[0].keys())
    species = _detect_species(fieldnames)
    same_pairs = _collect_wc_pairs(fieldnames, "same_layer", CONFIG.same_layer_shell_id)
    cross_pairs = _collect_wc_pairs(fieldnames, "cross_layer", CONFIG.cross_layer_shell_id)

    temps = np.array([float(row["temperature_K"]) for row in rows], dtype=float)
    polarization = _polarization_series(
        rows,
        species,
        CONFIG.use_absolute_polarization,
        CONFIG.use_canonical_polarization,
    )
    same_labels, same_data = _wc_relation_series(
        rows,
        same_pairs,
        "same_layer",
        CONFIG.same_layer_shell_id,
        CONFIG.use_unordered_pairs,
    )
    cross_labels, cross_data = _wc_relation_series(
        rows,
        cross_pairs,
        "cross_layer",
        CONFIG.cross_layer_shell_id,
        CONFIG.use_unordered_pairs,
    )

    pol_fig = _plot_polarization(temps, polarization)
    CONFIG.polarization_png.parent.mkdir(parents=True, exist_ok=True)
    pol_fig.savefig(CONFIG.polarization_png, bbox_inches="tight")
    pol_fig.savefig(CONFIG.polarization_pdf, bbox_inches="tight")
    plt.close(pol_fig)

    wc_fig = _plot_wc(temps, same_labels, same_data, cross_labels, cross_data)
    wc_fig.savefig(CONFIG.wc_png, bbox_inches="tight")
    wc_fig.savefig(CONFIG.wc_pdf, bbox_inches="tight")
    plt.close(wc_fig)

    print(f"Read summary: {CONFIG.summary_csv}")
    print(f"Detected species: {species}")
    print(f"Intralayer shell id: {CONFIG.same_layer_shell_id}")
    print(f"Interlayer shell id: {CONFIG.cross_layer_shell_id}")
    print(f"Plotted intralayer pairs: {[label for _, _, label in same_labels]}")
    print(f"Plotted interlayer pairs: {[label for _, _, label in cross_labels]}")
    print("Polarization mode:", "absolute" if CONFIG.use_absolute_polarization else "signed")
    print("Canonical polarization:", CONFIG.use_canonical_polarization)
    print(f"Wrote: {CONFIG.polarization_png}")
    print(f"Wrote: {CONFIG.polarization_pdf}")
    print(f"Wrote: {CONFIG.wc_png}")
    print(f"Wrote: {CONFIG.wc_pdf}")


if __name__ == "__main__":
    main()
