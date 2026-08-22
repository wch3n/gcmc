"""Plot adsorption-motif populations from per-frame analysis rows."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D


_FAMILY_ORDER = ("atop", "bridge", "hollow", "multifold", "off_site", "detached")
_FAMILY_COLORS = {
    "atop": "#D55E00",
    "bridge": "#0072B2",
    "hollow": "#009E73",
    "multifold": "#E69F00",
    "off_site": "#8C8C8C",
    "detached": "#CC79A7",
    "other": "#B8B8B8",
}
_FAMILY_MARKERS = {
    "atop": "o",
    "bridge": "s",
    "hollow": "^",
    "multifold": "D",
    "off_site": "v",
    "detached": "P",
}
_FAMILY_LINESTYLES = {
    "atop": "-",
    "bridge": "--",
    "hollow": "-.",
    "multifold": (0, (3, 1, 1, 1)),
    "off_site": ":",
    "detached": ":",
}
_SUPPORT_CMAP = LinearSegmentedColormap.from_list(
    "Ti_Zr_support",
    ["#D55E00", "#7B6AA8", "#0072B2"],
)
_SUPPORT_NORM = Normalize(vmin=0.0, vmax=1.0)


def _family_sort_key(family: str) -> tuple[int, str]:
    try:
        return _FAMILY_ORDER.index(family), family
    except ValueError:
        return len(_FAMILY_ORDER), family


def _mean_sem(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    mean = float(np.mean(array))
    if array.size < 2:
        return mean, float("nan")
    return mean, float(np.std(array, ddof=1) / math.sqrt(array.size))


def _run_fraction_table(
    rows: Sequence[dict[str, object]],
    *,
    key_fields: Sequence[str],
) -> tuple[
    list[float],
    list[tuple[str, ...]],
    dict[tuple[float, str, tuple[str, ...]], float],
]:
    counts: dict[tuple[float, str, tuple[str, ...]], int] = defaultdict(int)
    totals: dict[tuple[float, str], int] = defaultdict(int)
    temperatures: set[float] = set()
    trajectories: dict[float, set[str]] = defaultdict(set)
    keys: set[tuple[str, ...]] = set()
    for row in rows:
        temperature = float(row["temperature_K"])
        if not np.isfinite(temperature):
            continue
        traj = str(row["traj"])
        key = tuple(str(row[field]) for field in key_fields)
        temperatures.add(temperature)
        trajectories[temperature].add(traj)
        keys.add(key)
        counts[(temperature, traj, key)] += 1
        totals[(temperature, traj)] += 1

    fractions: dict[tuple[float, str, tuple[str, ...]], float] = {}
    for temperature in temperatures:
        for traj in trajectories[temperature]:
            total = totals[(temperature, traj)]
            for key in keys:
                fractions[(temperature, traj, key)] = (
                    counts[(temperature, traj, key)] / total if total else 0.0
                )
    return sorted(temperatures), sorted(keys), fractions


def _pattern_label(key: tuple[str, ...]) -> str:
    family, site_type, support_key = key
    if family == "hollow":
        return f"{site_type} hollow, {support_key}"
    if family in {"off_site", "detached"}:
        return family.replace("_", " ")
    return f"{family}, {support_key}"


def _normalized_pattern_rows(
    rows: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    normalized_rows: list[dict[str, object]] = []
    for row in rows:
        normalized = dict(row)
        if str(row["adsorption_motif"]) in {"off_site", "detached"}:
            normalized["site_type"] = "none"
            normalized["support_key"] = "none"
        normalized_rows.append(normalized)
    return normalized_rows


def _support_fraction(
    key: tuple[str, ...],
    support_color_elements: tuple[str, str],
) -> float:
    if key[0] in {"off_site", "detached", "other"}:
        return float("nan")
    counts = {
        element: int(count)
        for element, count in re.findall(r"([A-Z][a-z]?)([0-9]+)", key[2])
    }
    first, second = support_color_elements
    total = counts.get(first, 0) + counts.get(second, 0)
    return float(counts.get(second, 0) / total) if total > 0 else float("nan")


def _support_color(
    key: tuple[str, ...],
    support_color_elements: tuple[str, str],
):
    fraction = _support_fraction(key, support_color_elements)
    if not np.isfinite(fraction):
        return _FAMILY_COLORS.get(key[0], "#777777")
    return _SUPPORT_CMAP(_SUPPORT_NORM(fraction))


def _plot_family_panel(
    ax,
    rows: Sequence[dict[str, object]],
) -> None:
    temperatures, keys, fractions = _run_fraction_table(
        rows,
        key_fields=("adsorption_motif",),
    )
    families = sorted((key[0] for key in keys), key=_family_sort_key)
    trajectories = {
        temperature: sorted(
            {str(row["traj"]) for row in rows if float(row["temperature_K"]) == temperature}
        )
        for temperature in temperatures
    }

    means: dict[str, list[float]] = defaultdict(list)
    sems: dict[str, list[float]] = defaultdict(list)
    for family in families:
        for temperature in temperatures:
            values = [
                fractions[(temperature, traj, (family,))]
                for traj in trajectories[temperature]
            ]
            mean, sem = _mean_sem(values)
            means[family].append(mean)
            sems[family].append(sem)

    if len(temperatures) == 1:
        positions = np.arange(len(families), dtype=float)
        values = [means[family][0] for family in families]
        errors = np.asarray([sems[family][0] for family in families], dtype=float)
        errors = None if np.all(~np.isfinite(errors)) else np.nan_to_num(errors)
        ax.bar(
            positions,
            values,
            yerr=errors,
            color=[_FAMILY_COLORS.get(family, "#666666") for family in families],
            edgecolor="black",
            linewidth=0.4,
            error_kw={"elinewidth": 0.7, "capsize": 2.0, "capthick": 0.7},
        )
        ax.set_xticks(positions, [family.replace("_", " ") for family in families])
        ax.tick_params(axis="x", rotation=30)
        ax.set_xlabel(f"Adsorption motif at {temperatures[0]:g} K")
    else:
        for family in families:
            errors = np.asarray(sems[family], dtype=float)
            yerr = None if np.all(~np.isfinite(errors)) else np.nan_to_num(errors)
            ax.errorbar(
                temperatures,
                means[family],
                yerr=yerr,
                label=family.replace("_", " "),
                color=_FAMILY_COLORS.get(family, "#666666"),
                marker=_FAMILY_MARKERS.get(family, "o"),
                linewidth=1.0,
                markersize=3.2,
                capsize=1.8,
                elinewidth=0.6,
            )
        ax.set_xlabel("MC temperature (K)")
        ax.legend(frameon=False, ncol=2, handlelength=1.6, columnspacing=0.9)
    ax.set_ylabel("Motif fraction")
    ax.set_ylim(0.0, 1.02)


def _family_support_fractions(
    rows: Sequence[dict[str, object]],
    *,
    families: Sequence[str],
    temperatures: Sequence[float],
    trajectories: dict[float, list[str]],
    family_fractions: dict[tuple[float, str, tuple[str, ...]], float],
    support_color_elements: tuple[str, str],
) -> dict[str, list[float]]:
    support_sums: dict[tuple[float, str, str], float] = defaultdict(float)
    support_counts: dict[tuple[float, str, str], int] = defaultdict(int)
    for row in rows:
        temperature = float(row["temperature_K"])
        if temperature not in trajectories:
            continue
        family = str(row["adsorption_motif"])
        key = (family, str(row["site_type"]), str(row["support_key"]))
        support_fraction = _support_fraction(key, support_color_elements)
        if not np.isfinite(support_fraction):
            continue
        run_key = (temperature, str(row["traj"]), family)
        support_sums[run_key] += support_fraction
        support_counts[run_key] += 1

    compositions: dict[str, list[float]] = defaultdict(list)
    for family in families:
        for temperature in temperatures:
            weighted_sum = 0.0
            population_sum = 0.0
            for traj in trajectories[temperature]:
                run_key = (temperature, traj, family)
                count = support_counts.get(run_key, 0)
                if count == 0:
                    continue
                population = family_fractions[(temperature, traj, (family,))]
                weighted_sum += population * support_sums[run_key] / count
                population_sum += population
            compositions[family].append(
                weighted_sum / population_sum
                if population_sum > 0.0
                else float("nan")
            )
    return compositions


def _plot_support_colored_family_temperature_panel(
    ax,
    rows: Sequence[dict[str, object]],
    *,
    support_color_elements: tuple[str, str],
) -> None:
    temperatures, keys, fractions = _run_fraction_table(
        rows,
        key_fields=("adsorption_motif",),
    )
    families = sorted((key[0] for key in keys), key=_family_sort_key)
    trajectories = {
        temperature: sorted(
            {
                str(row["traj"])
                for row in rows
                if float(row["temperature_K"]) == temperature
            }
        )
        for temperature in temperatures
    }
    means_by_family: dict[str, list[float]] = defaultdict(list)
    sems_by_family: dict[str, list[float]] = defaultdict(list)
    for family in families:
        means: list[float] = []
        sems: list[float] = []
        for temperature in temperatures:
            values = [
                fractions[(temperature, traj, (family,))]
                for traj in trajectories[temperature]
            ]
            mean, sem = _mean_sem(values)
            means.append(mean)
            sems.append(sem)
        means_by_family[family] = means
        sems_by_family[family] = sems

    compositions = _family_support_fractions(
        rows,
        families=families,
        temperatures=temperatures,
        trajectories=trajectories,
        family_fractions=fractions,
        support_color_elements=support_color_elements,
    )
    for family in families:
        family_means = means_by_family[family]
        family_sems = sems_by_family[family]
        family_compositions = compositions[family]
        linestyle = _FAMILY_LINESTYLES.get(family, "-")
        fallback_color = _FAMILY_COLORS.get(family, "#777777")
        for index in range(len(temperatures) - 1):
            segment_values = np.asarray(
                family_compositions[index : index + 2], dtype=float
            )
            segment_composition = (
                float(np.mean(segment_values[np.isfinite(segment_values)]))
                if np.any(np.isfinite(segment_values))
                else float("nan")
            )
            color = (
                _SUPPORT_CMAP(_SUPPORT_NORM(segment_composition))
                if np.isfinite(segment_composition)
                else fallback_color
            )
            ax.plot(
                temperatures[index : index + 2],
                family_means[index : index + 2],
                color=color,
                linestyle=linestyle,
                linewidth=1.1,
                zorder=2,
            )
        for temperature, mean, sem, composition in zip(
            temperatures,
            family_means,
            family_sems,
            family_compositions,
        ):
            color = (
                _SUPPORT_CMAP(_SUPPORT_NORM(composition))
                if np.isfinite(composition)
                else fallback_color
            )
            ax.errorbar(
                temperature,
                mean,
                yerr=None if not np.isfinite(sem) else sem,
                color=color,
                marker=_FAMILY_MARKERS.get(family, "o"),
                linestyle="none",
                linewidth=0.8,
                markersize=3.4,
                capsize=1.8,
                elinewidth=0.6,
                zorder=3,
            )
    ax.set_xlabel("MC temperature (K)")
    ax.set_ylabel("Motif fraction")
    ax.set_ylim(0.0, 1.02)
    handles = [
        Line2D(
            [0],
            [0],
            color="#555555",
            marker=_FAMILY_MARKERS.get(family, "o"),
            linestyle=_FAMILY_LINESTYLES.get(family, "-"),
            linewidth=1.0,
            markersize=3.4,
            label=family.replace("_", " "),
        )
        for family in families
    ]
    ax.legend(
        handles=handles,
        frameon=False,
        ncol=2,
        handlelength=1.6,
        columnspacing=0.9,
    )


def _plot_pattern_panel(
    ax,
    rows: Sequence[dict[str, object]],
    *,
    target_temperature_K: float,
    top_patterns: int,
    support_color_elements: tuple[str, str],
) -> float:
    pattern_rows = _normalized_pattern_rows(rows)
    temperatures, keys, fractions = _run_fraction_table(
        pattern_rows,
        key_fields=("adsorption_motif", "site_type", "support_key"),
    )
    target = min(temperatures, key=lambda value: abs(value - target_temperature_K))
    trajectories = sorted(
        {
            str(row["traj"])
            for row in pattern_rows
            if float(row["temperature_K"]) == target
        }
    )
    statistics: list[tuple[float, float, tuple[str, ...]]] = []
    for key in keys:
        values = [fractions[(target, traj, key)] for traj in trajectories]
        mean, sem = _mean_sem(values)
        statistics.append((mean, sem, key))
    statistics = [item for item in statistics if item[0] > 1.0e-12]
    statistics.sort(key=lambda item: (-item[0], item[2]))
    selected = statistics[:top_patterns]

    remainder_by_run = [
        max(
            0.0,
            1.0 - sum(fractions[(target, traj, key)] for _, _, key in selected),
        )
        for traj in trajectories
    ]
    remainder_mean, remainder_sem = _mean_sem(remainder_by_run)
    if remainder_mean > 1.0e-6:
        selected.append((remainder_mean, remainder_sem, ("other", "other", "other")))

    selected.reverse()
    y = np.arange(len(selected), dtype=float)
    means = [item[0] for item in selected]
    errors = np.asarray([item[1] for item in selected], dtype=float)
    xerr = None if np.all(~np.isfinite(errors)) else np.nan_to_num(errors)
    labels = [
        "other" if item[2][0] == "other" else _pattern_label(item[2])
        for item in selected
    ]
    colors = [_support_color(item[2], support_color_elements) for item in selected]
    ax.barh(
        y,
        means,
        xerr=xerr,
        color=colors,
        edgecolor="black",
        linewidth=0.4,
        error_kw={"elinewidth": 0.7, "capsize": 2.0, "capthick": 0.7},
    )
    ax.set_yticks(y, labels)
    ax.set_xlabel(f"Pattern fraction at {target:g} K")
    finite_errors = np.nan_to_num(errors, nan=0.0)
    upper = max(
        (mean + error for mean, error in zip(means, finite_errors)),
        default=0.0,
    )
    ax.set_xlim(0.0, max(0.05, min(1.0, upper * 1.08)))
    return target


def plot_adsorption_motif_distributions(
    frame_rows: Iterable[dict[str, object]],
    output_prefix: str | Path,
    *,
    target_temperature_K: float | None = None,
    top_patterns: int = 8,
    support_color_elements: Sequence[str] = ("Ti", "Zr"),
    color_temperature_by_support: bool = True,
    formats: Sequence[str] = ("pdf", "png"),
) -> list[Path]:
    """Plot family populations versus temperature and detailed low-T patterns.

    Fractions are first calculated within each trajectory and then averaged over
    trajectories at the same temperature. Error bars are the standard error over
    trajectories when at least two trajectories are available. For support-colored
    temperature curves, point color is the conditional mean support composition
    within each adsorption family.
    """
    rows = list(frame_rows)
    if not rows:
        raise ValueError("frame_rows must contain at least one motif assignment.")
    if top_patterns < 1:
        raise ValueError("top_patterns must be >= 1.")
    if len(support_color_elements) != 2:
        raise ValueError("support_color_elements must contain exactly two elements.")
    support_elements = tuple(str(element) for element in support_color_elements)
    temperatures = sorted(
        {
            float(row["temperature_K"])
            for row in rows
            if np.isfinite(float(row["temperature_K"]))
        }
    )
    if not temperatures:
        raise ValueError("No finite trajectory temperatures are available for plotting.")
    target = temperatures[0] if target_temperature_K is None else float(target_temperature_K)

    params = {
        "axes.linewidth": 0.6,
        "axes.labelsize": 9,
        "font.size": 9,
        "font.family": "sans-serif",
        "font.sans-serif": ["Nimbus Sans", "DejaVu Sans"],
        "legend.fontsize": 7,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "mathtext.default": "regular",
    }
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    with plt.rc_context(params):
        fig, (ax_family, ax_pattern) = plt.subplots(
            1,
            2,
            figsize=(16.0 / 2.54, 3.0),
        )
        fig.subplots_adjust(
            left=0.09,
            right=0.985,
            bottom=0.20,
            top=0.88,
            wspace=0.62,
        )
        if len(temperatures) > 1 and color_temperature_by_support:
            _plot_support_colored_family_temperature_panel(
                ax_family,
                rows,
                support_color_elements=support_elements,
            )
        else:
            _plot_family_panel(ax_family, rows)
        _plot_pattern_panel(
            ax_pattern,
            rows,
            target_temperature_K=target,
            top_patterns=top_patterns,
            support_color_elements=support_elements,
        )
        colorbar_axes = (
            [ax_family, ax_pattern]
            if len(temperatures) > 1 and color_temperature_by_support
            else ax_pattern
        )
        colorbar = fig.colorbar(
            ScalarMappable(norm=_SUPPORT_NORM, cmap=_SUPPORT_CMAP),
            ax=colorbar_axes,
            orientation="horizontal",
            location="top",
            fraction=0.035,
            pad=0.03,
            aspect=40,
        )
        colorbar.set_ticks(
            [0.0, 1.0],
            labels=[support_elements[0], support_elements[1]],
        )
        colorbar.set_label("Support composition", fontsize=8, labelpad=1.5)
        colorbar.ax.tick_params(labelsize=7, width=0.5, length=2)
        for axis in (ax_family, ax_pattern):
            axis.tick_params(which="both", direction="in", top=True, right=True)
        for output_format in formats:
            normalized = str(output_format).lower().lstrip(".")
            if normalized not in {"pdf", "png", "svg"}:
                raise ValueError("plot formats must be PDF, PNG, or SVG.")
            path = Path(f"{output_prefix}.{normalized}")
            fig.savefig(path, dpi=300, facecolor="white")
            written.append(path)
        plt.close(fig)
    return written
