#!/usr/bin/env python3
"""Print a compact OER/CHE summary from a reaction workflow directory.

The script is intentionally read-only.  It consumes the standard files written
by ``gcmc-oer-workflow`` and reports the quantities that are usually
needed first: the free-energy profile, limiting step, overpotential, and the
largest per-site contributors.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    import yaml
except ImportError:  # pragma: no cover - PyYAML is a package dependency.
    yaml = None


KB_EV_PER_K = 8.617333262145e-5
STATE_DIRS = {
    "clean": "00_clean",
    "oh": "01_OH",
    "o": "02_O",
    "ooh": "03_OOH",
}
STEP_LABELS = {
    1: "* + H2O -> OH* + H+ + e-",
    2: "OH* -> O* + H+ + e-",
    3: "O* + H2O -> OOH* + H+ + e-",
    4: "OOH* -> * + O2 + H+ + e-",
}


@dataclass(frozen=True)
class RouteProfile:
    route_id: str
    site_id: str
    population: float
    weight: float | None
    deltas: tuple[float, float, float, float]
    limiting_step: int
    overpotential: float
    o_basin_id: str = ""
    o_basin_count: int | None = None
    ooh_basin_id: str = ""
    ooh_basin_count: int | None = None
    parent_state_dir: str = ""
    parent_state_candidate_id: str = ""
    parent_candidate_id: str = ""
    transition_builder: str = ""
    n_oh: int | None = None
    n_o: int | None = None
    n_ooh: int | None = None


@dataclass(frozen=True)
class EnsembleProfile:
    label: str
    deltas: tuple[float, float, float, float]
    mean_profile_limiting_step: int
    overpotential: float
    min_site_id: str
    min_site_overpotential: float
    dominant_site_id: str
    dominant_weight: float


@dataclass(frozen=True)
class WeightedStats:
    mean: float
    std: float
    q_low: float
    q_high: float


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _as_float(value: object, default: float = math.nan) -> float:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: object) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return None


def _is_true(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _finite(value: float) -> bool:
    return math.isfinite(value)


def _format_float(value: float, digits: int = 3) -> str:
    if not _finite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def _load_yaml(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    if yaml is None:
        raise RuntimeError("PyYAML is required to read reference summary YAML files.")
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        return {}
    return data


def _reference_energy(workflow: Path, name: str, field: str) -> float:
    summary = _load_yaml(workflow / "references" / name / "summary.yaml")
    value = _as_float(summary.get(field))
    if not _finite(value):
        raise ValueError(
            f"Could not read references/{name}/summary.yaml field {field!r}."
        )
    return value


def _optional_reference_energy(
    workflow: Path,
    name: str,
    field: str,
    override: float | None = None,
) -> float:
    if override is not None:
        return float(override)
    summary = _load_yaml(workflow / "references" / name / "summary.yaml")
    return _as_float(summary.get(field))


def _cluster_energies(values: Iterable[float], tol: float) -> list[float]:
    clusters: list[float] = []
    last: float | None = None
    for energy in sorted(value for value in values if _finite(value)):
        if last is None or abs(energy - last) > tol:
            clusters.append(energy)
            last = energy
    return clusters


def _boltzmann_free_energy(values: Iterable[float], temperature: float) -> float:
    energies = list(values)
    if not energies:
        return math.nan
    kbt = KB_EV_PER_K * temperature
    minimum = min(energies)
    partition = sum(math.exp(-(energy - minimum) / kbt) for energy in energies)
    return minimum - kbt * math.log(partition)


def _limiting_step(deltas: tuple[float, float, float, float]) -> int:
    return max(range(4), key=lambda idx: deltas[idx]) + 1


def _route_from_deltas(
    site_id: str,
    population: float,
    deltas: tuple[float, float, float, float],
    equilibrium: float,
    route_id: str = "",
    weight: float | None = None,
    o_basin_id: str = "",
    o_basin_count: int | None = None,
    ooh_basin_id: str = "",
    ooh_basin_count: int | None = None,
    parent_state_dir: str = "",
    parent_state_candidate_id: str = "",
    parent_candidate_id: str = "",
    transition_builder: str = "",
    n_oh: int | None = None,
    n_o: int | None = None,
    n_ooh: int | None = None,
) -> RouteProfile:
    limiting = _limiting_step(deltas)
    return RouteProfile(
        route_id=route_id or site_id,
        site_id=site_id,
        population=population,
        weight=weight,
        o_basin_id=o_basin_id,
        o_basin_count=o_basin_count,
        ooh_basin_id=ooh_basin_id,
        ooh_basin_count=ooh_basin_count,
        parent_state_dir=parent_state_dir,
        parent_state_candidate_id=parent_state_candidate_id,
        parent_candidate_id=parent_candidate_id,
        transition_builder=transition_builder,
        deltas=deltas,
        limiting_step=limiting,
        overpotential=max(deltas) - equilibrium,
        n_oh=n_oh,
        n_o=n_o,
        n_ooh=n_ooh,
    )


def _route_with_closed_dg4(
    route: RouteProfile,
    *,
    total_oer_free_energy: float,
    equilibrium: float,
) -> RouteProfile:
    dg1, dg2, dg3, _ = route.deltas
    return _route_from_deltas(
        site_id=route.site_id,
        population=route.population,
        route_id=route.route_id,
        weight=route.weight,
        o_basin_id=route.o_basin_id,
        o_basin_count=route.o_basin_count,
        ooh_basin_id=route.ooh_basin_id,
        ooh_basin_count=route.ooh_basin_count,
        parent_state_dir=route.parent_state_dir,
        parent_state_candidate_id=route.parent_state_candidate_id,
        parent_candidate_id=route.parent_candidate_id,
        transition_builder=route.transition_builder,
        deltas=(dg1, dg2, dg3, total_oer_free_energy - dg1 - dg2 - dg3),
        equilibrium=equilibrium,
        n_oh=route.n_oh,
        n_o=route.n_o,
        n_ooh=route.n_ooh,
    )


def close_routes_to_total_oer_free_energy(
    routes: list[RouteProfile],
    *,
    total_oer_free_energy: float,
    equilibrium: float,
) -> list[RouteProfile]:
    """Apply CHE closure using the route's active DeltaG1-3 values."""

    return [
        _route_with_closed_dg4(
            route,
            total_oer_free_energy=total_oer_free_energy,
            equilibrium=equilibrium,
        )
        for route in routes
    ]


def load_harmonic_routes(workflow: Path, equilibrium: float) -> list[RouteProfile]:
    """Read the CHE routes produced by the main post-process workflow."""

    routes = []
    for row in _read_csv(workflow / "oer_routes.csv"):
        if _is_true(row.get("ready")):
            deltas = tuple(_as_float(row.get(f"DeltaG{idx}_eV")) for idx in range(1, 5))
            if len(deltas) != 4 or not all(_finite(value) for value in deltas):
                continue
            routes.append(
                _route_from_deltas(
                    site_id=str(row.get("site_id", "")),
                    population=_as_float(row.get("population_total"), 0.0),
                    weight=_as_float(row.get("route_weight")),
                    route_id=str(row.get("route_id", "")),
                    o_basin_id=str(row.get("o_basin_id", "")),
                    o_basin_count=_as_int(row.get("o_basin_count")),
                    ooh_basin_id=str(row.get("ooh_basin_id", "")),
                    ooh_basin_count=_as_int(row.get("ooh_basin_count")),
                    parent_state_dir=str(row.get("parent_state_dir", "")),
                    parent_state_candidate_id=str(
                        row.get("parent_state_candidate_id", "")
                        or row.get("parent_o_candidate_id", "")
                    ),
                    parent_candidate_id=str(row.get("parent_candidate_id", "")),
                    transition_builder=str(row.get("transition_builder", "")),
                    deltas=deltas,  # type: ignore[arg-type]
                    equilibrium=equilibrium,
                    n_oh=_as_int(row.get("n_OH_candidates")),
                    n_o=_as_int(row.get("n_O_candidates")),
                    n_ooh=_as_int(row.get("n_OOH_candidates")),
                )
            )
            continue

        use_boltzmann = _is_true(row.get("ready_boltzmann"))
        prefix = "boltzmann" if use_boltzmann else "min"
        ready_key = "ready_boltzmann" if use_boltzmann else "ready_min"
        if not _is_true(row.get(ready_key)):
            continue
        deltas = tuple(
            _as_float(row.get(f"DeltaG{idx}_{prefix}_eV"))
            for idx in range(1, 5)
        )
        if len(deltas) != 4 or not all(_finite(value) for value in deltas):
            continue
        routes.append(
            _route_from_deltas(
                site_id=str(row.get("site_id", "")),
                population=_as_float(row.get("population_total"), 0.0),
                weight=None,
                route_id=str(row.get("site_id", "")),
                deltas=deltas,  # type: ignore[arg-type]
                equilibrium=equilibrium,
                n_oh=_as_int(row.get("n_OH_candidates")),
                n_o=_as_int(row.get("n_O_candidates")),
                n_ooh=_as_int(row.get("n_OOH_candidates")),
            )
        )
    return routes


def _state_source_sites(workflow: Path) -> dict[tuple[str, str], str]:
    sources: dict[tuple[str, str], str] = {}
    for row in _read_csv(workflow / "oer_states.csv"):
        site_id = str(row.get("site_id", ""))
        label = str(row.get("state_label", "")).lower()
        source = str(row.get("source_site_id", site_id) or site_id)
        if site_id and label:
            sources[(site_id, label)] = source
    return sources


def _electronic_energies_from_vibrations(
    workflow: Path,
) -> dict[tuple[str, str], list[float]]:
    by_state: dict[tuple[str, str], list[float]] = {}
    for row in _read_csv(workflow / "vibration_summary.csv"):
        if not _is_true(row.get("ready")):
            continue
        energy = _as_float(row.get("potential_energy_eV"))
        if not _finite(energy):
            continue
        key = (str(row.get("site_id", "")), str(row.get("state_dir", "")))
        by_state.setdefault(key, []).append(energy)
    return by_state


def _electronic_energies_from_states(
    workflow: Path,
) -> dict[tuple[str, str], list[float]]:
    by_state: dict[tuple[str, str], list[float]] = {}
    for row in _read_csv(workflow / "oer_states.csv"):
        if not _is_true(row.get("ready")):
            continue
        label = str(row.get("state_label", "")).lower()
        state_dir = STATE_DIRS.get(label)
        if state_dir is None:
            continue
        energy = _as_float(row.get("electronic_energy_eV"))
        if not _finite(energy):
            continue
        site_id = str(row.get("source_site_id") if label == "clean" else row.get("site_id"))
        by_state.setdefault((site_id, state_dir), []).append(energy)
    return by_state


def load_electronic_routes(
    workflow: Path,
    harmonic_routes: list[RouteProfile],
    *,
    temperature: float,
    cluster_tol: float,
    total_oer_free_energy: float,
    equilibrium: float,
    o2_energy: float | None = None,
    oer_reference_mode: str = "auto",
) -> list[RouteProfile]:
    """Reconstruct an electronic-only CHE profile from the workflow outputs."""

    direct = _load_electronic_routes_from_oer_routes(workflow, equilibrium)
    if direct:
        return direct

    h2 = _reference_energy(workflow, "h2", "potential_energy_eV")
    h2o = _reference_energy(workflow, "h2o", "potential_energy_eV")
    o2 = _optional_reference_energy(workflow, "o2", "potential_energy_eV", o2_energy)
    mode = _resolve_oer_reference_mode(oer_reference_mode, o2)
    state_sources = _state_source_sites(workflow)
    energies = _electronic_energies_from_vibrations(workflow)
    if not energies:
        energies = _electronic_energies_from_states(workflow)

    def state_energy(site_id: str, label: str) -> float:
        state_dir = STATE_DIRS[label]
        source_site = state_sources.get((site_id, label), site_id)
        if label != "clean":
            source_site = site_id
        values = energies.get((source_site, state_dir), [])
        clustered = _cluster_energies(values, cluster_tol)
        return _boltzmann_free_energy(clustered, temperature)

    routes: list[RouteProfile] = []
    for harmonic in harmonic_routes:
        clean = state_energy(harmonic.site_id, "clean")
        oh = state_energy(harmonic.site_id, "oh")
        o = state_energy(harmonic.site_id, "o")
        ooh = state_energy(harmonic.site_id, "ooh")
        if not all(_finite(value) for value in (clean, oh, o, ooh)):
            continue
        dg1 = oh - clean - h2o + 0.5 * h2
        dg2 = o - oh + 0.5 * h2
        dg3 = ooh - o - h2o + 0.5 * h2
        if mode == "explicit_o2":
            dg4 = clean + o2 - ooh + 0.5 * h2
        else:
            dg4 = total_oer_free_energy - dg1 - dg2 - dg3
        routes.append(
            _route_from_deltas(
                harmonic.site_id,
                harmonic.population,
                (dg1, dg2, dg3, dg4),
                equilibrium,
                harmonic.n_oh,
                harmonic.n_o,
                harmonic.n_ooh,
            )
        )
    return routes


def _load_electronic_routes_from_oer_routes(
    workflow: Path,
    equilibrium: float,
) -> list[RouteProfile]:
    routes: list[RouteProfile] = []
    for row in _read_csv(workflow / "oer_routes.csv"):
        use_boltzmann = _is_true(row.get("ready_boltzmann_electronic"))
        prefix = "boltzmann" if use_boltzmann else "min"
        ready_key = f"ready_{prefix}_electronic"
        if not _is_true(row.get(ready_key)):
            continue
        deltas = tuple(
            _as_float(row.get(f"DeltaG{idx}_{prefix}_electronic_eV"))
            for idx in range(1, 5)
        )
        if len(deltas) != 4 or not all(_finite(value) for value in deltas):
            continue
        routes.append(
            _route_from_deltas(
                site_id=str(row.get("site_id", "")),
                population=_as_float(row.get("population_total"), 0.0),
                deltas=deltas,  # type: ignore[arg-type]
                equilibrium=equilibrium,
                n_oh=_as_int(row.get("n_OH_candidates")),
                n_o=_as_int(row.get("n_O_candidates")),
                n_ooh=_as_int(row.get("n_OOH_candidates")),
            )
        )
    return routes


def _resolve_oer_reference_mode(mode: str, o2_energy: float) -> str:
    raw = str(mode).strip().lower()
    if raw == "auto":
        return "explicit_o2" if _finite(o2_energy) else "closure"
    if raw in {"closure", "total", "total_oer", "4.92"}:
        return "closure"
    if raw in {"explicit", "explicit_o2", "o2"}:
        if not _finite(o2_energy):
            raise ValueError(
                "Explicit O2 mode requested, but no O2 electronic reference was found. "
                "Use --o2-energy or provide references/o2/summary.yaml."
            )
        return "explicit_o2"
    raise ValueError("--oer-reference-mode must be auto, closure, or explicit_o2.")


def route_probability_weights(
    routes: list[RouteProfile],
    temperature: float,
) -> list[float]:
    """Route probabilities used for summaries.

    Current workflow outputs carry explicit empirical route probabilities in
    ``route_weight``.  The temperature-dependent fallback is only for older
    outputs that predate explicit route weights.
    """

    if not routes:
        return []
    explicit = [
        route.weight
        for route in routes
        if route.weight is not None and _finite(route.weight) and route.weight >= 0.0
    ]
    if len(explicit) == len(routes) and sum(explicit) > 0.0:
        total = sum(explicit)
        return [float(weight) / total for weight in explicit]
    kbt = KB_EV_PER_K * temperature
    minimum = min(route.deltas[0] for route in routes)
    raw = [math.exp(-(route.deltas[0] - minimum) / kbt) for route in routes]
    total = sum(raw)
    if total <= 0.0:
        return [1.0 / len(routes)] * len(routes)
    return [value / total for value in raw]


def _weighted_stats(
    values: Iterable[float],
    weights: Iterable[float],
    *,
    interval: float,
) -> WeightedStats:
    pairs = [
        (value, weight)
        for value, weight in zip(values, weights)
        if _finite(value) and _finite(weight) and weight > 0.0
    ]
    if not pairs:
        return WeightedStats(math.nan, math.nan, math.nan, math.nan)
    weight_sum = sum(weight for _, weight in pairs)
    if weight_sum <= 0.0:
        return WeightedStats(math.nan, math.nan, math.nan, math.nan)
    normalized = [(value, weight / weight_sum) for value, weight in pairs]
    mean = sum(value * weight for value, weight in normalized)
    variance = sum(weight * (value - mean) ** 2 for value, weight in normalized)
    low_probability = max(0.0, min(1.0, 0.5 * (1.0 - interval)))
    high_probability = max(0.0, min(1.0, 1.0 - low_probability))
    return WeightedStats(
        mean=mean,
        std=math.sqrt(max(0.0, variance)),
        q_low=_weighted_quantile(normalized, low_probability),
        q_high=_weighted_quantile(normalized, high_probability),
    )


def _weighted_quantile(
    normalized_pairs: Iterable[tuple[float, float]],
    probability: float,
) -> float:
    cumulative = 0.0
    last_value = math.nan
    for value, weight in sorted(normalized_pairs, key=lambda item: item[0]):
        cumulative += weight
        last_value = value
        if cumulative >= probability:
            return value
    return last_value


def ensemble_profile(
    label: str,
    routes: list[RouteProfile],
    *,
    temperature: float,
    equilibrium: float,
) -> EnsembleProfile | None:
    if not routes:
        return None
    weights = route_probability_weights(routes, temperature)
    deltas = tuple(
        sum(weight * route.deltas[idx] for weight, route in zip(weights, routes))
        for idx in range(4)
    )
    mean_profile_limiting = _limiting_step(deltas)  # type: ignore[arg-type]
    route_weighted_eta = sum(
        weight * route.overpotential for weight, route in zip(weights, routes)
    )
    min_site = min(routes, key=lambda route: route.overpotential)
    dominant_idx = max(range(len(weights)), key=lambda idx: weights[idx])
    return EnsembleProfile(
        label=label,
        deltas=deltas,  # type: ignore[arg-type]
        mean_profile_limiting_step=mean_profile_limiting,
        overpotential=route_weighted_eta,
        min_site_id=min_site.site_id,
        min_site_overpotential=min_site.overpotential,
        dominant_site_id=routes[dominant_idx].site_id,
        dominant_weight=weights[dominant_idx],
    )


def route_distribution_stats(
    routes: list[RouteProfile],
    *,
    temperature: float,
    interval: float,
) -> dict[str, WeightedStats]:
    weights = route_probability_weights(routes, temperature)
    stats: dict[str, WeightedStats] = {}
    for idx in range(4):
        stats[f"DG{idx + 1}"] = _weighted_stats(
            (route.deltas[idx] for route in routes),
            weights,
            interval=interval,
        )
    stats["eta"] = _weighted_stats(
        (route.overpotential for route in routes),
        weights,
        interval=interval,
    )
    return stats


def _table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [
        max(len(headers[idx]), *(len(row[idx]) for row in rows))
        for idx in range(len(headers))
    ]
    header = " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(headers))
    divider = " | ".join("-" * width for width in widths)
    body = [
        " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(row))
        for row in rows
    ]
    return "\n".join([header, divider, *body])


def _profile_row(label: str, profile: EnsembleProfile) -> list[str]:
    return [
        label,
        _format_float(profile.deltas[0]),
        _format_float(profile.deltas[1]),
        _format_float(profile.deltas[2]),
        _format_float(profile.deltas[3]),
        (
            f"{profile.mean_profile_limiting_step}: "
            f"{STEP_LABELS[profile.mean_profile_limiting_step]}"
        ),
        _format_float(profile.overpotential),
    ]


def _site_rows(routes: list[RouteProfile], weights: list[float], top: int) -> list[list[str]]:
    rows: list[list[str]] = []
    pairs = sorted(zip(routes, weights), key=lambda item: item[1], reverse=True)
    for route, weight in pairs[:top]:
        rows.append(
            [
                route.route_id,
                route.site_id,
                route.parent_state_candidate_id,
                route.transition_builder,
                _format_float(route.population, 4),
                _format_float(weight, 4),
                _format_float(route.deltas[0]),
                _format_float(route.deltas[1]),
                _format_float(route.deltas[2]),
                _format_float(route.deltas[3]),
                str(route.limiting_step),
                _format_float(route.overpotential),
                "/".join(
                    str(value) if value is not None else "?"
                    for value in (route.n_oh, route.n_o, route.n_ooh)
                ),
            ]
        )
    return rows


def _uncertainty_rows(
    label: str,
    routes: list[RouteProfile],
    *,
    temperature: float,
    interval: float,
) -> list[list[str]]:
    stats_by_quantity = route_distribution_stats(
        routes,
        temperature=temperature,
        interval=interval,
    )
    rows: list[list[str]] = []
    for quantity in ("DG1", "DG2", "DG3", "DG4", "eta"):
        stats = stats_by_quantity[quantity]
        rows.append(
            [
                label,
                quantity,
                _format_float(stats.mean),
                _format_float(stats.std),
                _format_float(stats.q_low),
                _format_float(stats.q_high),
            ]
        )
    return rows


def _basin_rows(routes: list[RouteProfile], basin: str) -> list[list[str]]:
    seen: dict[tuple[str, str], tuple[int | None, float]] = {}
    for route in routes:
        if basin == "O":
            basin_id = route.o_basin_id
            count = route.o_basin_count
        else:
            basin_id = route.ooh_basin_id
            count = route.ooh_basin_count
        if not basin_id:
            continue
        key = (route.site_id, basin_id)
        seen[key] = (count, seen.get(key, (None, 0.0))[1] + (route.weight or 0.0))
    rows: list[list[str]] = []
    for (site_id, basin_id), (count, weight_sum) in sorted(
        seen.items(),
        key=lambda item: (-item[1][1], item[0][0], item[0][1]),
    ):
        rows.append(
            [
                basin,
                site_id,
                basin_id,
                str(count) if count is not None else "?",
                _format_float(weight_sum, 4),
            ]
        )
    return rows


def render_summary(
    workflow: Path,
    *,
    top_sites: int,
    temperature: float,
    cluster_tol: float,
    total_oer_free_energy: float,
    equilibrium: float,
    uncertainty_interval: float,
    o2_energy: float | None = None,
    oer_reference_mode: str = "auto",
) -> str:
    o2_reference = _optional_reference_energy(
        workflow,
        "o2",
        "potential_energy_eV",
        o2_energy,
    )
    resolved_oer_mode = _resolve_oer_reference_mode(oer_reference_mode, o2_reference)
    harmonic_routes = load_harmonic_routes(workflow, equilibrium)
    if not harmonic_routes:
        raise ValueError(f"No ready OER routes found in {workflow / 'oer_routes.csv'}")
    if resolved_oer_mode == "closure":
        harmonic_routes = close_routes_to_total_oer_free_energy(
            harmonic_routes,
            total_oer_free_energy=total_oer_free_energy,
            equilibrium=equilibrium,
        )

    harmonic_ensemble = ensemble_profile(
        "harmonic",
        harmonic_routes,
        temperature=temperature,
        equilibrium=equilibrium,
    )

    lines = [
        f"# OER Summary: {workflow}",
        "",
        "## Ensemble Profile",
        "",
        (
            "`route_weight` is an empirical conditional route probability from "
            "`oer_routes.csv`; older outputs fall back to OH Boltzmann "
            "adsorption weights reconstructed from DeltaG1."
        ),
        (
            "The reported ensemble eta is the route-probability-weighted mean "
            "of the route-wise overpotentials, not max(<DeltaG1>, ..., <DeltaG4>)."
        ),
        (
            "DeltaG4 uses "
            f"{'explicit O2' if resolved_oer_mode == 'explicit_o2' else 'closure to total_oer_free_energy'} "
            "for the harmonic profile."
        ),
        "",
    ]

    profile_rows = []
    if harmonic_ensemble is not None:
        profile_rows.append(_profile_row("harmonic", harmonic_ensemble))
    lines.append(
        _table(
            [
                "model",
                "DG1/eV",
                "DG2/eV",
                "DG3/eV",
                "DG4/eV",
                "mean-profile max step",
                "route-avg eta/V",
            ],
            profile_rows,
        )
    )

    uncertainty_rows: list[list[str]] = []
    if harmonic_routes:
        uncertainty_rows.extend(
            _uncertainty_rows(
                "harmonic",
                harmonic_routes,
                temperature=temperature,
                interval=uncertainty_interval,
            )
        )
    if uncertainty_rows:
        interval_pct = 100.0 * uncertainty_interval
        lines.extend(
            [
                "",
                "## Route-Distribution Spread",
                "",
                (
                    "This is the weighted spread over site-conditioned routes, "
                    "not a formal electronic-structure or MLIP error bar."
                ),
                "",
                _table(
                    [
                        "model",
                        "quantity",
                        "prob. mean",
                        "site sigma",
                        f"q{0.5 * (100.0 - interval_pct):.1f}",
                        f"q{100.0 - 0.5 * (100.0 - interval_pct):.1f}",
                    ],
                    uncertainty_rows,
                ),
            ]
        )

    basin_rows = _basin_rows(harmonic_routes, "O") + _basin_rows(harmonic_routes, "OOH")
    if basin_rows:
        lines.extend(
            [
                "",
                "## Basin Provenance",
                "",
                (
                    "Raw O*/OOH* candidates are clustered into relaxed geometry basins "
                    "before CHE route construction."
                ),
                (
                    "Basin and route probabilities are empirical retained-sample frequencies; "
                    "they are not guaranteed to be unbiased thermodynamic probabilities "
                    "unless local CMC/PT sampling is well equilibrated and well mixed."
                ),
                "",
                _table(
                    ["state", "site", "basin", "raw samples", "route prob. sum"],
                    basin_rows,
                ),
            ]
        )

    if harmonic_ensemble is not None:
        lines.extend(
            [
                "",
                "## Harmonic Site Routes",
                "",
                (
                    f"Best site eta: {harmonic_ensemble.min_site_id} "
                    f"({_format_float(harmonic_ensemble.min_site_overpotential)} V); "
                    f"dominant route-probability site: {harmonic_ensemble.dominant_site_id} "
                    f"(prob. {_format_float(harmonic_ensemble.dominant_weight, 4)})."
                ),
                "",
                _table(
                    [
                        "route",
                        "site",
                        "parent candidate",
                        "transition",
                        "pop",
                        "route prob.",
                        "DG1",
                        "DG2",
                        "DG3",
                        "DG4",
                        "lim",
                        "eta",
                        "n OH/O/OOH",
                    ],
                    _site_rows(
                        harmonic_routes,
                        route_probability_weights(harmonic_routes, temperature),
                        top_sites,
                    ),
                ),
            ]
        )

    ensemble_csv = workflow / "oer_ensemble.csv"
    if ensemble_csv.exists():
        rows = _read_csv(ensemble_csv)
        if rows:
            row = rows[0]
            lines.extend(
                [
                    "",
                    "## Workflow Ensemble Metadata",
                    "",
                    _table(
                        ["field", "value"],
                        [
                            ["route_weight_model", str(row.get("route_weight_model", ""))],
                            ["n_ready_routes", str(row.get("n_ready_routes", ""))],
                            [
                                "population_sum_ready",
                                str(row.get("population_sum_ready", "")),
                            ],
                            [
                                "population_missing",
                                str(row.get("population_missing", "")),
                            ],
                            [
                                "dominant_route_id",
                                str(row.get("dominant_route_id", "")),
                            ],
                            [
                                "dominant_route_weight",
                                str(row.get("dominant_route_weight", "")),
                            ],
                        ],
                    ),
                ]
            )

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workflow",
        type=Path,
        required=True,
        help="Path to a gcmc-oer-workflow output directory.",
    )
    parser.add_argument(
        "--top-sites",
        type=int,
        default=10,
        help="Number of site routes to print in the detailed tables.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=298.0,
        help=(
            "Temperature in K for legacy OH Boltzmann fallback weights when "
            "oer_routes.csv lacks explicit route_weight values."
        ),
    )
    parser.add_argument(
        "--cluster-tol",
        type=float,
        default=0.01,
        help="Energy tolerance in eV for clustering candidate basins.",
    )
    parser.add_argument(
        "--total-oer-free-energy",
        type=float,
        default=4.92,
        help="Total four-electron OER Gibbs free energy in eV for closure mode.",
    )
    parser.add_argument(
        "--equilibrium-potential",
        type=float,
        default=1.23,
        help="Equilibrium OER potential in V.",
    )
    parser.add_argument(
        "--oer-reference-mode",
        choices=["auto", "closure", "explicit_o2"],
        default="auto",
        help="How to evaluate DeltaG4 in the reported OER profiles.",
    )
    parser.add_argument(
        "--o2-energy",
        type=float,
        default=None,
        help="Optional O2 electronic energy in eV for explicit-O2 reconstruction.",
    )
    parser.add_argument(
        "--uncertainty-interval",
        type=float,
        default=0.95,
        help="Central weighted interval to report for route-distribution spread.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional path for a Markdown summary. Default: print to stdout.",
    )
    args = parser.parse_args()

    summary = render_summary(
        args.workflow,
        top_sites=args.top_sites,
        temperature=args.temperature,
        cluster_tol=args.cluster_tol,
        total_oer_free_energy=args.total_oer_free_energy,
        equilibrium=args.equilibrium_potential,
        uncertainty_interval=args.uncertainty_interval,
        o2_energy=args.o2_energy,
        oer_reference_mode=args.oer_reference_mode,
    )
    if args.out is None:
        print(summary, end="")
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(summary)
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
