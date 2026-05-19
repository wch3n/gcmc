"""Utilities for visualizing adsorption-site registries."""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Mapping, Sequence

import numpy as np
from ase import Atoms


DEFAULT_SITE_MARKER_SYMBOLS = {
    "atop": "He",
    "bridge": "Ne",
    "fcc": "Ar",
    "hcp": "Kr",
}


def build_site_marker_atoms(
    site_registry: Sequence[Mapping[str, object]],
    *,
    cell=None,
    pbc=None,
    include_blocked: bool = False,
    marker_symbols: Mapping[str, str] | None = None,
    z_field: str = "suggested_z_A",
) -> Atoms:
    """Return marker atoms for a site registry.

    Each site type is mapped to a marker element so the registry can be opened in
    ASE GUI, VESTA, OVITO, etc. Sites with non-finite marker coordinates are
    skipped.
    """

    symbols_map = dict(DEFAULT_SITE_MARKER_SYMBOLS)
    if marker_symbols is not None:
        symbols_map.update(marker_symbols)

    marker_symbols_out: list[str] = []
    marker_positions: list[list[float]] = []
    marker_tags: list[int] = []

    site_type_order = {name: i for i, name in enumerate(DEFAULT_SITE_MARKER_SYMBOLS, start=1)}
    next_tag = max(site_type_order.values(), default=0) + 1

    for row in site_registry:
        if not include_blocked and bool(row.get("blocked_by_termination", False)):
            continue

        z_value = float(row.get(z_field, np.nan))
        xy_value = row.get("xy", None)
        if xy_value is not None:
            xy_array = np.asarray(xy_value, dtype=float).reshape(-1)
            if xy_array.size >= 2:
                x_value = float(xy_array[0])
                y_value = float(xy_array[1])
            else:
                x_value = float(row.get("x_A", np.nan))
                y_value = float(row.get("y_A", np.nan))
        else:
            x_value = float(row.get("x_A", np.nan))
            y_value = float(row.get("y_A", np.nan))
        if not (np.isfinite(x_value) and np.isfinite(y_value) and np.isfinite(z_value)):
            continue

        site_type = str(row.get("site_type", "atop"))
        marker_symbols_out.append(symbols_map.get(site_type, "Xe"))
        marker_positions.append([x_value, y_value, z_value])
        marker_tags.append(site_type_order.get(site_type, next_tag))

    markers = Atoms(
        symbols=marker_symbols_out,
        positions=np.asarray(marker_positions, dtype=float) if marker_positions else np.zeros((0, 3), dtype=float),
        cell=cell,
        pbc=pbc,
    )
    if marker_tags:
        markers.set_tags(marker_tags)
    return markers


def overlay_site_markers(
    slab: Atoms,
    site_registry: Sequence[Mapping[str, object]],
    *,
    include_blocked: bool = False,
    marker_symbols: Mapping[str, str] | None = None,
    z_field: str = "suggested_z_A",
) -> Atoms:
    """Return a copy of ``slab`` with site markers appended."""

    overlay = slab.copy()
    markers = build_site_marker_atoms(
        site_registry,
        cell=overlay.cell.array,
        pbc=overlay.pbc,
        include_blocked=include_blocked,
        marker_symbols=marker_symbols,
        z_field=z_field,
    )
    overlay += markers
    return overlay


def summarize_site_registry(
    site_registry: Iterable[Mapping[str, object]],
    *,
    include_blocked: bool = False,
) -> dict[str, int]:
    """Count registry entries by site type."""

    counts: Counter[str] = Counter()
    for row in site_registry:
        if not include_blocked and bool(row.get("blocked_by_termination", False)):
            continue
        counts[str(row.get("site_type", "unknown"))] += 1
    return dict(counts)
