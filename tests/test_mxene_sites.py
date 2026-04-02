import unittest

import numpy as np
from ase import Atoms

from gcmc.utils import (
    build_surface_site_registry,
    classify_hollow_sites_on_surface,
    get_hollow_xy,
)


def _make_triangular_layer(nx: int = 2, ny: int = 2, z: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    a1 = np.array([3.0, 0.0])
    a2 = np.array([1.5, 2.598076211353316])
    xy = []
    for i in range(nx):
        for j in range(ny):
            xy.append(i * a1 + j * a2)
    cell = np.array(
        [
            nx * a1,
            ny * a2,
            np.array([0.0, 0.0]),
        ],
        dtype=float,
    )
    pos = np.column_stack((np.asarray(xy, dtype=float), np.full(len(xy), float(z))))
    return pos, cell


def _make_toy_mxene(include_carbon: bool, *, rumple_top: bool = True) -> Atoms:
    top_pos, cell2d = _make_triangular_layer(z=8.4)
    if rumple_top:
        # Deliberately rumple the exposed Ti layer so a small z tolerance would split it.
        top_pos[:, 2] = np.array([8.95, 8.10, 8.85, 8.05], dtype=float)

    cell = np.array(
        [
            [cell2d[0, 0], cell2d[0, 1], 0.0],
            [cell2d[1, 0], cell2d[1, 1], 0.0],
            [0.0, 0.0, 20.0],
        ],
        dtype=float,
    )
    hollow_xy = get_hollow_xy(top_pos[:, :2], cell)

    symbols = ["Ti"] * len(top_pos)
    positions = [pos.copy() for pos in top_pos]

    if include_carbon:
        for xy in hollow_xy:
            symbols.append("C")
            positions.append(np.array([xy[0], xy[1], 6.5], dtype=float))

    for xy in hollow_xy:
        symbols.append("Ti")
        positions.append(np.array([xy[0], xy[1], 5.0], dtype=float))

    return Atoms(symbols=symbols, positions=positions, cell=cell, pbc=[True, True, False])


class TestMxeneSiteRegistry(unittest.TestCase):
    def test_fcc_hollow_returns_empty_without_second_subsurface_layer(self):
        atoms = _make_toy_mxene(include_carbon=False, rumple_top=False)
        top_xy = atoms.positions[:4, :2]
        hollow_xy = get_hollow_xy(top_xy, atoms.cell.array)

        fcc = classify_hollow_sites_on_surface(
            atoms,
            hollow_xy,
            elements=("Ti",),
            stacking="fcc",
            layer_tol=0.5,
            surface_side="top",
        )
        hcp = classify_hollow_sites_on_surface(
            atoms,
            hollow_xy,
            elements=("Ti",),
            stacking="hcp",
            layer_tol=0.5,
            surface_side="top",
        )

        self.assertEqual(fcc.shape, (0, 2))
        self.assertGreater(len(hcp), 0)

    def test_registry_uses_substrate_layers_for_mxene_hollows(self):
        atoms = _make_toy_mxene(include_carbon=True, rumple_top=True)

        registry = build_surface_site_registry(
            atoms,
            site_elements=("Ti",),
            substrate_elements=("Ti", "C"),
            surface_side="top",
            site_types=("atop", "fcc", "hcp"),
            layer_tol=0.5,
            xy_tol=0.6,
            support_xy_tol=0.6,
            vertical_offset=1.0,
        )

        counts = {}
        for row in registry:
            counts[row["site_type"]] = counts.get(row["site_type"], 0) + 1

        self.assertEqual(counts.get("atop"), 4)
        self.assertGreater(counts.get("hcp", 0), 0)
        self.assertGreater(counts.get("fcc", 0), 0)


if __name__ == "__main__":
    unittest.main()
