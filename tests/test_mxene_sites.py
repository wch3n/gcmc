import unittest
from collections import Counter

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


def _make_toy_terminated_mxene() -> Atoms:
    atoms = _make_toy_mxene(include_carbon=True, rumple_top=True)
    top_xy = atoms.positions[:4, :2].copy()

    top_o_z = np.array([10.25, 10.10, 9.95, 9.85], dtype=float)
    bot_o_z = np.array([3.35, 3.45, 3.25, 3.15], dtype=float)

    symbols = list(atoms.get_chemical_symbols())
    positions = [pos.copy() for pos in atoms.positions]
    for xy, z in zip(top_xy, top_o_z):
        symbols.append("O")
        positions.append(np.array([xy[0], xy[1], z], dtype=float))
    for xy, z in zip(top_xy, bot_o_z):
        symbols.append("O")
        positions.append(np.array([xy[0], xy[1], z], dtype=float))

    buried_top_o_z = np.array([6.35, 6.45, 6.25, 6.15], dtype=float)
    for xy, z in zip(top_xy, buried_top_o_z):
        symbols.append("O")
        positions.append(np.array([xy[0], xy[1], z], dtype=float))

    return Atoms(symbols=symbols, positions=positions, cell=atoms.cell, pbc=atoms.pbc)


def _make_offset_terminated_mxene(x_shift: float = 0.8) -> Atoms:
    atoms = _make_toy_terminated_mxene()
    positions = atoms.positions.copy()
    top_o_indices = [
        i for i, atom in enumerate(atoms)
        if atom.symbol == "O" and atom.position[2] > 9.0
    ]
    positions[top_o_indices, 0] += float(x_shift)
    atoms.positions[:] = positions
    return atoms


def _make_shifted_subsurface_mxene(x_shift: float = 0.7) -> Atoms:
    top_pos, cell2d = _make_triangular_layer(z=8.4)
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

    for xy in hollow_xy:
        symbols.append("C")
        positions.append(np.array([xy[0] + float(x_shift), xy[1], 6.5], dtype=float))

    for xy in hollow_xy:
        symbols.append("Ti")
        positions.append(np.array([xy[0] + float(x_shift), xy[1], 5.0], dtype=float))

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
        self.assertTrue(
            all(np.isfinite(float(row["suggested_z_A"])) for row in registry),
            "Site-local support fallback should keep hollow heights finite.",
        )

    def test_registry_selects_all_top_side_terminations_for_atop_sites(self):
        atoms = _make_toy_terminated_mxene()

        registry = build_surface_site_registry(
            atoms,
            site_elements=("O",),
            substrate_elements=("Ti", "C"),
            termination_elements=("O",),
            surface_side="top",
            site_types=("atop",),
            layer_tol=0.5,
            xy_tol=0.6,
            support_xy_tol=0.6,
            vertical_offset=1.0,
            min_termination_dist=0.0,
        )

        self.assertEqual(len(registry), 4)
        self.assertTrue(all(row["site_type"] == "atop" for row in registry))

    def test_registry_selects_all_top_side_metals_from_terminations(self):
        atoms = _make_toy_terminated_mxene()

        registry = build_surface_site_registry(
            atoms,
            site_elements=("Ti",),
            substrate_elements=("Ti", "C"),
            termination_elements=("O",),
            surface_side="top",
            site_types=("atop",),
            layer_tol=0.5,
            xy_tol=0.6,
            support_xy_tol=0.6,
            vertical_offset=1.0,
            min_termination_dist=0.0,
        )

        self.assertEqual(len(registry), 4)
        self.assertTrue(all(row["site_type"] == "atop" for row in registry))

    def test_registry_decouples_termination_matching_from_support_radius(self):
        atoms = _make_offset_terminated_mxene()

        registry = build_surface_site_registry(
            atoms,
            site_elements=("Ti",),
            substrate_elements=("Ti", "C"),
            termination_elements=("O",),
            surface_side="top",
            site_types=("atop",),
            layer_tol=0.5,
            xy_tol=0.6,
            support_xy_tol=0.6,
            termination_site_xy_tol=1.2,
            vertical_offset=1.0,
            min_termination_dist=0.0,
        )

        self.assertEqual(len(registry), 4)

    def test_registry_uses_robust_hollow_match_floor_for_mxenes(self):
        atoms = _make_shifted_subsurface_mxene()

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

        counts = Counter(row["site_type"] for row in registry)
        self.assertEqual(counts.get("atop"), 4)
        self.assertEqual(counts.get("fcc"), 8)
        self.assertEqual(counts.get("hcp"), 8)


if __name__ == "__main__":
    unittest.main()
