import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

from ase import Atoms
from ase.geometry import get_distances


@dataclass
class MoveProposal:
    atoms: Atoms
    move_name: str
    is_md: bool = False
    delta_e: Optional[float] = None
    delta_h: Optional[float] = None
    metadata: Dict[str, object] = field(default_factory=dict)


def _rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm <= 0.0:
        return np.eye(3)
    axis = axis / norm
    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    one_c = 1.0 - c
    return np.array(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=float,
    )


class AdsorbateMoveProposalMixin:
    def _xy_displacements(
        self,
        point_xy: np.ndarray,
        other_xy: np.ndarray,
        atoms: Optional[Atoms] = None,
    ) -> np.ndarray:
        if atoms is None:
            atoms = self.atoms

        point_xy = np.asarray(point_xy, dtype=float).reshape(1, 2)
        other_xy = np.asarray(other_xy, dtype=float).reshape(-1, 2)
        deltas = other_xy - point_xy

        pbc = np.asarray(atoms.get_pbc(), dtype=bool)[:2]
        if not np.any(pbc) or deltas.size == 0:
            return deltas

        xy_cell = np.asarray(atoms.get_cell(), dtype=float)[:2, :2]
        try:
            frac = np.linalg.solve(xy_cell.T, deltas.T).T
        except np.linalg.LinAlgError:
            return deltas

        frac[:, pbc] -= np.rint(frac[:, pbc])
        return frac @ xy_cell

    def _xy_distances(
        self,
        point_xy: np.ndarray,
        other_xy: np.ndarray,
        atoms: Optional[Atoms] = None,
    ) -> np.ndarray:
        return np.linalg.norm(
            self._xy_displacements(point_xy, other_xy, atoms=atoms),
            axis=1,
        )

    def _trial_atoms_with_group(
        self,
        group: np.ndarray,
        trial_positions: np.ndarray,
        atoms: Optional[Atoms] = None,
    ) -> Atoms:
        if atoms is None:
            atoms = self.atoms
        atoms_trial = atoms.copy()
        atoms_trial.positions[np.asarray(group, dtype=int)] = np.asarray(
            trial_positions,
            dtype=float,
        )
        return atoms_trial

    def _rotate_group_about_anchor(
        self,
        group: np.ndarray,
        axis: np.ndarray,
        angle: float,
        atoms: Optional[Atoms] = None,
    ) -> np.ndarray:
        if atoms is None:
            atoms = self.atoms

        group = np.asarray(group, dtype=int)
        anchor_pos = self._group_anchor_position(group, atoms=atoms)
        relative = self._molecular_move_relative_positions(group, atoms=atoms)
        rotation = _rotation_matrix(axis, angle)
        return anchor_pos + relative @ rotation.T

    def _proposal_for_mode(self, mode: str) -> Callable[[], Optional[Atoms]]:
        proposals: dict[str, Callable[[], Optional[Atoms]]] = {
            "displacement": self._propose_displacement,
            "site_hop": self._propose_site_hop,
            "reorientation": self._propose_reorientation,
            "hop_reorientation": self._propose_hop_reorientation,
            "hop_puckering": self._propose_hop_puckering,
            "hop_puckering_reorientation": self._propose_hop_puckering_reorientation,
            "puckering": self._propose_puckering,
        }
        try:
            return proposals[mode]
        except KeyError as exc:
            raise ValueError(f"Unknown adsorbate move mode: {mode!r}") from exc

    def _build_hybrid_move_table(
        self,
    ) -> list[tuple[float, str, Callable[[], Optional[Atoms]]]]:
        weighted_modes = (
            (self.site_hop_prob, "site_hop"),
            (self.reorientation_prob, "reorientation"),
            (self.hop_reorientation_prob, "hop_reorientation"),
            (self.hop_puckering_prob, "hop_puckering"),
            (
                self.hop_puckering_reorientation_prob,
                "hop_puckering_reorientation",
            ),
            (self.puckering_prob, "puckering"),
        )
        table = [
            (float(weight), mode, self._proposal_for_mode(mode))
            for weight, mode in weighted_modes
            if float(weight) > 0.0
        ]
        residual = 1.0 - sum(weight for weight, _, _ in table)
        if residual > 1e-12:
            table.append((residual, "displacement", self._propose_displacement))
        return table

    def _propose_displacement(self) -> Optional[Atoms]:
        movable_group_ids = self.get_non_buried_adsorbate_indices(
            support_xy_tol=self.support_xy_tol
        )
        if not movable_group_ids:
            return None

        group_id = int(self.rng.choice(movable_group_ids))
        group = np.asarray(self.ads_groups[group_id], dtype=int)
        anchor_pos = self._group_anchor_position(group)
        relative = self._current_group_relative_positions(group)
        all_pos = self.atoms.get_positions()
        cell = self.atoms.get_cell()
        pbc = self.atoms.get_pbc()
        xy_matrix = cell[:2, :2]
        surface_indices = self._surface_reference_indices_for_atoms(self.atoms, group)

        for _ in range(self.max_displacement_trials):
            delta = self.rng.normal(0.0, self.displacement_sigma, size=2)
            new_xy = anchor_pos[:2] + delta
            if any(pbc[:2]):
                frac = np.linalg.solve(xy_matrix.T, new_xy)
                frac = frac % 1.0
                new_xy = np.dot(xy_matrix.T, frac)

            if surface_indices.size > 0:
                support_dxy = self._xy_distances(
                    new_xy,
                    all_pos[surface_indices, :2],
                )
                support_indices = surface_indices[support_dxy < self.support_xy_tol]
            else:
                support_indices = np.asarray([], dtype=int)
            if len(support_indices) == 0:
                new_z = float(anchor_pos[2])
            elif self.surface_side == "top":
                new_z = float(
                    np.max(all_pos[support_indices, 2]) + self.vertical_offset
                )
            else:
                new_z = float(
                    np.min(all_pos[support_indices, 2]) - self.vertical_offset
                )

            new_anchor = np.array([new_xy[0], new_xy[1], new_z], dtype=float)
            if not self._point_within_site_region(new_anchor):
                continue
            trial_positions = new_anchor + relative
            trial_positions = self._adjust_trial_positions_vertically(
                group, trial_positions
            )
            if trial_positions is None:
                continue
            if not self._group_orientation_is_valid(group, trial_positions):
                continue
            atoms_new = self.atoms.copy()
            atoms_new.positions[group] = trial_positions
            return atoms_new

        return None

    def _same_site_as_current(
        self,
        current_anchor: np.ndarray,
        target_xy: np.ndarray,
    ) -> bool:
        mic_xy = self._xy_distances(
            np.asarray(current_anchor, dtype=float)[:2],
            np.asarray(target_xy, dtype=float).reshape(1, 2),
        )[0]
        return mic_xy < self.same_site_tol

    def _site_contains_support_atom(
        self,
        site: dict[str, object],
        atom_index: Optional[int],
    ) -> bool:
        if atom_index is None:
            return False
        support_indices = np.asarray(site.get("support_indices", ()), dtype=int)
        return bool(np.any(support_indices == int(atom_index)))

    def _rotated_relative_positions_for_hop(
        self,
        relative: np.ndarray,
        max_angle_rad: float,
    ) -> Optional[np.ndarray]:
        axis = self.rng.normal(size=3)
        angle = self.rng.uniform(-max_angle_rad, max_angle_rad)
        if np.linalg.norm(axis) <= 1e-12 or abs(angle) <= 1e-12:
            return None
        rotation = _rotation_matrix(axis, angle)
        return relative @ rotation.T

    def _propose_hop(self, *, reorient: bool, pucker: bool = False) -> Optional[Atoms]:
        if reorient and (
            (not self.is_molecular_adsorbate)
            or self.hop_reorientation_angle_rad <= 0.0
        ):
            return None
        if pucker and self.puckering_height_A <= 0.0:
            return None

        movable_group_ids = self.get_non_buried_adsorbate_indices(
            support_xy_tol=self.support_xy_tol
        )
        if not movable_group_ids:
            return None

        site_registry = self._get_site_registry()
        if not site_registry:
            return None

        group_id = int(self.rng.choice(movable_group_ids))
        group = np.asarray(self.ads_groups[group_id], dtype=int)
        current_anchor = self._group_anchor_position(group)
        relative = (
            self._sample_template_group_relative_positions()
            if self.is_molecular_adsorbate
            else self._current_group_relative_positions(group)
        )
        if np.allclose(relative, 0.0):
            return None
        current_support = self._nearest_support_atom_for_anchor(group)
        reset_current_support = (
            current_support
            if current_support is not None
            and (pucker or self._support_atom_is_puckered(current_support))
            else None
        )

        site_order = self.rng.permutation(len(site_registry))
        trials_per_site = (
            max(1, int(self.max_hop_reorientation_trials)) if reorient else 1
        )
        direction = 1.0 if self.surface_side == "top" else -1.0
        for site_idx in site_order:
            site = site_registry[int(site_idx)]
            if pucker and str(site.get("site_type", "")).lower() != "atop":
                continue
            if bool(site.get("blocked_by_termination", False)):
                continue
            if pucker and self._site_contains_support_atom(site, current_support):
                continue
            xy = np.asarray(site["xy"], dtype=float)
            if self._same_site_as_current(current_anchor, xy):
                continue

            height = self._sample_puckering_height() if pucker else 0.0
            if pucker and height <= 1e-12:
                continue
            dz = direction * height
            target_support = None
            if pucker:
                target_support = self._support_atom_for_site(
                    site,
                    np.array([xy[0], xy[1], current_anchor[2]], dtype=float),
                    avoid_index=current_support,
                )
                if target_support is None:
                    continue
                target_support_position = self._puckered_support_position(
                    int(target_support),
                    dz=dz,
                )
                new_anchor = self._anchor_position_from_puckered_support(
                    target_support_position
                )
            else:
                support_atoms = self.atoms
                if reset_current_support is not None:
                    support_atoms = self.atoms.copy()
                    support_atoms.positions[int(reset_current_support)] = (
                        self._puckering_reference_position(int(reset_current_support))
                    )
                suggested_z = float(site.get("suggested_z_A", np.nan))
                if suggested_z is None or not np.isfinite(suggested_z):
                    suggested_z = self._candidate_support_z(xy, atoms=support_atoms)
                    if suggested_z is None:
                        continue
                new_anchor = np.array([xy[0], xy[1], float(suggested_z)], dtype=float)
                if self.has_afloat_adsorbates(
                    self._trial_atoms_with_group(
                        group,
                        new_anchor + relative,
                        atoms=support_atoms,
                    ),
                    support_xy_tol=self.support_xy_tol,
                    z_max_support=self.z_max_support,
                ):
                    continue
            if not self._point_within_site_region(new_anchor):
                continue

            for _ in range(trials_per_site):
                trial_relative = relative
                if reorient:
                    trial_relative = self._rotated_relative_positions_for_hop(
                        relative,
                        self.hop_reorientation_angle_rad,
                    )
                    if trial_relative is None:
                        continue

                trial_positions = new_anchor + trial_relative
                trial_positions = self._adjust_trial_positions_vertically(
                    group, trial_positions
                )
                if trial_positions is None:
                    continue
                if not self._group_orientation_is_valid(group, trial_positions):
                    continue

                atoms_new = self.atoms.copy()
                if reset_current_support is not None:
                    atoms_new.positions[int(reset_current_support)] = (
                        self._puckering_reference_position(int(reset_current_support))
                    )
                if pucker:
                    atoms_new.positions[int(target_support)] = target_support_position
                atoms_new.positions[group] = trial_positions
                if not self._anchors_within_site_region(atoms=atoms_new):
                    continue
                if self.enforce_molecular_integrity and not (
                    self._molecular_adsorbates_are_intact(atoms_new)
                ):
                    continue
                return atoms_new

        return None

    def _propose_site_hop(self) -> Optional[Atoms]:
        return self._propose_hop(reorient=False)

    def _propose_reorientation(self) -> Optional[Atoms]:
        if (not self.is_molecular_adsorbate) or self.rotation_max_angle_rad <= 0.0:
            return None

        movable_group_ids = self.get_non_buried_adsorbate_indices(
            support_xy_tol=self.support_xy_tol
        )
        if not movable_group_ids:
            return None

        group_id = int(self.rng.choice(movable_group_ids))
        group = np.asarray(self.ads_groups[group_id], dtype=int)
        relative = self._sample_template_group_relative_positions()
        if np.allclose(relative, 0.0):
            return None
        anchor_pos = self._group_anchor_position(group)

        for _ in range(self.max_reorientation_trials):
            axis = self.rng.normal(size=3)
            angle = self.rng.uniform(
                -self.rotation_max_angle_rad, self.rotation_max_angle_rad
            )
            if np.linalg.norm(axis) <= 1e-12 or abs(angle) <= 1e-12:
                continue

            rotation = _rotation_matrix(axis, angle)
            trial_positions = anchor_pos + relative @ rotation.T
            if not self._group_positions_are_valid(group, trial_positions):
                continue
            if not self._group_clears_terminations(group, trial_positions):
                continue
            if not self._group_orientation_is_valid(group, trial_positions):
                continue

            atoms_new = self.atoms.copy()
            atoms_new.positions[group] = trial_positions
            return atoms_new

        return None

    def _propose_hop_reorientation(self) -> Optional[Atoms]:
        return self._propose_hop(reorient=True)

    def _propose_hop_puckering(self) -> Optional[Atoms]:
        return self._propose_hop(reorient=False, pucker=True)

    def _propose_hop_puckering_reorientation(self) -> Optional[Atoms]:
        return self._propose_hop(reorient=True, pucker=True)

    def _nearest_support_atom_for_anchor(
        self,
        group: np.ndarray,
        atoms: Optional[Atoms] = None,
    ) -> Optional[int]:
        if atoms is None:
            atoms = self.atoms

        group = np.asarray(group, dtype=int)
        if group.size == 0:
            return None

        anchor_pos = self._group_anchor_position(group, atoms=atoms)
        group_set = set(int(idx) for idx in group.tolist())
        puckering_element_set = set(self.puckering_elements)
        candidate_indices = np.asarray(
            [
                idx
                for idx, atom in enumerate(atoms)
                if idx not in group_set and atom.symbol in puckering_element_set
            ],
            dtype=int,
        )
        if candidate_indices.size == 0:
            return None

        candidate_pos = atoms.positions[candidate_indices]
        deltas = get_distances(
            anchor_pos.reshape(1, 3),
            candidate_pos,
            cell=atoms.get_cell(),
            pbc=atoms.get_pbc(),
        )[0][0]
        dxy = np.linalg.norm(deltas[:, :2], axis=1)
        side_sign = 1.0 if self.surface_side == "top" else -1.0
        dz = side_sign * (-deltas[:, 2])
        mask = (dxy < self.support_xy_tol) & (dz > 0.0) & (dz < self.z_max_support)
        if not np.any(mask):
            return None

        masked_positions = np.where(mask)[0]
        nearest_local = int(masked_positions[int(np.argmin(dxy[masked_positions]))])
        return int(candidate_indices[nearest_local])

    def _support_atom_for_site(
        self,
        site: dict[str, object],
        anchor_position: np.ndarray,
        *,
        avoid_index: Optional[int] = None,
        atoms: Optional[Atoms] = None,
    ) -> Optional[int]:
        if atoms is None:
            atoms = self.atoms

        support_indices = np.asarray(site.get("support_indices", ()), dtype=int)
        support_indices = support_indices[
            (support_indices >= 0) & (support_indices < len(atoms))
        ]
        puckering_element_set = set(self.puckering_elements)
        support_indices = np.asarray(
            [
                idx
                for idx in support_indices
                if atoms[int(idx)].symbol in puckering_element_set
            ],
            dtype=int,
        )
        if support_indices.size == 0:
            return self._nearest_support_atom_for_point(
                anchor_position,
                avoid_index=avoid_index,
                atoms=atoms,
            )

        preferred = support_indices
        if avoid_index is not None and support_indices.size > 1:
            without_avoid = support_indices[support_indices != int(avoid_index)]
            if without_avoid.size > 0:
                preferred = without_avoid

        deltas = get_distances(
            np.asarray(anchor_position, dtype=float).reshape(1, 3),
            atoms.positions[preferred],
            cell=atoms.get_cell(),
            pbc=atoms.get_pbc(),
        )[0][0]
        dxy = np.linalg.norm(deltas[:, :2], axis=1)
        return int(preferred[int(np.argmin(dxy))])

    def _nearest_support_atom_for_point(
        self,
        anchor_position: np.ndarray,
        *,
        avoid_index: Optional[int] = None,
        atoms: Optional[Atoms] = None,
    ) -> Optional[int]:
        if atoms is None:
            atoms = self.atoms

        anchor_position = np.asarray(anchor_position, dtype=float)
        puckering_element_set = set(self.puckering_elements)
        candidate_indices = np.asarray(
            [
                idx
                for idx, atom in enumerate(atoms)
                if atom.symbol in puckering_element_set
                and (avoid_index is None or idx != int(avoid_index))
            ],
            dtype=int,
        )
        if candidate_indices.size == 0:
            return None

        candidate_pos = atoms.positions[candidate_indices]
        deltas = get_distances(
            anchor_position.reshape(1, 3),
            candidate_pos,
            cell=atoms.get_cell(),
            pbc=atoms.get_pbc(),
        )[0][0]
        dxy = np.linalg.norm(deltas[:, :2], axis=1)
        side_sign = 1.0 if self.surface_side == "top" else -1.0
        dz = side_sign * (-deltas[:, 2])
        mask = (dxy < self.support_xy_tol) & (dz > 0.0) & (dz < self.z_max_support)
        if not np.any(mask):
            return None

        masked_positions = np.where(mask)[0]
        nearest_local = int(masked_positions[int(np.argmin(dxy[masked_positions]))])
        return int(candidate_indices[nearest_local])

    def _puckering_reference_z(self, atom_index: int) -> float:
        if 0 <= int(atom_index) < len(self._puckering_reference_positions):
            return float(self._puckering_reference_positions[int(atom_index), 2])
        return float(self.atoms.positions[int(atom_index), 2])

    def _puckering_reference_position(self, atom_index: int) -> np.ndarray:
        if 0 <= int(atom_index) < len(self._puckering_reference_positions):
            return np.asarray(
                self._puckering_reference_positions[int(atom_index)],
                dtype=float,
            ).copy()
        return np.asarray(self.atoms.positions[int(atom_index)], dtype=float).copy()

    def _puckered_support_position(
        self,
        atom_index: int,
        *,
        dz: float,
    ) -> np.ndarray:
        position = self._puckering_reference_position(atom_index)
        position[2] += float(dz)
        return position

    def _support_atom_is_puckered(self, atom_index: int) -> bool:
        direction = 1.0 if self.surface_side == "top" else -1.0
        dz = direction * (
            float(self.atoms.positions[int(atom_index), 2])
            - self._puckering_reference_z(int(atom_index))
        )
        threshold = max(0.05, 0.15 * float(self.puckering_height_A))
        return dz > threshold

    def _anchor_position_from_puckered_support(
        self,
        support_position: np.ndarray,
    ) -> np.ndarray:
        direction = 1.0 if self.surface_side == "top" else -1.0
        anchor = np.asarray(support_position, dtype=float).copy()
        anchor[2] += direction * self.vertical_offset
        return anchor

    def _puckered_group_positions(
        self,
        group: np.ndarray,
        support_position: np.ndarray,
        *,
        relative: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if relative is None:
            relative = self._molecular_move_relative_positions(group)
        target_anchor = self._anchor_position_from_puckered_support(support_position)
        return target_anchor + np.asarray(relative, dtype=float)

    def _reference_site_suggested_z(self, site: dict[str, object]) -> Optional[float]:
        support_indices = np.asarray(site.get("support_indices", ()), dtype=int)
        support_indices = support_indices[
            (support_indices >= 0)
            & (support_indices < len(self._puckering_reference_positions))
        ]
        if support_indices.size == 0:
            suggested_z = float(site.get("suggested_z_A", np.nan))
            return suggested_z if np.isfinite(suggested_z) else None

        ref_z = self._puckering_reference_positions[support_indices, 2]
        if self.surface_side == "top":
            return float(np.max(ref_z) + self.vertical_offset)
        return float(np.min(ref_z) - self.vertical_offset)

    def _sample_puckering_height(self) -> float:
        if self.puckering_height_A <= 0.0:
            return 0.0
        jitter = max(0.0, float(self.puckering_height_jitter_A))
        if jitter <= 1e-12:
            return float(self.puckering_height_A)
        low = max(0.0, float(self.puckering_height_A) - jitter)
        high = float(self.puckering_height_A) + jitter
        return float(self.rng.uniform(low, high))

    def _propose_puckering(self) -> Optional[Atoms]:
        if self.puckering_height_A <= 0.0:
            return None

        movable_group_ids = self.get_non_buried_adsorbate_indices(
            support_xy_tol=self.support_xy_tol
        )
        if not movable_group_ids:
            return None

        direction = 1.0 if self.surface_side == "top" else -1.0
        for _ in range(self.max_puckering_trials):
            group_id = int(self.rng.choice(movable_group_ids))
            group = np.asarray(self.ads_groups[group_id], dtype=int)
            support_idx = self._nearest_support_atom_for_anchor(group)
            if support_idx is None:
                continue

            height = self._sample_puckering_height()
            if height <= 1e-12:
                continue
            dz = direction * height
            atoms_new = self.atoms.copy()
            support_position = self._puckered_support_position(support_idx, dz=dz)
            atoms_new.positions[support_idx] = support_position
            trial_positions = self._puckered_group_positions(group, support_position)
            if not self._group_orientation_is_valid(group, trial_positions):
                continue
            atoms_new.positions[group] = trial_positions

            if not self._anchors_within_site_region(atoms=atoms_new):
                continue
            if not self._group_positions_are_valid(
                group, trial_positions, atoms=atoms_new
            ):
                continue
            if not self._group_clears_terminations(
                group, trial_positions, atoms=atoms_new
            ):
                continue
            if self.enforce_molecular_integrity and not (
                self._molecular_adsorbates_are_intact(atoms_new)
            ):
                continue

            return atoms_new

        return None

    def _propose_move(self) -> Optional[Atoms]:
        self._last_proposal_move_name = None
        if self.move_mode != "hybrid":
            self._last_proposal_move_name = self.move_mode
            return self._proposal_for_mode(self.move_mode)()

        selector = self.rng.random()
        cumulative = 0.0
        for weight, mode, propose in self._hybrid_move_table:
            cumulative += weight
            if selector < cumulative:
                self._last_proposal_move_name = mode
                return propose()

        self._last_proposal_move_name = "displacement"
        return self._propose_displacement()
