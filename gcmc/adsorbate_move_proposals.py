import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

from ase import Atoms
from ase.geometry import find_mic, get_distances


@dataclass
class MoveProposal:
    atoms: Atoms
    move_name: str
    is_md: bool = False
    delta_e: Optional[float] = None
    delta_h: Optional[float] = None
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class AdsorptionChannel:
    channel_id: int
    site_index: int
    basin: str
    label: str


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
    def _reject_move_proposal(self, reason: str) -> None:
        self._last_proposal_reject_reason = str(reason)
        return None

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
            "channel_hop": self._propose_channel_hop,
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
            (self.channel_hop_prob, "channel_hop"),
            (self.reorientation_prob, "reorientation"),
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

    def _nearest_site_index_for_anchor(
        self,
        anchor: np.ndarray,
        site_registry: list[dict[str, object]],
    ) -> Optional[int]:
        if not site_registry:
            return None
        distances = np.asarray(
            [
                self._xy_distances(
                    np.asarray(anchor, dtype=float)[:2],
                    np.asarray(site["xy"], dtype=float).reshape(1, 2),
                )[0]
                for site in site_registry
            ],
            dtype=float,
        )
        return int(np.argmin(distances))

    def _site_support_indices(
        self,
        site: dict[str, object],
        *,
        atoms: Optional[Atoms] = None,
    ) -> np.ndarray:
        if atoms is None:
            atoms = self.atoms
        support_indices = np.asarray(site.get("support_indices", ()), dtype=int)
        return support_indices[
            (support_indices >= 0) & (support_indices < len(atoms))
        ]

    def _site_anchor_position(
        self,
        site: dict[str, object],
        *,
        atoms: Optional[Atoms] = None,
    ) -> np.ndarray:
        """Return a dynamic anchor for a site's fixed support-atom identity."""
        if atoms is None:
            atoms = self.atoms

        anchor = np.array(
            [
                float(np.asarray(site["xy"], dtype=float)[0]),
                float(np.asarray(site["xy"], dtype=float)[1]),
                float(site.get("suggested_z_A", np.nan)),
            ],
            dtype=float,
        )
        support_indices = self._site_support_indices(site, atoms=atoms)
        reference_mask = support_indices < len(self._puckering_reference_positions)
        reference_indices = support_indices[reference_mask]
        if reference_indices.size > 0:
            displacements = (
                atoms.positions[reference_indices]
                - self._puckering_reference_positions[reference_indices]
            )
            displacements, _ = find_mic(
                displacements,
                atoms.get_cell(),
                atoms.get_pbc(),
            )
            anchor[:2] += np.mean(displacements[:, :2], axis=0)

        if support_indices.size > 0:
            if self.surface_side == "top":
                anchor[2] = (
                    float(np.max(atoms.positions[support_indices, 2]))
                    + self.vertical_offset
                )
            else:
                anchor[2] = (
                    float(np.min(atoms.positions[support_indices, 2]))
                    - self.vertical_offset
                )
        return anchor

    def _atop_puckering_support(
        self,
        site: dict[str, object],
        *,
        atoms: Optional[Atoms] = None,
    ) -> Optional[int]:
        if atoms is None:
            atoms = self.atoms
        if str(site.get("site_type", "")).lower() != "atop":
            return None
        eligible = set(self.puckering_elements)
        support_indices = self._site_support_indices(site, atoms=atoms)
        for support_idx in support_indices:
            element = atoms[int(support_idx)].symbol
            if (
                element in eligible
                and self._puckering_height_parameters(element)[0] > 0.0
            ):
                return int(support_idx)
        return None

    def _signed_puckering_displacement(
        self,
        atom_index: int,
        *,
        atoms: Optional[Atoms] = None,
        exclude_indices: tuple[int, ...] = (),
    ) -> float:
        if atoms is None:
            atoms = self.atoms
        direction = 1.0 if self.surface_side == "top" else -1.0
        local_reference = self._puckering_local_reference_displacement(
            int(atom_index),
            atoms=atoms,
            exclude_indices=exclude_indices,
        )
        return direction * (
            float(atoms.positions[int(atom_index), 2])
            - self._puckering_reference_z(int(atom_index))
            - local_reference
        )

    def _puckering_surface_support_indices(self) -> np.ndarray:
        cached = getattr(self, "_puckering_surface_support_indices_cache", None)
        if cached is not None:
            return cached

        support_indices: set[int] = set()
        for site in self._get_site_registry():
            support_indices.update(
                int(idx) for idx in self._site_support_indices(site, atoms=self.atoms)
            )

        eligible = set(self.site_elements)
        candidates = np.asarray(
            [
                int(idx)
                for idx in self.sub_indices
                if self.atoms[int(idx)].symbol in eligible
            ],
            dtype=int,
        )
        if candidates.size:
            direction = 1.0 if self.surface_side == "top" else -1.0
            signed_z = direction * self._puckering_reference_positions[
                candidates, 2
            ]
            order = np.argsort(signed_z, kind="stable")
            clusters = np.split(
                order,
                np.where(np.diff(signed_z[order]) > self.surface_layer_tol)[0] + 1,
            )
            largest_size = max(len(cluster) for cluster in clusters)
            minimum_surface_size = min(
                largest_size,
                max(2, int(np.ceil(0.5 * largest_size))),
            )
            surface_cluster = max(
                (
                    cluster
                    for cluster in clusters
                    if len(cluster) >= minimum_surface_size
                ),
                key=lambda cluster: float(np.mean(signed_z[cluster])),
            )
            support_indices.update(int(idx) for idx in candidates[surface_cluster])

        cached = np.asarray(sorted(support_indices), dtype=int)
        self._puckering_surface_support_indices_cache = cached
        return cached

    def _puckering_local_reference_weights(
        self,
        atom_index: int,
        *,
        exclude_indices: tuple[int, ...] = (),
    ) -> tuple[np.ndarray, np.ndarray]:
        excluded_set = {int(atom_index)}
        excluded_set.update(int(idx) for idx in exclude_indices)
        excluded = tuple(sorted(excluded_set))
        cache_key = (int(atom_index), excluded)
        cache = self._puckering_local_reference_cache
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        candidates = self._puckering_surface_support_indices()
        if candidates.size:
            candidates = candidates[~np.isin(candidates, np.asarray(excluded, dtype=int))]
        if candidates.size == 0:
            result = (np.asarray([], dtype=int), np.asarray([], dtype=float))
            cache[cache_key] = result
            return result

        reference_xy = self._puckering_reference_positions[:, :2]
        displacements = self._xy_displacements(
            reference_xy[int(atom_index)],
            reference_xy[candidates],
            atoms=self.atoms,
        )
        distances = np.linalg.norm(displacements, axis=1)
        order = np.argsort(distances, kind="stable")[: min(8, len(candidates))]
        neighbors = candidates[order]
        local_xy = displacements[order]

        if len(neighbors) >= 3:
            # The intercept of this fixed affine fit follows translation and tilt.
            design = np.column_stack((np.ones(len(neighbors)), local_xy))
            if np.linalg.matrix_rank(design) == 3:
                weights = np.linalg.pinv(design, rcond=1e-12)[0]
            else:
                weights = np.full(len(neighbors), 1.0 / len(neighbors))
        else:
            weights = np.full(len(neighbors), 1.0 / len(neighbors))

        weight_sum = float(np.sum(weights))
        if not np.isfinite(weight_sum) or abs(weight_sum) <= 1e-12:
            weights = np.full(len(neighbors), 1.0 / len(neighbors))
        else:
            weights = np.asarray(weights, dtype=float) / weight_sum
        result = (np.asarray(neighbors, dtype=int), weights)
        cache[cache_key] = result
        return result

    def _puckering_local_reference_displacement(
        self,
        atom_index: int,
        *,
        atoms: Optional[Atoms] = None,
        exclude_indices: tuple[int, ...] = (),
    ) -> float:
        if atoms is None:
            atoms = self.atoms
        neighbors, weights = self._puckering_local_reference_weights(
            int(atom_index),
            exclude_indices=exclude_indices,
        )
        if neighbors.size == 0:
            return 0.0
        displacements = (
            atoms.positions[neighbors, 2]
            - self._puckering_reference_positions[neighbors, 2]
        )
        return float(np.dot(weights, displacements))

    def _puckering_basin_limits(self, element: str) -> tuple[float, float, float]:
        height, jitter = self._puckering_height_parameters(element)
        boundary = 0.5 * float(height)
        return -boundary, boundary, float(height) + float(jitter)

    def _puckering_state_for_height(
        self,
        height: float,
        *,
        element: str,
    ) -> Optional[str]:
        lower, boundary, upper = self._puckering_basin_limits(element)
        tolerance = 1e-10
        if height < lower - tolerance or height > upper + tolerance:
            return None
        return "high" if height > boundary else "low"

    def _puckering_state(
        self,
        atom_index: int,
        *,
        atoms: Optional[Atoms] = None,
        exclude_indices: tuple[int, ...] = (),
    ) -> Optional[str]:
        if atoms is None:
            atoms = self.atoms
        height = self._signed_puckering_displacement(
            int(atom_index),
            atoms=atoms,
            exclude_indices=exclude_indices,
        )
        return self._puckering_state_for_height(
            height,
            element=atoms[int(atom_index)].symbol,
        )

    def _build_adsorption_channel_registry(self) -> list[AdsorptionChannel]:
        channels: list[AdsorptionChannel] = []
        for site_index, site in enumerate(self._get_site_registry()):
            site_type = str(site.get("site_type", "site")).lower()
            site_id = int(site.get("site_id", site_index))
            support_idx = self._atop_puckering_support(site, atoms=self.atoms)
            basins = (
                ("low", "high")
                if self.channel_puckering_enabled and support_idx is not None
                else ("base",)
            )
            for basin in basins:
                channels.append(
                    AdsorptionChannel(
                        channel_id=len(channels),
                        site_index=int(site_index),
                        basin=basin,
                        label=f"{site_type}_{site_id}:{basin}",
                    )
                )
        return channels

    def _get_adsorption_channel_registry(self) -> list[AdsorptionChannel]:
        if self._adsorption_channel_registry is None:
            channels = self._build_adsorption_channel_registry()
            self._adsorption_channel_registry = channels
        else:
            channels = self._adsorption_channel_registry
        if self._adsorption_channel_lookup is None:
            self._adsorption_channel_lookup = {
                (channel.site_index, channel.basin): channel
                for channel in channels
            }
        return self._adsorption_channel_registry

    def _site_channel_basin(
        self,
        site: dict[str, object],
        *,
        atoms: Optional[Atoms] = None,
        exclude_indices: tuple[int, ...] = (),
    ) -> Optional[str]:
        if atoms is None:
            atoms = self.atoms
        support_idx = self._atop_puckering_support(site, atoms=atoms)
        if support_idx is None:
            return "base"
        state = self._puckering_state(
            support_idx,
            atoms=atoms,
            exclude_indices=exclude_indices or (support_idx,),
        )
        if state == "high" and not self.channel_puckering_enabled:
            return None
        return state

    def _channel_for_group(
        self,
        group: np.ndarray,
        *,
        atoms: Optional[Atoms] = None,
    ) -> Optional[AdsorptionChannel]:
        if atoms is None:
            atoms = self.atoms
        registry = self._get_site_registry()
        anchor = self._group_anchor_position(group, atoms=atoms)
        site_index = self._nearest_site_index_for_anchor(anchor, registry)
        if site_index is None:
            return None
        site = registry[site_index]
        basin = self._site_channel_basin(site, atoms=atoms)
        if basin is None:
            return None
        self._get_adsorption_channel_registry()
        return self._adsorption_channel_lookup.get((site_index, basin))

    def _draw_target_channel(
        self,
        source_channel: AdsorptionChannel,
    ) -> Optional[AdsorptionChannel]:
        candidates = [
            channel
            for channel in self._get_adsorption_channel_registry()
            if channel.channel_id != source_channel.channel_id
        ]
        if not candidates:
            return None
        return candidates[int(self.rng.choice(np.arange(len(candidates), dtype=int)))]

    def _apply_channel_endpoint_states(
        self,
        atoms_new: Atoms,
        *,
        source_site: dict[str, object],
        source_basin: str,
        target_site: dict[str, object],
        target_basin: str,
    ) -> Optional[str]:
        source_support = self._atop_puckering_support(
            source_site,
            atoms=self.atoms,
        )
        target_support = self._atop_puckering_support(
            target_site,
            atoms=self.atoms,
        )
        same_support = (
            source_support is not None and source_support == target_support
        )
        endpoint_supports = tuple(
            dict.fromkeys(
                int(idx)
                for idx in (source_support, target_support)
                if idx is not None
            )
        )

        transitions: list[tuple[int, str, str]] = []
        if same_support:
            transitions.append(
                (int(source_support), source_basin, target_basin)
            )
        else:
            if source_support is not None:
                transitions.append(
                    (int(source_support), source_basin, "low")
                )
            if target_support is not None:
                transitions.append(
                    (int(target_support), "low", target_basin)
                )

        reflected: list[tuple[int, float]] = []
        for support_idx, expected_state, desired_state in transitions:
            current_height = self._signed_puckering_displacement(
                support_idx,
                atoms=self.atoms,
                exclude_indices=endpoint_supports,
            )
            element = self.atoms[support_idx].symbol
            current_state = self._puckering_state_for_height(
                current_height,
                element=element,
            )
            if current_state is None:
                return "channel_endpoint_out_of_bounds"
            if current_state != expected_state:
                return f"channel_endpoint_not_{expected_state}"
            if desired_state == current_state:
                continue
            transition_height = self._sample_puckering_height(element)
            target_height = transition_height - current_height
            if self._puckering_state_for_height(
                target_height,
                element=element,
            ) != desired_state:
                return "channel_reflection_misses_target_basin"
            reflected.append((support_idx, transition_height))

        for support_idx, transition_height in reflected:
            atoms_new.positions[support_idx] = self._complement_puckering_position(
                support_idx,
                atoms=self.atoms,
                transition_height=transition_height,
                exclude_indices=endpoint_supports,
            )
        return None

    def _propose_channel_transition(
        self,
        *,
        target_same_site: bool = False,
        force_reorient: Optional[bool] = None,
    ) -> Optional[Atoms]:
        if not self.ads_groups:
            return self._reject_move_proposal("no_adsorbate_group")

        group_id = int(self.rng.choice(np.arange(len(self.ads_groups), dtype=int)))
        if group_id not in self.get_non_buried_adsorbate_indices(
            support_xy_tol=self.support_xy_tol
        ):
            return self._reject_move_proposal("buried_adsorbate")
        group = np.asarray(self.ads_groups[group_id], dtype=int)
        source_channel = self._channel_for_group(group, atoms=self.atoms)
        if source_channel is None:
            return self._reject_move_proposal("current_channel_ambiguous")

        if target_same_site:
            candidates = [
                channel
                for channel in self._get_adsorption_channel_registry()
                if channel.site_index == source_channel.site_index
                and channel.channel_id != source_channel.channel_id
            ]
            target_channel = candidates[0] if len(candidates) == 1 else None
        else:
            target_channel = self._draw_target_channel(source_channel)
        if target_channel is None:
            return self._reject_move_proposal("target_channel_unavailable")

        registry = self._get_site_registry()
        source_site = registry[source_channel.site_index]
        target_site = registry[target_channel.site_index]
        if bool(source_site.get("blocked_by_termination", False)) or bool(
            target_site.get("blocked_by_termination", False)
        ):
            return self._reject_move_proposal("termination_blocked_channel")

        current_anchor = self._group_anchor_position(group)
        source_base = self._site_anchor_position(source_site, atoms=self.atoms)
        anchor_residual = get_distances(
            source_base.reshape(1, 3),
            current_anchor.reshape(1, 3),
            cell=self.atoms.get_cell(),
            pbc=self.atoms.get_pbc(),
        )[0][0, 0]

        atoms_new = self.atoms.copy()
        reject_reason = self._apply_channel_endpoint_states(
            atoms_new,
            source_site=source_site,
            source_basin=source_channel.basin,
            target_site=target_site,
            target_basin=target_channel.basin,
        )
        if reject_reason is not None:
            return self._reject_move_proposal(reject_reason)

        target_base = self._site_anchor_position(target_site, atoms=atoms_new)
        new_anchor = target_base + anchor_residual
        if not self._point_within_site_region(new_anchor, atoms=atoms_new):
            return self._reject_move_proposal("site_region")

        relative = self._current_group_relative_positions(group)
        spatial_hop = source_channel.site_index != target_channel.site_index
        if force_reorient is None:
            reorient = (
                spatial_hop
                and self.is_molecular_adsorbate
                and self.channel_reorientation_prob > 0.0
                and self.rng.random() < self.channel_reorientation_prob
            )
        else:
            reorient = bool(force_reorient) and spatial_hop
        if reorient:
            relative = self._rotated_relative_positions_for_hop(
                relative,
                self.hop_reorientation_angle_rad,
            )
            if relative is None:
                return self._reject_move_proposal("degenerate_reorientation")
        if np.allclose(relative, 0.0):
            return self._reject_move_proposal("degenerate_adsorbate_geometry")

        trial_positions = new_anchor + relative
        if not self._group_positions_are_valid(
            group,
            trial_positions,
            atoms=atoms_new,
        ) or not self._group_clears_terminations(
            group,
            trial_positions,
            atoms=atoms_new,
        ):
            return self._reject_move_proposal("proposal_geometry")
        if not self._group_orientation_is_valid(
            group,
            trial_positions,
            atoms=atoms_new,
        ):
            return self._reject_move_proposal("molecular_orientation")

        atoms_new.positions[group] = trial_positions
        final_channel = self._channel_for_group(group, atoms=atoms_new)
        if (
            final_channel is None
            or final_channel.channel_id != target_channel.channel_id
        ):
            return self._reject_move_proposal("target_channel_mismatch")
        if not self._anchors_within_site_region(atoms=atoms_new):
            return self._reject_move_proposal("site_region")
        if self.has_afloat_adsorbates(
            atoms_new,
            support_xy_tol=self.support_xy_tol,
            z_max_support=self.z_max_support,
        ):
            return self._reject_move_proposal("afloat_adsorbate")
        if self.enforce_molecular_integrity and not (
            self._molecular_adsorbates_are_intact(atoms_new)
        ):
            return self._reject_move_proposal("molecular_integrity")

        self._last_proposal_metadata = {
            "source_channel": source_channel.label,
            "target_channel": target_channel.label,
            "channel_reoriented": bool(reorient),
        }
        return atoms_new

    def _propose_channel_hop(self) -> Optional[Atoms]:
        return self._propose_channel_transition()

    def _complement_puckering_position(
        self,
        atom_index: int,
        *,
        atoms: Optional[Atoms] = None,
        transition_height: Optional[float] = None,
        exclude_indices: tuple[int, ...] = (),
    ) -> np.ndarray:
        """Reflect the outward support displacement as h' = H - h."""
        if atoms is None:
            atoms = self.atoms
        if transition_height is None:
            transition_height = self._puckering_height_parameters(
                atoms[int(atom_index)].symbol
            )[0]
        current_height = self._signed_puckering_displacement(
            int(atom_index),
            atoms=atoms,
            exclude_indices=exclude_indices,
        )
        target_height = float(transition_height) - current_height
        direction = 1.0 if self.surface_side == "top" else -1.0
        local_reference = self._puckering_local_reference_displacement(
            int(atom_index),
            atoms=atoms,
            exclude_indices=exclude_indices,
        )
        position = np.asarray(atoms.positions[int(atom_index)], dtype=float).copy()
        position[2] = (
            self._puckering_reference_z(int(atom_index))
            + local_reference
            + direction * target_height
        )
        return position

    def _apply_reversible_endpoint_puckering(
        self,
        atoms_new: Atoms,
        *,
        source_site: dict[str, object],
        target_site: dict[str, object],
    ) -> Optional[str]:
        """Unpucker an occupied source and pucker an unoccupied atop target."""
        source_support = self._atop_puckering_support(
            source_site,
            atoms=self.atoms,
        )
        target_support = self._atop_puckering_support(
            target_site,
            atoms=self.atoms,
        )
        if source_support is not None and source_support == target_support:
            return "same_puckering_support"

        endpoint_supports = tuple(
            int(idx)
            for idx in (source_support, target_support)
            if idx is not None
        )
        if not endpoint_supports:
            return "no_atop_puckering_endpoint"

        endpoint_data: list[tuple[int, float]] = []
        for support_idx, departing in (
            (source_support, True),
            (target_support, False),
        ):
            if support_idx is None:
                continue
            # For a realized pair, H = h + h'. The same element-specific
            # density therefore appears in the reverse proposal.
            transition_height = self._sample_puckering_height(
                self.atoms[int(support_idx)].symbol
            )
            if transition_height <= 0.0:
                return "nonpositive_puckering_height"
            current_height = self._signed_puckering_displacement(
                int(support_idx),
                atoms=self.atoms,
                exclude_indices=endpoint_supports,
            )
            element = self.atoms[int(support_idx)].symbol
            current_state = self._puckering_state_for_height(
                current_height,
                element=element,
            )
            if current_state is None:
                return "puckering_endpoint_out_of_bounds"
            expected_state = "high" if departing else "low"
            if current_state != expected_state:
                return f"puckering_endpoint_not_{expected_state}"
            target_height = transition_height - current_height
            target_state = self._puckering_state_for_height(
                target_height,
                element=element,
            )
            expected_target_state = "low" if departing else "high"
            if target_state != expected_target_state:
                return "puckering_reflection_misses_opposite_basin"
            endpoint_data.append((int(support_idx), transition_height))

        for support_idx, transition_height in endpoint_data:
            atoms_new.positions[support_idx] = self._complement_puckering_position(
                support_idx,
                atoms=self.atoms,
                transition_height=transition_height,
                exclude_indices=endpoint_supports,
            )
        return None

    def _single_hop_target_index(
        self,
        source_index: int,
        site_registry: list[dict[str, object]],
    ) -> Optional[int]:
        """Draw one target; callers treat an invalid draw as a self-transition."""
        candidates = np.asarray(
            [idx for idx in range(len(site_registry)) if idx != int(source_index)],
            dtype=int,
        )
        if candidates.size == 0:
            return None
        return int(self.rng.choice(candidates))

    def _propose_hop(self, *, reorient: bool, pucker: bool = False) -> Optional[Atoms]:
        if reorient and (
            (not self.is_molecular_adsorbate)
            or self.hop_reorientation_angle_rad <= 0.0
        ):
            return self._reject_move_proposal("reorientation_unavailable")
        if pucker and not self._has_positive_puckering_height():
            return self._reject_move_proposal("nonpositive_puckering_height")

        if not self.ads_groups:
            return self._reject_move_proposal("no_adsorbate_group")

        site_registry = self._get_site_registry()
        if len(site_registry) < 2:
            return self._reject_move_proposal("insufficient_hop_sites")

        group_id = int(self.rng.choice(np.arange(len(self.ads_groups), dtype=int)))
        if group_id not in self.get_non_buried_adsorbate_indices(
            support_xy_tol=self.support_xy_tol
        ):
            return self._reject_move_proposal("buried_adsorbate")
        group = np.asarray(self.ads_groups[group_id], dtype=int)
        current_anchor = self._group_anchor_position(group)
        source_index = self._nearest_site_index_for_anchor(
            current_anchor,
            site_registry,
        )
        if source_index is None:
            return self._reject_move_proposal("source_site_not_found")
        target_index = self._single_hop_target_index(source_index, site_registry)
        if target_index is None:
            return self._reject_move_proposal("target_site_not_found")

        source_site = site_registry[source_index]
        target_site = site_registry[target_index]
        if bool(source_site.get("blocked_by_termination", False)) or bool(
            target_site.get("blocked_by_termination", False)
        ):
            return self._reject_move_proposal("termination_blocked_site")

        source_support = self._atop_puckering_support(source_site, atoms=self.atoms)
        target_support = self._atop_puckering_support(target_site, atoms=self.atoms)
        endpoint_supports = tuple(
            int(idx)
            for idx in (source_support, target_support)
            if idx is not None
        )
        if not pucker:
            for support_idx in endpoint_supports:
                state = self._puckering_state(
                    support_idx,
                    atoms=self.atoms,
                    exclude_indices=endpoint_supports,
                )
                if state is None:
                    return self._reject_move_proposal(
                        "plain_hop_endpoint_out_of_bounds"
                    )
                if state != "low":
                    return self._reject_move_proposal(
                        "plain_hop_puckered_endpoint"
                    )

        relative = self._current_group_relative_positions(group)
        if not pucker and self.is_molecular_adsorbate:
            relative = self._sample_template_group_relative_positions()
        if np.allclose(relative, 0.0):
            return self._reject_move_proposal("degenerate_adsorbate_geometry")

        source_base = self._site_anchor_position(source_site, atoms=self.atoms)
        anchor_residual = get_distances(
            source_base.reshape(1, 3),
            current_anchor.reshape(1, 3),
            cell=self.atoms.get_cell(),
            pbc=self.atoms.get_pbc(),
        )[0][0, 0]

        atoms_new = self.atoms.copy()
        if pucker:
            pucker_reject_reason = self._apply_reversible_endpoint_puckering(
                atoms_new,
                source_site=source_site,
                target_site=target_site,
            )
            if pucker_reject_reason is not None:
                return self._reject_move_proposal(pucker_reject_reason)

        target_base = self._site_anchor_position(target_site, atoms=atoms_new)
        new_anchor = target_base + anchor_residual
        if not self._point_within_site_region(new_anchor, atoms=atoms_new):
            return self._reject_move_proposal("site_region")

        trial_relative = relative
        if reorient:
            trial_relative = self._rotated_relative_positions_for_hop(
                relative,
                self.hop_reorientation_angle_rad,
            )
            if trial_relative is None:
                return self._reject_move_proposal("degenerate_reorientation")

        trial_positions = new_anchor + trial_relative
        if not pucker:
            trial_positions = self._adjust_trial_positions_vertically(
                group,
                trial_positions,
                atoms=atoms_new,
            )
            if trial_positions is None:
                return self._reject_move_proposal("vertical_adjustment_failed")
        elif not self._group_positions_are_valid(
            group,
            trial_positions,
            atoms=atoms_new,
        ) or not self._group_clears_terminations(
            group,
            trial_positions,
            atoms=atoms_new,
        ):
            return self._reject_move_proposal("proposal_geometry")

        if not self._group_orientation_is_valid(
            group,
            trial_positions,
            atoms=atoms_new,
        ):
            return self._reject_move_proposal("molecular_orientation")

        atoms_new.positions[group] = trial_positions
        proposed_anchor = self._group_anchor_position(group, atoms=atoms_new)
        if self._nearest_site_index_for_anchor(
            proposed_anchor,
            site_registry,
        ) != target_index:
            return self._reject_move_proposal("target_site_mismatch")
        if not self._anchors_within_site_region(atoms=atoms_new):
            return self._reject_move_proposal("site_region")
        if self.has_afloat_adsorbates(
            atoms_new,
            support_xy_tol=self.support_xy_tol,
            z_max_support=self.z_max_support,
        ):
            return self._reject_move_proposal("afloat_adsorbate")
        if self.enforce_molecular_integrity and not (
            self._molecular_adsorbates_are_intact(atoms_new)
        ):
            return self._reject_move_proposal("molecular_integrity")
        return atoms_new

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

    def _puckering_reference_z(self, atom_index: int) -> float:
        if 0 <= int(atom_index) < len(self._puckering_reference_positions):
            return float(self._puckering_reference_positions[int(atom_index), 2])
        return float(self.atoms.positions[int(atom_index), 2])

    def _support_atom_is_puckered(
        self,
        atom_index: int,
        *,
        exclude_indices: tuple[int, ...] = (),
    ) -> bool:
        return self._puckering_state(
            int(atom_index),
            atoms=self.atoms,
            exclude_indices=exclude_indices,
        ) == "high"

    def _puckering_height_parameters(
        self,
        element: Optional[str] = None,
    ) -> tuple[float, float]:
        specific = (
            self.puckering_heights.get(str(element))
            if element is not None
            else None
        )
        if specific is not None:
            return specific
        return (
            float(self.puckering_height_A),
            float(self.puckering_height_jitter_A),
        )

    def _has_positive_puckering_height(self) -> bool:
        return any(
            self._puckering_height_parameters(element)[0] > 0.0
            for element in self.puckering_elements
        )

    def _sample_puckering_height(self, element: Optional[str] = None) -> float:
        height, jitter = self._puckering_height_parameters(element)
        if height <= 0.0:
            return 0.0
        jitter = max(0.0, jitter)
        if jitter <= 1e-12:
            return height
        low = max(0.0, height - jitter)
        high = height + jitter
        return float(self.rng.uniform(low, high))

    def _propose_puckering(self) -> Optional[Atoms]:
        if not self._has_positive_puckering_height():
            return self._reject_move_proposal("nonpositive_puckering_height")

        if not self.ads_groups:
            return self._reject_move_proposal("no_adsorbate_group")

        group_id = int(self.rng.choice(np.arange(len(self.ads_groups), dtype=int)))
        if group_id not in self.get_non_buried_adsorbate_indices(
            support_xy_tol=self.support_xy_tol
        ):
            return self._reject_move_proposal("buried_adsorbate")
        group = np.asarray(self.ads_groups[group_id], dtype=int)
        support_idx = self._nearest_support_atom_for_anchor(group)
        if support_idx is None:
            return self._reject_move_proposal("adsorbate_not_atop")

        transition_height = self._sample_puckering_height(
            self.atoms[int(support_idx)].symbol
        )
        if transition_height <= 1e-12:
            return self._reject_move_proposal("nonpositive_puckering_height")

        endpoint_supports = (int(support_idx),)
        current_height = self._signed_puckering_displacement(
            support_idx,
            atoms=self.atoms,
            exclude_indices=endpoint_supports,
        )
        element = self.atoms[int(support_idx)].symbol
        current_state = self._puckering_state_for_height(
            current_height,
            element=element,
        )
        if current_state is None:
            return self._reject_move_proposal("puckering_endpoint_out_of_bounds")
        target_height = transition_height - current_height
        target_state = self._puckering_state_for_height(
            target_height,
            element=element,
        )
        expected_target = "high" if current_state == "low" else "low"
        if target_state != expected_target:
            return self._reject_move_proposal(
                "puckering_reflection_misses_opposite_basin"
            )

        atoms_new = self.atoms.copy()
        support_position = self._complement_puckering_position(
            support_idx,
            atoms=self.atoms,
            transition_height=transition_height,
            exclude_indices=endpoint_supports,
        )
        support_delta = support_position - self.atoms.positions[int(support_idx)]
        atoms_new.positions[int(support_idx)] = support_position
        trial_positions = self.atoms.positions[group] + support_delta
        if not self._group_orientation_is_valid(
            group,
            trial_positions,
            atoms=atoms_new,
        ):
            return self._reject_move_proposal("molecular_orientation")
        atoms_new.positions[group] = trial_positions

        if not self._anchors_within_site_region(atoms=atoms_new):
            return self._reject_move_proposal("site_region")
        if not self._group_positions_are_valid(
            group,
            trial_positions,
            atoms=atoms_new,
        ):
            return self._reject_move_proposal("proposal_geometry")
        if not self._group_clears_terminations(
            group,
            trial_positions,
            atoms=atoms_new,
        ):
            return self._reject_move_proposal("termination_clearance")
        if self.enforce_molecular_integrity and not (
            self._molecular_adsorbates_are_intact(atoms_new)
        ):
            return self._reject_move_proposal("molecular_integrity")
        return atoms_new

    def _propose_move(self) -> Optional[Atoms]:
        self._last_proposal_move_name = None
        self._last_proposal_reject_reason = None
        self._last_proposal_metadata = {}
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
