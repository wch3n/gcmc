import logging
import os
import pickle
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
from ase import Atoms
from ase.geometry import get_distances
from ase.io import Trajectory

from .adsorbate_cmc import AdsorbateCMC, _load_adsorbate_template
from .constants import ADSORBATE_TAG_OFFSET, KB_EV_PER_K
from .utils import place_adsorbate_on_site

logger = logging.getLogger("mc")


class AdsorbateGCMC(AdsorbateCMC):
    """
    Site-based grand-canonical MC for rigid adsorbates.

    Current scope:
    - rigid adsorbates with a designated anchor atom
    - insertion/deletion on the instantaneous high-symmetry site registry
    - canonical within-N moves reuse ``AdsorbateCMC``
    - optional hybrid MD/local relaxation reuse ``AdsorbateCMC``

    Grand-canonical acceptance uses the site-based discrete proposal ratio.
    If an insertion is proposed uniformly on empty sites and a deletion is
    proposed uniformly on occupied adsorbates, then

      acc(ins) = min(1, exp[-beta (dE - mu)] * q_rev / q_fwd)
      acc(del) = min(1, exp[-beta (dE + mu)] * q_rev / q_fwd)

    where ``q_rev / q_fwd`` includes the move-type and site-selection terms.
    """

    def __init__(
        self,
        atoms: Union[Atoms, str],
        calculator: Any,
        mu: float,
        T: float = 300.0,
        nsteps: int = 1000,
        w_insert: float = 1.0,
        w_delete: float = 1.0,
        w_canonical: float = 1.0,
        max_n_adsorbates: Optional[int] = None,
        repeat: Sequence[int] = (1, 1, 1),
        supercell_matrix: Optional[Sequence[Sequence[int]]] = None,
        element: Optional[str] = None,
        adsorbate_element: str = "H",
        adsorbate: Optional[Union[str, Atoms]] = None,
        traj_file: str = "adsorbate_gcmc.traj",
        thermo_file: str = "adsorbate_gcmc.dat",
        checkpoint_file: str = "adsorbate_gcmc.pkl",
        checkpoint_interval: int = 100,
        seed: int = 81,
        resume: bool = False,
        allow_ambiguous_empty_adsorbates: bool = False,
        md_without_adsorbate: bool = False,
        **kwargs,
    ):
        if element is not None:
            if adsorbate_element != "H" and adsorbate_element != element:
                raise ValueError(
                    "Provide either element or adsorbate_element, not conflicting values."
                )
            adsorbate_element = element
        if adsorbate is None:
            template_size = 1
        else:
            template_size = len(_load_adsorbate_template(adsorbate, adsorbate_element))
        if template_size < 1:
            raise ValueError("adsorbate template must contain at least one atom.")

        self.mu = float(mu)
        self.nsteps = int(nsteps)
        self.w_insert = float(w_insert)
        self.w_delete = float(w_delete)
        self.w_canonical = float(w_canonical)
        self.max_n_adsorbates = max_n_adsorbates
        self.attempted_insertions = 0
        self.accepted_insertions = 0
        self.attempted_deletions = 0
        self.accepted_deletions = 0
        self.attempted_canonical = 0
        self.accepted_canonical = 0
        self.sum_N = 0.0
        self.sum_N_sq = 0.0
        self.n_hist: Dict[int, int] = {}
        self._eligible_site_rows: Optional[list[dict[str, object]]] = None
        self._eligible_site_xy: Optional[np.ndarray] = None
        self._eligible_site_z: Optional[np.ndarray] = None
        self._single_site_occupancy_fast_path = max_n_adsorbates == 1
        self.md_without_adsorbate = bool(md_without_adsorbate)

        super().__init__(
            atoms=atoms,
            calculator=calculator,
            T=T,
            adsorbate_element=adsorbate_element,
            adsorbate=adsorbate,
            repeat=repeat,
            supercell_matrix=supercell_matrix,
            traj_file=traj_file,
            thermo_file=thermo_file,
            checkpoint_file=checkpoint_file,
            checkpoint_interval=checkpoint_interval,
            seed=seed,
            resume=resume,
            allow_ambiguous_empty_adsorbates=allow_ambiguous_empty_adsorbates,
            **kwargs,
        )

        if self.w_insert < 0.0 or self.w_delete < 0.0 or self.w_canonical < 0.0:
            raise ValueError("Move weights must be >= 0.")
        if (self.w_insert > 0.0) != (self.w_delete > 0.0):
            raise ValueError(
                "w_insert and w_delete must either both be > 0 or both be 0. "
                "A zero reverse-move weight breaks detailed balance."
            )
        if self.max_n_adsorbates is not None and int(self.max_n_adsorbates) < 0:
            raise ValueError("max_n_adsorbates must be >= 0 when provided.")

    def _invalidate_gc_site_cache(self, clear_registry: bool = False) -> None:
        self._eligible_site_rows = None
        self._eligible_site_xy = None
        self._eligible_site_z = None
        if clear_registry:
            self._site_registry = None

    def _refresh_cached_state(self) -> None:
        self._invalidate_gc_site_cache(clear_registry=True)
        self._update_indices()

    def _refresh_after_adsorbate_move(self, invalidate_site_registry: bool) -> None:
        if invalidate_site_registry:
            self._refresh_cached_state()
        else:
            # Adsorbate-only moves do not change the slab-derived site registry.
            self._update_indices()

    def _save_checkpoint(self):
        atoms_copy = self.atoms.copy()
        atoms_copy.calc = None
        state = {
            "atoms": atoms_copy,
            "sweep": self.sweep,
            "e_old": self.e_old,
            "T": self.T,
            "mu": self.mu,
            "rng_state": self.rng.bit_generator.state,
            "sum_E": self.sum_E,
            "sum_E_sq": self.sum_E_sq,
            "sum_N": self.sum_N,
            "sum_N_sq": self.sum_N_sq,
            "n_samples": self.n_samples,
            "n_hist": dict(self.n_hist),
            "accepted_moves": self.accepted_moves,
            "total_moves": self.total_moves,
            "md_attempted_moves": self.md_attempted_moves,
            "md_accepted_moves": self.md_accepted_moves,
            "attempted_insertions": self.attempted_insertions,
            "accepted_insertions": self.accepted_insertions,
            "attempted_deletions": self.attempted_deletions,
            "accepted_deletions": self.accepted_deletions,
            "attempted_canonical": self.attempted_canonical,
            "accepted_canonical": self.accepted_canonical,
        }
        with open(self.checkpoint_file, "wb") as handle:
            pickle.dump(state, handle)

    def _load_checkpoint(self):
        if not os.path.exists(self.checkpoint_file):
            return
        with open(self.checkpoint_file, "rb") as handle:
            state = pickle.load(handle)
        if "atoms" in state:
            self.atoms = state["atoms"]
            self.atoms.calc = self.calculator
        self.sweep = state.get("sweep", 0)
        self.e_old = state.get("e_old", self.e_old)
        self.T = state.get("T", self.T)
        self.mu = state.get("mu", self.mu)
        rng_state = state.get("rng_state")
        if rng_state is not None:
            self.rng.bit_generator.state = rng_state
        self.sum_E = state.get("sum_E", 0.0)
        self.sum_E_sq = state.get("sum_E_sq", 0.0)
        self.sum_N = state.get("sum_N", 0.0)
        self.sum_N_sq = state.get("sum_N_sq", 0.0)
        self.n_samples = state.get("n_samples", 0)
        self.n_hist = {int(k): int(v) for k, v in state.get("n_hist", {}).items()}
        self.accepted_moves = state.get("accepted_moves", 0)
        self.total_moves = state.get("total_moves", 0)
        self.md_attempted_moves = state.get("md_attempted_moves", 0)
        self.md_accepted_moves = state.get("md_accepted_moves", 0)
        self.attempted_insertions = state.get("attempted_insertions", 0)
        self.accepted_insertions = state.get("accepted_insertions", 0)
        self.attempted_deletions = state.get("attempted_deletions", 0)
        self.accepted_deletions = state.get("accepted_deletions", 0)
        self.attempted_canonical = state.get("attempted_canonical", 0)
        self.accepted_canonical = state.get("accepted_canonical", 0)
        self._refresh_cached_state()
        logger.info(f"[{self.T:.0f}K] Resumed adsorbate GCMC from checkpoint.")

    def _eligible_site_data(
        self,
    ) -> tuple[list[dict[str, object]], np.ndarray, np.ndarray]:
        if self._eligible_site_rows is None:
            rows: list[dict[str, object]] = []
            xy = []
            z = []
            for row in self._get_site_registry():
                z_val = float(row.get("suggested_z_A", np.nan))
                if not np.isfinite(z_val):
                    continue
                if bool(row.get("blocked_by_termination", False)):
                    continue
                rows.append(row)
                xy.append(np.asarray(row["xy"], dtype=float))
                z.append(z_val)

            self._eligible_site_rows = rows
            if rows:
                self._eligible_site_xy = np.asarray(xy, dtype=float)
                self._eligible_site_z = np.asarray(z, dtype=float)
            else:
                self._eligible_site_xy = np.empty((0, 2), dtype=float)
                self._eligible_site_z = np.empty((0,), dtype=float)

        return (
            self._eligible_site_rows,
            self._eligible_site_xy,
            self._eligible_site_z,
        )

    def _eligible_sites(self) -> list[dict[str, object]]:
        return self._eligible_site_data()[0]

    def _eligible_site_count(self) -> int:
        return len(self._eligible_site_data()[0])

    def _assign_groups_to_sites(
        self,
        candidate_sites: Optional[Sequence[Dict[str, object]]] = None,
    ) -> Dict[int, int]:
        if candidate_sites is None:
            candidate_sites, site_xy, _ = self._eligible_site_data()
        else:
            site_xy = np.asarray(
                [np.asarray(site["xy"], dtype=float) for site in candidate_sites],
                dtype=float,
            )
        if not candidate_sites or not self.ads_groups:
            return {}

        cell = self.atoms.get_cell()
        pbc = self.atoms.get_pbc()
        anchor_xy = self.atoms.positions[np.asarray(self.ads_anchor_indices, dtype=int), :2]
        anchor_xyz = np.zeros((len(anchor_xy), 3), dtype=float)
        site_xyz = np.zeros((len(site_xy), 3), dtype=float)
        anchor_xyz[:, :2] = anchor_xy
        site_xyz[:, :2] = site_xy
        deltas = get_distances(anchor_xyz, site_xyz, cell=cell, pbc=pbc)[0]
        pair_distances = np.linalg.norm(deltas[:, :, :2], axis=2)
        order = np.argsort(pair_distances, axis=None)
        assigned_groups = set()
        assigned_sites = set()
        mapping: Dict[int, int] = {}
        n_sites = len(candidate_sites)
        for flat_idx in order.tolist():
            group_id = int(flat_idx // n_sites)
            site_id = int(flat_idx % n_sites)
            if group_id in assigned_groups or site_id in assigned_sites:
                continue
            mapping[group_id] = site_id
            assigned_groups.add(group_id)
            assigned_sites.add(site_id)
            if len(assigned_groups) == len(self.ads_groups):
                break
        return mapping

    def _move_probabilities(self, n_ads: int, n_sites: int) -> Dict[str, float]:
        weights: Dict[str, float] = {}
        max_n = n_sites if self.max_n_adsorbates is None else min(n_sites, int(self.max_n_adsorbates))
        if self.w_insert > 0.0 and n_ads < max_n and n_sites > n_ads:
            weights["insert"] = self.w_insert
        if self.w_delete > 0.0 and n_ads > 0:
            weights["delete"] = self.w_delete
        if self.w_canonical > 0.0 and n_ads > 0:
            weights["canonical"] = self.w_canonical
        total = float(sum(weights.values()))
        if total <= 0.0:
            return {}
        return {key: value / total for key, value in weights.items()}

    def _next_group_tag(self, atoms: Atoms) -> int:
        tags = np.asarray(atoms.get_tags(), dtype=int)
        if len(tags) != len(atoms):
            return ADSORBATE_TAG_OFFSET
        tagged = tags[tags >= ADSORBATE_TAG_OFFSET]
        if tagged.size == 0:
            return ADSORBATE_TAG_OFFSET
        return int(np.max(tagged)) + 1

    def _propose_insertion(self):
        candidate_sites = self._eligible_sites()
        n_sites = len(candidate_sites)
        if n_sites == 0:
            return None

        n_ads = len(self.ads_groups)
        if self.max_n_adsorbates is not None and n_ads >= int(self.max_n_adsorbates):
            return None
        if self._single_site_occupancy_fast_path:
            if n_ads != 0:
                return None
            empty_site_ids = None
        else:
            # Greedy nearest-pair matching is sufficient for the dilute site occupancies
            # we target here. If dense coverages become important, replace this with an
            # exact assignment algorithm before relying on n_empty_before.
            occupancy_map = self._assign_groups_to_sites(candidate_sites)
            occupied_sites = set(occupancy_map.values())
            empty_site_ids = [i for i in range(n_sites) if i not in occupied_sites]
            if not empty_site_ids:
                return None

        if empty_site_ids is None:
            site_id = int(self.rng.integers(n_sites))
            n_empty_before = n_sites
        else:
            site_id = int(self.rng.choice(empty_site_ids))
            n_empty_before = len(empty_site_ids)
        site = candidate_sites[site_id]
        atoms_trial, _ = place_adsorbate_on_site(
            self.atoms,
            self.adsorbate_template,
            site,
            anchor_index=self.adsorbate_anchor_index,
        )
        group = np.arange(len(self.atoms), len(atoms_trial), dtype=int)
        tags = np.asarray(atoms_trial.get_tags(), dtype=int)
        if len(tags) != len(atoms_trial):
            tags = np.zeros(len(atoms_trial), dtype=int)
        tags[group] = self._next_group_tag(self.atoms)
        atoms_trial.set_tags(tags)
        trial_positions = atoms_trial.get_positions()[group]
        if not self._group_positions_are_valid(group, trial_positions, atoms=atoms_trial):
            return None
        if not self._group_clears_terminations(group, trial_positions, atoms=atoms_trial):
            return None
        return atoms_trial, {
            "n_sites": n_sites,
            "n_ads": n_ads,
            "n_empty_before": n_empty_before,
        }

    def _propose_deletion(self):
        n_ads = len(self.ads_groups)
        if n_ads == 0:
            return None
        candidate_sites = self._eligible_sites()
        n_sites = len(candidate_sites)
        if n_sites == 0:
            return None

        if self._single_site_occupancy_fast_path:
            if n_ads < 1:
                return None
            group_id = 0 if n_ads == 1 else int(self.rng.integers(n_ads))
        else:
            group_id = int(self.rng.integers(n_ads))
        group = np.asarray(self.ads_groups[group_id], dtype=int)
        atoms_trial = self.atoms.copy()
        for idx in np.sort(group)[::-1]:
            del atoms_trial[int(idx)]
        return atoms_trial, {
            "n_sites": n_sites,
            "n_ads": n_ads,
            "n_empty_after": n_sites - n_ads + 1,
        }

    def _grand_accept(self, delta_e: float, delta_n: int, log_q_ratio: float, beta: float) -> bool:
        log_weight = -beta * (delta_e - self.mu * float(delta_n)) + float(log_q_ratio)
        if log_weight >= 0.0:
            return True
        return self.rng.random() < np.exp(log_weight)

    def _insertion_log_q_ratio(self, info: Dict[str, int]) -> float:
        n_ads_new = int(info["n_ads"]) + 1
        probs_old = self._move_probabilities(int(info["n_ads"]), int(info["n_sites"]))
        probs_new = self._move_probabilities(n_ads_new, int(info["n_sites"]))
        if "insert" not in probs_old or "delete" not in probs_new:
            return -np.inf
        return (
            np.log(probs_new["delete"])
            - np.log(probs_old["insert"])
            + np.log(int(info["n_empty_before"]))
            - np.log(n_ads_new)
        )

    def _deletion_log_q_ratio(self, info: Dict[str, int]) -> float:
        n_ads_old = int(info["n_ads"])
        n_ads_new = n_ads_old - 1
        probs_old = self._move_probabilities(n_ads_old, int(info["n_sites"]))
        probs_new = self._move_probabilities(n_ads_new, int(info["n_sites"]))
        if "delete" not in probs_old or "insert" not in probs_new:
            return -np.inf
        return (
            np.log(probs_new["insert"])
            - np.log(probs_old["delete"])
            + np.log(n_ads_old)
            - np.log(int(info["n_empty_after"]))
        )

    def _moves_per_sweep(self) -> int:
        return self._eligible_site_count()

    def run(
        self,
        nsweeps: Optional[int] = None,
        traj_file: Optional[str] = None,
        interval: int = 10,
        sample_interval: int = 1,
        equilibration: int = 0,
        max_moves: Optional[int] = None,
        log_every: Optional[int] = None,
        progress_interval_moves: Optional[int] = None,
    ) -> Dict[str, float]:
        if nsweeps is None:
            nsweeps = self.nsteps
        if traj_file is None:
            traj_file = self.traj_file
        if log_every is not None:
            interval = int(log_every)

        self.traj_file = traj_file
        mode = "a" if os.path.exists(self.traj_file) and os.path.getsize(self.traj_file) > 0 else "w"
        traj_writer = Trajectory(self.traj_file, mode)
        accepted_writer = self._open_optional_traj(self.accepted_traj_file)
        rejected_writer = self._open_optional_traj(self.rejected_traj_file)
        attempted_writer = self._open_optional_traj(self.attempted_traj_file)

        self.sum_E = 0.0
        self.sum_E_sq = 0.0
        self.sum_N = 0.0
        self.sum_N_sq = 0.0
        self.n_samples = 0
        self.n_hist = {}
        self.accepted_moves = 0
        self.total_moves = 0
        self.md_attempted_moves = 0
        self.md_accepted_moves = 0
        self.attempted_insertions = 0
        self.accepted_insertions = 0
        self.attempted_deletions = 0
        self.accepted_deletions = 0
        self.attempted_canonical = 0
        self.accepted_canonical = 0
        progress_every = (
            None
            if progress_interval_moves is None or int(progress_interval_moves) <= 0
            else int(progress_interval_moves)
        )

        logger.info(
            f"Start adsorbate GCMC | T={self.T:.0f}K | nsweeps={int(nsweeps)} | "
            f"eligible_sites={self._eligible_site_count()} | moves_per_sweep={self._moves_per_sweep()} | "
            f"write_interval={int(interval)} | sample_interval={int(sample_interval)} | "
            f"equilibration={int(equilibration)}"
            + (
                ""
                if progress_every is None
                else f" | progress_interval_moves={progress_every}"
            )
        )

        for sweep in range(int(nsweeps)):
            beta = 1.0 / (KB_EV_PER_K * self.T)
            moves_this_sweep = self._moves_per_sweep()
            if max_moves is not None:
                moves_this_sweep = min(int(max_moves), moves_this_sweep)

            for move_idx in range(moves_this_sweep):
                self.total_moves += 1
                try:
                    n_ads = len(self.ads_groups)
                    do_md = (
                        self.enable_hybrid_md
                        and (n_ads > 0 or self.md_without_adsorbate)
                        and self.rng.random() < self.md_move_prob
                    )
                    if do_md:
                        self.md_attempted_moves += 1
                        atoms_trial, delta_e, delta_h = self._propose_md_move()
                        if atoms_trial is None:
                            continue
                        if attempted_writer is not None:
                            attempted_writer.write(atoms_trial)
                        if self.has_detached_functional_groups(
                            atoms_trial, detach_tol=self.detach_tol
                        ):
                            if rejected_writer is not None:
                                rejected_writer.write(atoms_trial)
                            continue
                        if self.has_afloat_adsorbates(
                            atoms_trial,
                            support_xy_tol=self.support_xy_tol,
                            z_max_support=self.z_max_support,
                        ):
                            if rejected_writer is not None:
                                rejected_writer.write(atoms_trial)
                            continue
                        md_delta = (
                            delta_h if self.md_accept_mode == "hamiltonian" else delta_e
                        )
                        if self._metropolis_accept(md_delta, beta=beta):
                            self.e_old += delta_e
                            self.accepted_moves += 1
                            self.md_accepted_moves += 1
                            self.atoms.positions = atoms_trial.positions
                            self.atoms.cell = atoms_trial.cell
                            self._invalidate_gc_site_cache(clear_registry=True)
                            if accepted_writer is not None:
                                accepted_writer.write(self.atoms)
                        elif rejected_writer is not None:
                            rejected_writer.write(atoms_trial)
                        continue

                    move_probs = self._move_probabilities(
                        len(self.ads_groups), self._eligible_site_count()
                    )
                    if not move_probs:
                        continue
                    move_types = tuple(move_probs.keys())
                    move_weights = np.asarray(
                        [move_probs[key] for key in move_types], dtype=float
                    )
                    move_type = str(
                        self.rng.choice(
                            move_types, p=move_weights / move_weights.sum()
                        )
                    )

                    if move_type == "insert":
                        self.attempted_insertions += 1
                        proposal = self._propose_insertion()
                        if proposal is None:
                            continue
                        atoms_trial, proposal_info = proposal
                        if attempted_writer is not None:
                            attempted_writer.write(atoms_trial)
                        if self.relax:
                            atoms_trial, converged = self.relax_structure(
                                atoms_trial, move_ind=[self.sweep, move_idx]
                            )
                            if not converged:
                                continue
                            if self.has_detached_functional_groups(
                                atoms_trial, detach_tol=self.detach_tol
                            ):
                                if rejected_writer is not None:
                                    rejected_writer.write(atoms_trial)
                                continue
                        if self.has_afloat_adsorbates(
                            atoms_trial,
                            support_xy_tol=self.support_xy_tol,
                            z_max_support=self.z_max_support,
                        ):
                            if rejected_writer is not None:
                                rejected_writer.write(atoms_trial)
                            continue
                        e_new = self.get_potential_energy(atoms_trial)
                        delta_e = e_new - self.e_old
                        log_q_ratio = self._insertion_log_q_ratio(proposal_info)
                        if self._grand_accept(
                            delta_e, delta_n=1, log_q_ratio=log_q_ratio, beta=beta
                        ):
                            self.atoms = atoms_trial
                            self.atoms.calc = self.calculator
                            self.e_old = e_new
                            self.accepted_moves += 1
                            self.accepted_insertions += 1
                            self._refresh_after_adsorbate_move(
                                invalidate_site_registry=self.relax
                            )
                            if accepted_writer is not None:
                                accepted_writer.write(self.atoms)
                        elif rejected_writer is not None:
                            rejected_writer.write(atoms_trial)
                        continue

                    if move_type == "delete":
                        self.attempted_deletions += 1
                        proposal = self._propose_deletion()
                        if proposal is None:
                            continue
                        atoms_trial, proposal_info = proposal
                        if attempted_writer is not None:
                            attempted_writer.write(atoms_trial)
                        if self.relax:
                            atoms_trial, converged = self.relax_structure(
                                atoms_trial, move_ind=[self.sweep, move_idx]
                            )
                            if not converged:
                                continue
                            if self.has_detached_functional_groups(
                                atoms_trial, detach_tol=self.detach_tol
                            ):
                                if rejected_writer is not None:
                                    rejected_writer.write(atoms_trial)
                                continue
                        # A deletion that removes the last adsorbate leaves a clean slab.
                        # In that case there is no adsorbate support geometry to validate.
                        if int(proposal_info["n_ads"]) > 1:
                            if self.has_afloat_adsorbates(
                                atoms_trial,
                                support_xy_tol=self.support_xy_tol,
                                z_max_support=self.z_max_support,
                            ):
                                if rejected_writer is not None:
                                    rejected_writer.write(atoms_trial)
                                continue
                        e_new = self.get_potential_energy(atoms_trial)
                        delta_e = e_new - self.e_old
                        log_q_ratio = self._deletion_log_q_ratio(proposal_info)
                        if self._grand_accept(
                            delta_e, delta_n=-1, log_q_ratio=log_q_ratio, beta=beta
                        ):
                            self.atoms = atoms_trial
                            self.atoms.calc = self.calculator
                            self.e_old = e_new
                            self.accepted_moves += 1
                            self.accepted_deletions += 1
                            self._refresh_after_adsorbate_move(
                                invalidate_site_registry=self.relax
                            )
                            if accepted_writer is not None:
                                accepted_writer.write(self.atoms)
                        elif rejected_writer is not None:
                            rejected_writer.write(atoms_trial)
                        continue

                    self.attempted_canonical += 1
                    atoms_trial = self._propose_move()
                    if atoms_trial is None:
                        continue
                    if attempted_writer is not None:
                        attempted_writer.write(atoms_trial)
                    if self.relax:
                        atoms_trial, converged = self.relax_structure(
                            atoms_trial, move_ind=[self.sweep, move_idx]
                        )
                        if not converged:
                            continue
                        if self.has_detached_functional_groups(
                            atoms_trial, detach_tol=self.detach_tol
                        ):
                            if rejected_writer is not None:
                                rejected_writer.write(atoms_trial)
                            continue
                    if self.has_afloat_adsorbates(
                        atoms_trial,
                        support_xy_tol=self.support_xy_tol,
                        z_max_support=self.z_max_support,
                    ):
                        if rejected_writer is not None:
                            rejected_writer.write(atoms_trial)
                        continue
                    e_new = self.get_potential_energy(atoms_trial)
                    delta_e = e_new - self.e_old
                    if self._metropolis_accept(delta_e, beta=beta):
                        self.atoms = atoms_trial
                        self.atoms.calc = self.calculator
                        self.e_old = e_new
                        self.accepted_moves += 1
                        self.accepted_canonical += 1
                        self._refresh_after_adsorbate_move(
                            invalidate_site_registry=self.relax
                        )
                        if accepted_writer is not None:
                            accepted_writer.write(self.atoms)
                    elif rejected_writer is not None:
                        rejected_writer.write(atoms_trial)
                finally:
                    if (
                        progress_every is not None
                        and (move_idx + 1) % progress_every == 0
                        and (move_idx + 1) < moves_this_sweep
                    ):
                        acc = (
                            self.accepted_moves / self.total_moves * 100.0
                            if self.total_moves
                            else 0.0
                        )
                        logger.info(
                            f"T={self.T:4.0f}K | sweep {sweep + 1:6d}/{int(nsweeps):6d} | "
                            f"move {move_idx + 1:4d}/{moves_this_sweep:4d} | "
                            f"E: {self.e_old:10.4f} | Nads: {len(self.ads_groups):4d} | "
                            f"Acc: {acc:4.1f}%"
                        )

            self.sweep += 1

            if sweep >= equilibration and (sweep + 1) % sample_interval == 0:
                n_ads = len(self.ads_groups)
                self.sum_E += self.e_old
                self.sum_E_sq += self.e_old**2
                self.sum_N += n_ads
                self.sum_N_sq += n_ads**2
                self.n_samples += 1
                self.n_hist[n_ads] = self.n_hist.get(n_ads, 0) + 1

            if (sweep + 1) % interval == 0:
                traj_writer.write(self.atoms)
                with open(self.thermo_file, "a") as handle:
                    handle.write(f"{self.sweep} {self.e_old:.6f} {len(self.ads_groups)}\n")

                acc = (self.accepted_moves / self.total_moves * 100.0) if self.total_moves else 0.0
                avg_e = self.sum_E / self.n_samples if self.n_samples else self.e_old
                avg_n = self.sum_N / self.n_samples if self.n_samples else len(self.ads_groups)
                cv = 0.0
                if self.n_samples > 1:
                    var_e = (self.sum_E_sq / self.n_samples) - (avg_e**2)
                    cv = var_e / (KB_EV_PER_K * self.T**2)

                logger.info(
                    f"T={self.T:4.0f}K | {self.sweep:6d} | E: {self.e_old:10.4f} | "
                    f"AvgE: {avg_e:10.4f} | AvgN: {avg_n:6.2f} | Cv: {cv:8.4f} | "
                    f"Acc: {acc:4.1f}% | Nads: {len(self.ads_groups):4d} | "
                    f"Ins: {self.accepted_insertions}/{self.attempted_insertions} | "
                    f"Del: {self.accepted_deletions}/{self.attempted_deletions} | "
                    f"Can: {self.accepted_canonical}/{self.attempted_canonical}"
                    + (
                        ""
                        if not self.enable_hybrid_md
                        else (
                            f" | MD: {self.md_accepted_moves}/{self.md_attempted_moves}"
                            f" ({((self.md_accepted_moves / self.md_attempted_moves) * 100.0) if self.md_attempted_moves else 0.0:4.1f}%)"
                            f" | MD_frac: {((self.md_attempted_moves / self.total_moves) * 100.0) if self.total_moves else 0.0:4.1f}%"
                            f" | MD_accept: {self.md_accept_mode}"
                            f" | planar: {self.md_planar}"
                        )
                    )
                )

            if self.checkpoint_interval > 0 and self.sweep % self.checkpoint_interval == 0:
                self._save_checkpoint()

        self._save_checkpoint()
        traj_writer.close()
        for writer in (accepted_writer, rejected_writer, attempted_writer):
            if writer is not None:
                writer.close()

        final_avg_e = self.sum_E / self.n_samples if self.n_samples else self.e_old
        final_avg_n = self.sum_N / self.n_samples if self.n_samples else len(self.ads_groups)
        final_cv = 0.0
        if self.n_samples > 1:
            var_e = (self.sum_E_sq / self.n_samples) - (final_avg_e**2)
            final_cv = var_e / (KB_EV_PER_K * self.T**2)

        return {
            "T": self.T,
            "mu": self.mu,
            "energy": final_avg_e,
            "cv": final_cv,
            "acceptance": (
                (self.accepted_moves / self.total_moves * 100.0)
                if self.total_moves
                else 0.0
            ),
            "n_adsorbates": len(self.ads_groups),
            "n_adsorbates_avg": final_avg_n,
            "n_hist": dict(self.n_hist),
            "insert_attempted": self.attempted_insertions,
            "insert_accepted": self.accepted_insertions,
            "delete_attempted": self.attempted_deletions,
            "delete_accepted": self.accepted_deletions,
            "canonical_attempted": self.attempted_canonical,
            "canonical_accepted": self.accepted_canonical,
            "md_attempted": self.md_attempted_moves,
            "md_accepted": self.md_accepted_moves,
            "md_accept_mode": self.md_accept_mode,
        }
