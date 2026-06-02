"""Local adsorbate CMC basin discovery for reaction-state candidates."""

from __future__ import annotations

import csv
import os
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Sequence

import numpy as np
from ase import Atoms
from ase.io import read, write

from gcmc.adsorbate_cmc import AdsorbateCMC
from gcmc.constants import ADSORBATE_TAG_OFFSET
from gcmc.move_config import normalize_adsorbate_move_config
from gcmc.replica import ReplicaExchange
from gcmc.workflows import build_adsorbate_gcmc_calculator, build_replica_calculator_spec

from .output_files import output_path
from .states import ReactionCandidateGenerator


_DEFAULT_LOCAL_CMC_CONFIG = {
    "enabled": False,
    "states": ["02_O", "03_OOH"],
    "center_state": "01_OH",
    "radius_A": 3.0,
    "distance_metric": "xy",
    "temperature_K": 303.0,
    "pt_enabled": False,
    "temperatures_K": None,
    "target_temperature_K": None,
    "swap_interval": 10,
    "swap_stride": 1,
    "pt_n_cycles": None,
    "pt_equilibration_cycles": None,
    "pt_local_eq_fraction": 0.2,
    "backend": "multiprocessing",
    "n_gpus": None,
    "workers_per_gpu": 1,
    "ray_address": "auto",
    "ray_num_cpus_per_task": 1,
    "ray_num_gpus_per_task": None,
    "ray_log_to_driver": False,
    "ray_actor_max_restarts": 0,
    "ray_actor_max_task_retries": 0,
    "ray_get_timeout_s": None,
    "use_placement_group": False,
    "placement_group_strategy": "SPREAD",
    "remove_placement_group_on_stop": True,
    "shutdown_on_stop": False,
    "n_cycles": 500,
    "sample_interval": 25,
    "equilibration_cycles": 100,
    "max_seed_candidates_per_state": 1,
    "max_output_candidates_per_state": 50,
    "output_selection": "diverse",
    "replace_candidates": True,
    "skip_existing": False,
    "sequential_enabled": None,
    "sequential_rules": None,
    "sequential_ooh_from_o": False,
    "sequential_ooh_orientations": 1,
    "move_mode": "hybrid",
    "site_hop_prob": 0.6,
    "reorientation_prob": 0.3,
    "hop_reorientation_prob": 0.0,
    "hop_puckering_prob": 0.0,
    "hop_puckering_reorientation_prob": 0.0,
    "puckering_prob": 0.0,
    "puckering_hop_prob": 0.0,
    "puckering_elements": None,
    "puckering_height_A": 0.15,
    "puckering_height_jitter_A": None,
    "displacement_sigma": 0.2,
    "max_displacement_trials": 20,
    "max_hop_reorientation_trials": None,
    "hop_reorientation_angle_deg": 180.0,
    "adsorbate_surface_clearance_A": 0.0,
    "adsorbate_surface_xy_tol_A": None,
    "max_puckering_trials": None,
    "rotation_max_angle_deg": 35.0,
    "relax": False,
    "relax_steps": 20,
    "fmax": 0.05,
    "enable_hybrid_md": False,
    "md_move_prob": 0.05,
    "md_steps": 20,
    "md_timestep_fs": 0.5,
    "md_ensemble": "nve",
    "md_accept_mode": "potential",
    "checkpoint_interval": 0,
    "write_debug_trajs": False,
    "write_attempted_traj": False,
    "write_accepted_traj": False,
    "write_rejected_traj": False,
    "debug_traj_interval": 1,
    "progress_log": None,
    "progress_stdout": True,
    "calculator": None,
    "model": None,
    "model_file": None,
    "device": None,
    "use_kokkos": None,
}


def default_local_cmc_config() -> dict[str, object]:
    return dict(_DEFAULT_LOCAL_CMC_CONFIG)


class ReactionLocalCMCWorkflow:
    """Replace or augment O*/OOH* candidates using local adsorbate CMC."""

    def __init__(self, config: SimpleNamespace):
        self.config = config
        raw = getattr(config, "local_cmc", {}) or {}
        merged = default_local_cmc_config()
        if isinstance(raw, dict):
            merged.update(raw)
            self._apply_nested_local_config(merged, raw)
        self.local_config = merged

    @classmethod
    def _apply_nested_local_config(
        cls,
        merged: dict[str, object],
        raw: dict[str, object],
    ) -> None:
        section_maps: dict[str, dict[str, str]] = {
            "region": {
                "center_state": "center_state",
                "radius_A": "radius_A",
                "distance_metric": "distance_metric",
            },
            "sampling": {
                "temperature_K": "temperature_K",
                "n_cycles": "n_cycles",
                "sample_interval": "sample_interval",
                "equilibration_cycles": "equilibration_cycles",
                "max_seed_candidates_per_state": "max_seed_candidates_per_state",
                "max_output_candidates_per_state": "max_output_candidates_per_state",
                "output_selection": "output_selection",
                "replace_candidates": "replace_candidates",
                "skip_existing": "skip_existing",
                "sequential_ooh_from_o": "sequential_ooh_from_o",
                "sequential_ooh_orientations": "sequential_ooh_orientations",
            },
            "sequential": {
                "enabled": "sequential_enabled",
                "rules": "sequential_rules",
            },
            "pt": {
                "enabled": "pt_enabled",
                "temperatures_K": "temperatures_K",
                "target_temperature_K": "target_temperature_K",
                "swap_interval": "swap_interval",
                "swap_stride": "swap_stride",
                "n_cycles": "pt_n_cycles",
                "equilibration_cycles": "pt_equilibration_cycles",
                "local_eq_fraction": "pt_local_eq_fraction",
            },
            "backend_config": {
                "backend": "backend",
                "n_gpus": "n_gpus",
                "workers_per_gpu": "workers_per_gpu",
                "ray_address": "ray_address",
                "ray_num_cpus_per_task": "ray_num_cpus_per_task",
                "ray_num_gpus_per_task": "ray_num_gpus_per_task",
                "ray_log_to_driver": "ray_log_to_driver",
                "ray_actor_max_restarts": "ray_actor_max_restarts",
                "ray_actor_max_task_retries": "ray_actor_max_task_retries",
                "ray_get_timeout_s": "ray_get_timeout_s",
                "use_placement_group": "use_placement_group",
                "placement_group_strategy": "placement_group_strategy",
                "remove_placement_group_on_stop": "remove_placement_group_on_stop",
                "shutdown_on_stop": "shutdown_on_stop",
            },
            "moves": {
                "mode": "move_mode",
                "move_mode": "move_mode",
                "site_hop_prob": "site_hop_prob",
                "reorientation_prob": "reorientation_prob",
                "hop_reorientation_prob": "hop_reorientation_prob",
                "hop_puckering_prob": "hop_puckering_prob",
                "hop_puckering_reorientation_prob": "hop_puckering_reorientation_prob",
                "displacement_sigma": "displacement_sigma",
                "max_displacement_trials": "max_displacement_trials",
                "max_hop_reorientation_trials": "max_hop_reorientation_trials",
                "hop_reorientation_angle_deg": "hop_reorientation_angle_deg",
                "surface_clearance_A": "adsorbate_surface_clearance_A",
                "adsorbate_surface_clearance_A": "adsorbate_surface_clearance_A",
                "surface_xy_tol_A": "adsorbate_surface_xy_tol_A",
                "adsorbate_surface_xy_tol_A": "adsorbate_surface_xy_tol_A",
                "rotation_max_angle_deg": "rotation_max_angle_deg",
            },
            "puckering": {
                "prob": "puckering_prob",
                "puckering_prob": "puckering_prob",
                "hop_prob": "puckering_hop_prob",
                "puckering_hop_prob": "puckering_hop_prob",
                "elements": "puckering_elements",
                "puckering_elements": "puckering_elements",
                "height_A": "puckering_height_A",
                "puckering_height_A": "puckering_height_A",
                "height_jitter_A": "puckering_height_jitter_A",
                "puckering_height_jitter_A": "puckering_height_jitter_A",
                "max_trials": "max_puckering_trials",
                "max_puckering_trials": "max_puckering_trials",
            },
            "relaxation": {
                "enabled": "relax",
                "relax": "relax",
                "steps": "relax_steps",
                "relax_steps": "relax_steps",
                "fmax": "fmax",
            },
            "md": {
                "enabled": "enable_hybrid_md",
                "enable_hybrid_md": "enable_hybrid_md",
                "move_prob": "md_move_prob",
                "md_move_prob": "md_move_prob",
                "steps": "md_steps",
                "md_steps": "md_steps",
                "timestep_fs": "md_timestep_fs",
                "md_timestep_fs": "md_timestep_fs",
                "ensemble": "md_ensemble",
                "md_ensemble": "md_ensemble",
                "accept_mode": "md_accept_mode",
                "md_accept_mode": "md_accept_mode",
            },
            "output": {
                "checkpoint_interval": "checkpoint_interval",
                "write_debug_trajs": "write_debug_trajs",
                "write_attempted_traj": "write_attempted_traj",
                "write_accepted_traj": "write_accepted_traj",
                "write_rejected_traj": "write_rejected_traj",
                "debug_traj_interval": "debug_traj_interval",
                "progress_log": "progress_log",
                "progress_stdout": "progress_stdout",
            },
            "calculator": {
                "calculator": "calculator",
                "model": "model",
                "model_file": "model_file",
                "device": "device",
                "use_kokkos": "use_kokkos",
            },
        }
        for section_name, key_map in section_maps.items():
            section = raw.get(section_name)
            if not isinstance(section, dict):
                continue
            for source_key, target_key in key_map.items():
                if source_key in section:
                    merged[target_key] = section[source_key]

        if isinstance(raw.get("moves"), dict):
            normalize_adsorbate_move_config(merged)

        backend_section = raw.get("backend")
        if isinstance(backend_section, dict):
            for source_key, target_key in section_maps["backend_config"].items():
                if source_key in backend_section:
                    merged[target_key] = backend_section[source_key]

    def enabled(self) -> bool:
        return bool(self.local_config.get("enabled", False))

    def run(self, candidate_manifest_path: str | Path) -> dict[str, str]:
        if not self.enabled():
            return {}

        candidate_manifest_path = Path(candidate_manifest_path)
        if not candidate_manifest_path.exists():
            return {}

        rows = self._read_csv(candidate_manifest_path)
        selected_states = self._selected_states()
        work_rows = [row for row in rows if self._row_selected(row, selected_states)]
        if not work_rows:
            return {}

        self._progress(
            "local-cmc start "
            f"manifest={candidate_manifest_path} work_blocks={len(work_rows)}"
        )
        calculator = (
            None
            if bool(self.local_config.get("pt_enabled", False))
            else self._build_calculator()
        )
        manifest_rows: list[dict[str, object]] = []
        updated_rows = [dict(row) for row in rows]
        rows_by_site_state = self._rows_by_site_state(updated_rows)
        sequential_rules = self._sequential_rules(selected_states)

        for manifest_row in self._local_work_order(
            updated_rows,
            selected_states,
            sequential_rules,
        ):
            if not self._row_selected(manifest_row, selected_states):
                continue
            self._refresh_sequential_target_candidates(
                manifest_row,
                rows_by_site_state,
                sequential_rules,
            )
            summary = self._run_state_block(manifest_row, calculator)
            if summary is None:
                continue
            manifest_rows.append(summary)
            manifest_row["n_candidates"] = summary.get("n_samples", 0)

        if manifest_rows:
            self._write_csv(candidate_manifest_path, updated_rows)
            local_manifest = output_path(self.config, "local_cmc_manifest_csv")
            self._write_csv(local_manifest, manifest_rows)
            self._progress(
                "local-cmc complete "
                f"manifest={local_manifest} blocks={len(manifest_rows)}"
            )
            return {"local_cmc_manifest_csv": str(local_manifest)}
        return {}

    @staticmethod
    def _rows_by_site_state(
        rows: Sequence[dict[str, object]],
    ) -> dict[tuple[str, str], dict[str, object]]:
        grouped: dict[tuple[str, str], dict[str, object]] = {}
        for row in rows:
            key = (str(row.get("site_id", "")), str(row.get("state_dir", "")))
            grouped[key] = row
        return grouped

    @staticmethod
    def _local_work_order(
        rows: Sequence[dict[str, object]],
        selected_states: set[str],
        sequential_rules: Sequence[dict[str, object]] = (),
    ) -> list[dict[str, object]]:
        priority: dict[str, int] = {}
        for rule in sequential_rules:
            source = str(rule.get("source_state", ""))
            target = str(rule.get("target_state", ""))
            if source and source not in priority:
                priority[source] = 20
            if source and target:
                priority[target] = max(
                    priority.get(target, 10),
                    priority.get(source, 20) + 10,
                )
        changed = True
        while changed:
            changed = False
            for rule in sequential_rules:
                source = str(rule.get("source_state", ""))
                target = str(rule.get("target_state", ""))
                if not source or not target:
                    continue
                source_priority = priority.get(source, 20)
                target_priority = priority.get(target, 10)
                if target_priority <= source_priority:
                    priority[target] = source_priority + 10
                    changed = True
        return sorted(
            (row for row in rows if str(row.get("state_dir", "")) in selected_states),
            key=lambda row: (
                str(row.get("site_id", "")),
                priority.get(str(row.get("state_dir", "")), 10),
            ),
        )

    def _refresh_sequential_target_candidates(
        self,
        manifest_row: dict[str, object],
        rows_by_site_state: dict[tuple[str, str], dict[str, object]],
        sequential_rules: Sequence[dict[str, object]],
    ) -> None:
        state_dir = str(manifest_row.get("state_dir", ""))
        rules = [
            rule
            for rule in sequential_rules
            if str(rule.get("target_state", "")) == state_dir
        ]
        if not rules:
            return

        site_id = str(manifest_row.get("site_id", ""))
        sequential_candidates: list[tuple[Atoms, dict[str, object]]] = []
        total_sources = 0
        for rule in rules:
            source_state = str(rule.get("source_state", ""))
            source_row = rows_by_site_state.get((site_id, source_state))
            if source_row is None:
                continue
            source_candidates = self._read_state_candidates(source_row)
            if not source_candidates:
                continue
            total_sources += len(source_candidates)
            sequential_candidates.extend(
                self._build_sequential_candidates(
                    source_candidates,
                    rule,
                    site_id=site_id,
                )
            )
        if not sequential_candidates:
            return

        candidates_traj = Path(str(manifest_row.get("candidates_traj", "")))
        candidates_csv = Path(str(manifest_row.get("candidates_csv", "")))
        self._write_candidates(sequential_candidates, candidates_traj, candidates_csv)
        manifest_row["n_candidates"] = len(sequential_candidates)
        manifest_row["_local_cmc_use_all_seeds"] = True

        signature = self._candidate_seed_signature(
            metadata for _, metadata in sequential_candidates
        )
        manifest_row["_local_cmc_seed_signature"] = signature
        local_dir = candidates_traj.parent / "local_cmc"
        if not self._local_seed_signature_matches(local_dir, signature):
            manifest_row["_local_cmc_ignore_existing"] = True
        self._progress(
            "local-cmc refreshed sequential seeds "
            f"site={site_id} target={state_dir} "
            f"source_seeds={total_sources} target_seeds={len(sequential_candidates)}"
        )

    def _sequential_rules(self, selected_states: set[str]) -> list[dict[str, object]]:
        if self.local_config.get("sequential_enabled") is False:
            return []
        raw_rules = self.local_config.get("sequential_rules")
        rules: list[dict[str, object]] = []
        if isinstance(raw_rules, dict):
            raw_rules = [raw_rules]
        if isinstance(raw_rules, (list, tuple)):
            for raw_rule in raw_rules:
                if not isinstance(raw_rule, dict):
                    continue
                rule = self._normalize_sequential_rule(raw_rule)
                if rule is not None:
                    rules.append(rule)

        if not rules and self._sequential_ooh_from_o_enabled():
            rules.append(
                {
                    "source_state": "02_O",
                    "target_state": "03_OOH",
                    "builder": "ooh_from_o",
                    "n_orientations": int(
                        self.local_config.get("sequential_ooh_orientations", 1)
                    ),
                }
            )

        return [
            rule
            for rule in rules
            if str(rule.get("source_state", "")) in selected_states
            and str(rule.get("target_state", "")) in selected_states
        ]

    @staticmethod
    def _normalize_sequential_rule(
        raw_rule: dict[str, object],
    ) -> dict[str, object] | None:
        source = raw_rule.get("source_state", raw_rule.get("source"))
        target = raw_rule.get("target_state", raw_rule.get("target"))
        builder = raw_rule.get("builder")
        if source in (None, "") or target in (None, "") or builder in (None, ""):
            return None
        rule = dict(raw_rule)
        rule["source_state"] = str(source)
        rule["target_state"] = str(target)
        rule["builder"] = str(builder)
        if "n_orientations" not in rule and "orientations" in rule:
            rule["n_orientations"] = rule["orientations"]
        return rule

    def _sequential_ooh_from_o_enabled(self) -> bool:
        if bool(self.local_config.get("sequential_ooh_from_o", False)):
            return True
        che_config = getattr(self.config, "che", {}) or {}
        if not isinstance(che_config, dict):
            return False
        route_mode = str(che_config.get("route_pairing_mode", "")).strip().lower()
        return route_mode in {"compatible", "linked", "parent_o"}

    def _read_state_candidates(
        self,
        manifest_row: dict[str, object],
    ) -> list[tuple[Atoms, dict[str, object]]]:
        traj = Path(str(manifest_row.get("candidates_traj", "")))
        csv_path = Path(str(manifest_row.get("candidates_csv", "")))
        if not traj.exists() or not csv_path.exists():
            return []
        atoms = self._read_atoms_list(traj)
        rows = self._read_csv(csv_path)
        count = min(len(atoms), len(rows))
        return [(atoms[index], dict(rows[index])) for index in range(count)]

    def _build_sequential_candidates(
        self,
        source_candidates: Sequence[tuple[Atoms, dict[str, object]]],
        rule: dict[str, object],
        *,
        site_id: str,
    ) -> list[tuple[Atoms, dict[str, object]]]:
        builder = str(rule.get("builder", "")).strip().lower()
        if builder in {"ooh_from_o", "o_to_ooh"}:
            generator = ReactionCandidateGenerator(self.config)
            orientations = rule.get("n_orientations", None)
            if orientations is not None:
                generator.candidate_config["ooh_orientations"] = max(
                    1,
                    int(orientations),
                )
            candidates = generator._generate_ooh_candidates(
                source_candidates,
                {"site_id": site_id},
            )
            return [
                (atoms, self._annotate_sequential_metadata(metadata, rule))
                for atoms, metadata in candidates
            ]
        raise ValueError(f"Unsupported local_cmc sequential builder: {builder!r}")

    @staticmethod
    def _annotate_sequential_metadata(
        metadata: dict[str, object],
        rule: dict[str, object],
    ) -> dict[str, object]:
        output = dict(metadata)
        parent_id = (
            output.get("parent_state_candidate_id")
            or output.get("parent_candidate_id")
            or output.get("parent_o_candidate_id")
            or ""
        )
        output.setdefault("parent_state_dir", str(rule.get("source_state", "")))
        output.setdefault("parent_state_candidate_id", parent_id)
        output.setdefault("parent_candidate_id", parent_id)
        output.setdefault("transition_builder", str(rule.get("builder", "")))
        return output

    @staticmethod
    def _candidate_seed_signature(rows: Iterable[dict[str, object]]) -> str:
        values: list[str] = []
        for row in rows:
            values.append(
                "|".join(
                    [
                        str(row.get("candidate_id", "")),
                        str(row.get("parent_o_candidate_id", "")),
                        str(row.get("parent_state_candidate_id", "")),
                        str(row.get("parent_candidate_id", "")),
                        str(row.get("parent_state_dir", "")),
                        str(row.get("transition_builder", "")),
                        str(row.get("orientation_index", "")),
                    ]
                )
            )
        return "\n".join(values)

    @staticmethod
    def _local_seed_signature_path(local_dir: Path) -> Path:
        return local_dir / "seed_signature.txt"

    def _local_seed_signature_matches(
        self,
        local_dir: Path,
        expected_signature: str,
    ) -> bool:
        if not expected_signature:
            return True
        path = self._local_seed_signature_path(local_dir)
        return path.exists() and path.read_text() == expected_signature

    def _write_local_seed_signature(
        self,
        local_dir: Path,
        signature: str,
    ) -> None:
        if not signature:
            return
        local_dir.mkdir(parents=True, exist_ok=True)
        self._local_seed_signature_path(local_dir).write_text(signature)

    def _run_state_block(
        self,
        manifest_row: dict[str, object],
        calculator,
    ) -> dict[str, object] | None:
        state_dir = str(manifest_row.get("state_dir", ""))
        state_path = Path(str(manifest_row.get("candidates_traj", ""))).parent
        candidates_traj = Path(str(manifest_row.get("candidates_traj", "")))
        candidates_csv = Path(str(manifest_row.get("candidates_csv", "")))
        if not candidates_traj.exists():
            self._progress(
                "local-cmc skip missing candidates "
                f"site={manifest_row.get('site_id', '')} state={state_dir}"
            )
            return None

        local_dir = state_path / "local_cmc"
        final_marker = local_dir / "done"
        ignore_existing = bool(manifest_row.get("_local_cmc_ignore_existing", False))
        expected_seed_signature = str(
            manifest_row.get("_local_cmc_seed_signature", "")
        ).strip()
        if (
            bool(self.local_config.get("skip_existing", False))
            and final_marker.exists()
            and not ignore_existing
            and self._local_seed_signature_matches(
                local_dir,
                expected_seed_signature,
            )
        ):
            seed_atoms = self._read_atoms_list(candidates_traj)
            seed_rows = self._read_csv(candidates_csv) if candidates_csv.exists() else []
            max_seeds = self.local_config.get("max_seed_candidates_per_state")
            if max_seeds is not None and not bool(
                manifest_row.get("_local_cmc_use_all_seeds", False)
            ):
                seed_atoms = seed_atoms[: int(max_seeds)]
                seed_rows = seed_rows[: int(max_seeds)]
            samples = self._reuse_existing_local_samples(
                seed_atoms,
                seed_rows,
                manifest_row,
                local_dir,
            )
            if samples:
                max_samples = self.local_config.get("max_output_candidates_per_state")
                if max_samples is not None:
                    samples = self._select_output_samples(samples, int(max_samples))
                if not bool(self.local_config.get("replace_candidates", True)):
                    original = [
                        (atoms.copy(), seed_rows[idx] if idx < len(seed_rows) else {})
                        for idx, atoms in enumerate(seed_atoms)
                    ]
                    samples = original + samples
                self._write_candidates(samples, candidates_traj, candidates_csv)
                self._progress(
                    "local-cmc reused existing trajectories "
                    f"site={manifest_row.get('site_id', '')} state={state_dir} "
                    f"samples={len(samples)}"
                )
            else:
                self._progress(
                    "local-cmc skip existing "
                    f"site={manifest_row.get('site_id', '')} state={state_dir}"
                )
            return {
                "site_id": manifest_row.get("site_id", ""),
                "state_dir": state_dir,
                "species": manifest_row.get("species", ""),
                "n_seeds": "existing",
                "n_samples": len(samples) if samples else manifest_row.get("n_candidates", ""),
                "local_cmc_dir": str(local_dir),
                "candidates_traj": str(candidates_traj),
                "candidates_csv": str(candidates_csv),
            }

        local_dir.mkdir(parents=True, exist_ok=True)
        seed_atoms = self._read_atoms_list(candidates_traj)
        seed_rows = self._read_csv(candidates_csv) if candidates_csv.exists() else []
        max_seeds = self.local_config.get("max_seed_candidates_per_state")
        if max_seeds is not None and not bool(
            manifest_row.get("_local_cmc_use_all_seeds", False)
        ):
            seed_atoms = seed_atoms[: int(max_seeds)]
            seed_rows = seed_rows[: int(max_seeds)]

        samples: list[tuple[Atoms, dict[str, object]]] = []
        self._progress(
            "local-cmc block start "
            f"site={manifest_row.get('site_id', '')} state={state_dir} "
            f"seeds={len(seed_atoms)} radius={self.local_config.get('radius_A')}"
        )
        for seed_index, atoms in enumerate(seed_atoms):
            seed_row = seed_rows[seed_index] if seed_index < len(seed_rows) else {}
            samples.extend(
                self._run_seed(
                    atoms,
                    seed_row,
                    manifest_row,
                    local_dir,
                    seed_index,
                    calculator,
                )
            )

        max_samples = self.local_config.get("max_output_candidates_per_state")
        if max_samples is not None:
            samples = self._select_output_samples(samples, int(max_samples))

        if not bool(self.local_config.get("replace_candidates", True)):
            original = [
                (atoms.copy(), seed_rows[idx] if idx < len(seed_rows) else {})
                for idx, atoms in enumerate(seed_atoms)
            ]
            samples = original + samples

        self._write_candidates(samples, candidates_traj, candidates_csv)
        final_marker.write_text(f"{datetime.now().isoformat()} samples={len(samples)}\n")
        self._write_local_seed_signature(local_dir, expected_seed_signature)
        self._progress(
            "local-cmc block done "
            f"site={manifest_row.get('site_id', '')} state={state_dir} "
            f"samples={len(samples)}"
        )
        return {
            "site_id": manifest_row.get("site_id", ""),
            "state_dir": state_dir,
            "species": manifest_row.get("species", ""),
            "n_seeds": len(seed_atoms),
            "n_samples": len(samples),
            "local_cmc_dir": str(local_dir),
            "candidates_traj": str(candidates_traj),
            "candidates_csv": str(candidates_csv),
        }

    def _reuse_existing_local_samples(
        self,
        seed_atoms: Sequence[Atoms],
        seed_rows: Sequence[dict[str, object]],
        manifest_row: dict[str, object],
        local_dir: Path,
    ) -> list[tuple[Atoms, dict[str, object]]]:
        samples: list[tuple[Atoms, dict[str, object]]] = []
        for seed_index, atoms in enumerate(seed_atoms):
            seed_row = seed_rows[seed_index] if seed_index < len(seed_rows) else {}
            center = self._local_center_for_seed(
                manifest_row,
                seed_row,
                fallback=atoms.positions[self._anchor_index(atoms)].copy(),
            )
            prefix = f"seed{seed_index:03d}"
            if bool(self.local_config.get("pt_enabled", False)):
                pt_dir = local_dir / f"{prefix}_pt"
                temperatures = self._pt_temperatures()
                target_temperature = self._pt_target_temperature(temperatures)
                target_path = (
                    pt_dir / f"replica_{self._temperature_label(target_temperature)}.traj"
                )
                frames = self._read_atoms_list(target_path) if target_path.exists() else []
                samples.extend(
                    self._annotate_local_frames(
                        frames,
                        seed_row,
                        seed_index,
                        center,
                        prefix,
                        candidate_kind="local_cmc_pt_sample",
                        suffix="localpt",
                        extra_metadata={
                            "local_cmc_pt_dir": str(pt_dir),
                            "local_cmc_pt_temperature_K": float(target_temperature),
                        },
                    )
                )
                continue

            traj_path = local_dir / f"{prefix}_samples.traj"
            frames = self._read_atoms_list(traj_path) if traj_path.exists() else []
            skip = int(self.local_config.get("equilibration_cycles", 100)) // max(
                1,
                int(self.local_config.get("sample_interval", 25)),
            )
            frames = frames[skip:]
            samples.extend(
                self._annotate_local_frames(
                    frames,
                    seed_row,
                    seed_index,
                    center,
                    prefix,
                    candidate_kind="local_cmc_sample",
                    suffix="localcmc",
                    extra_metadata={},
                )
            )
        return samples

    def _run_seed(
        self,
        atoms: Atoms,
        seed_row: dict[str, object],
        manifest_row: dict[str, object],
        local_dir: Path,
        seed_index: int,
        calculator,
    ) -> list[tuple[Atoms, dict[str, object]]]:
        center = self._local_center_for_seed(
            manifest_row,
            seed_row,
            fallback=atoms.positions[self._anchor_index(atoms)].copy(),
        )
        template, anchor_local_index = self._adsorbate_template(atoms)
        prefix = f"seed{seed_index:03d}"
        if bool(self.local_config.get("pt_enabled", False)):
            return self._run_seed_pt(
                atoms,
                seed_row,
                manifest_row,
                local_dir,
                seed_index,
                center,
                template,
                anchor_local_index,
                prefix,
            )

        traj_path = local_dir / f"{prefix}_samples.traj"
        thermo_path = local_dir / f"{prefix}.dat"
        checkpoint_path = local_dir / f"{prefix}.pkl"
        attempted_path = self._debug_traj_path(local_dir, prefix, "attempted")
        accepted_path = self._debug_traj_path(local_dir, prefix, "accepted")
        rejected_path = self._debug_traj_path(local_dir, prefix, "rejected")
        for path in (
            traj_path,
            thermo_path,
            checkpoint_path,
            attempted_path,
            accepted_path,
            rejected_path,
        ):
            if path is not None and path.exists():
                path.unlink()

        driver = AdsorbateCMC(
            atoms=atoms,
            calculator=calculator,
            T=float(self.local_config.get("temperature_K", 303.0)),
            adsorbate=template,
            adsorbate_anchor_index=anchor_local_index,
            substrate_elements=tuple(self.config.substrate_elements),
            functional_elements=tuple(self.config.functional_elements),
            site_elements=tuple(self.config.site_elements),
            surface_side=str(self.config.surface_side),
            site_type=self.local_config.get("site_types", None) or self.config.site_types,
            move_mode=str(self.local_config.get("move_mode", "hybrid")),
            site_hop_prob=float(self.local_config.get("site_hop_prob", 0.6)),
            reorientation_prob=float(self.local_config.get("reorientation_prob", 0.3)),
            hop_reorientation_prob=float(
                self.local_config.get("hop_reorientation_prob", 0.0)
            ),
            hop_puckering_prob=float(
                self.local_config.get("hop_puckering_prob", 0.0)
            ),
            hop_puckering_reorientation_prob=float(
                self.local_config.get("hop_puckering_reorientation_prob", 0.0)
            ),
            puckering_prob=float(self.local_config.get("puckering_prob", 0.0)),
            puckering_hop_prob=float(
                self.local_config.get("puckering_hop_prob", 0.0)
            ),
            puckering_elements=(
                self._parse_element_list(self.local_config.get("puckering_elements"))
                or tuple(self.config.site_elements)
            ),
            puckering_height_A=float(
                self.local_config.get("puckering_height_A", 0.15)
            ),
            puckering_height_jitter_A=(
                None
                if self.local_config.get("puckering_height_jitter_A") is None
                else float(self.local_config.get("puckering_height_jitter_A", 0.0))
            ),
            displacement_sigma=float(self.local_config.get("displacement_sigma", 0.2)),
            max_displacement_trials=int(
                self.local_config.get("max_displacement_trials", 20)
            ),
            max_hop_reorientation_trials=(
                None
                if self.local_config.get("max_hop_reorientation_trials") is None
                else int(self.local_config.get("max_hop_reorientation_trials", 20))
            ),
            max_puckering_trials=(
                None
                if self.local_config.get("max_puckering_trials") is None
                else int(self.local_config.get("max_puckering_trials", 20))
            ),
            rotation_max_angle_deg=float(
                self.local_config.get("rotation_max_angle_deg", 35.0)
            ),
            hop_reorientation_angle_deg=float(
                self.local_config.get("hop_reorientation_angle_deg", 180.0)
            ),
            site_match_tol=float(self.config.site_match_tol),
            support_xy_tol=float(self.config.support_xy_tol),
            adsorbate_surface_clearance_A=float(
                self.local_config.get("adsorbate_surface_clearance_A", 0.0)
            ),
            adsorbate_surface_xy_tol_A=(
                None
                if self.local_config.get("adsorbate_surface_xy_tol_A") is None
                else float(self.local_config.get("adsorbate_surface_xy_tol_A", 0.0))
            ),
            termination_site_xy_tol=self.config.termination_site_xy_tol,
            surface_layer_tol=float(self.config.surface_layer_tol),
            termination_clearance=float(self.config.termination_clearance),
            vertical_offset=float(self.config.vertical_offset),
            relax=bool(self.local_config.get("relax", False)),
            relax_steps=int(self.local_config.get("relax_steps", 20)),
            fmax=float(self.local_config.get("fmax", 0.05)),
            traj_file=str(traj_path),
            attempted_traj_file=str(attempted_path) if attempted_path else None,
            accepted_traj_file=str(accepted_path) if accepted_path else None,
            rejected_traj_file=str(rejected_path) if rejected_path else None,
            debug_traj_interval=int(self.local_config.get("debug_traj_interval", 1)),
            thermo_file=str(thermo_path),
            checkpoint_file=str(checkpoint_path),
            checkpoint_interval=int(self.local_config.get("checkpoint_interval", 0)),
            seed=int(self.local_config.get("seed", 81)) + seed_index,
            enable_hybrid_md=bool(self.local_config.get("enable_hybrid_md", False)),
            md_move_prob=float(self.local_config.get("md_move_prob", 0.05)),
            md_steps=int(self.local_config.get("md_steps", 20)),
            md_timestep_fs=float(self.local_config.get("md_timestep_fs", 0.5)),
            md_ensemble=str(self.local_config.get("md_ensemble", "nve")),
            md_accept_mode=str(self.local_config.get("md_accept_mode", "potential")),
            enforce_molecular_integrity=True,
            site_region_center_A=center,
            site_region_radius_A=float(self.local_config.get("radius_A", 3.0)),
            site_region_distance_metric=str(
                self.local_config.get("distance_metric", "xy")
            ),
        )
        driver.run(
            nsweeps=int(self.local_config.get("n_cycles", 500)),
            traj_file=str(traj_path),
            interval=max(1, int(self.local_config.get("sample_interval", 25))),
            sample_interval=max(1, int(self.local_config.get("sample_interval", 25))),
            equilibration=int(self.local_config.get("equilibration_cycles", 100)),
            sweeps_are_total=True,
        )
        frames = self._read_atoms_list(traj_path) if traj_path.exists() else []
        skip = int(self.local_config.get("equilibration_cycles", 100)) // max(
            1,
            int(self.local_config.get("sample_interval", 25)),
        )
        frames = frames[skip:]
        if not frames:
            frames = [driver.atoms.copy()]

        return self._annotate_local_frames(
            frames,
            seed_row,
            seed_index,
            center,
            prefix,
            candidate_kind="local_cmc_sample",
            suffix="localcmc",
            extra_metadata={},
        )

    def _run_seed_pt(
        self,
        atoms: Atoms,
        seed_row: dict[str, object],
        manifest_row: dict[str, object],
        local_dir: Path,
        seed_index: int,
        center: np.ndarray,
        template: Atoms,
        anchor_local_index: int,
        prefix: str,
    ) -> list[tuple[Atoms, dict[str, object]]]:
        pt_dir = local_dir / f"{prefix}_pt"
        pt_dir.mkdir(parents=True, exist_ok=True)
        if not bool(self.local_config.get("pt_resume", False)):
            for path in pt_dir.iterdir():
                if path.is_file():
                    path.unlink()

        temperatures = self._pt_temperatures()
        target_temperature = self._pt_target_temperature(temperatures)
        calc_class, calc_kwargs = self._replica_calculator_spec()
        mc_kwargs = self._pt_mc_kwargs(
            template=template,
            anchor_local_index=anchor_local_index,
            center=center,
        )
        replica_states = self._pt_replica_states(atoms, temperatures, pt_dir, mc_kwargs)
        worker_init_info = {
            "calculator_module": calc_class.__module__,
            "calculator_class_name": calc_class.__name__,
            "mc_module": AdsorbateCMC.__module__,
            "mc_class": AdsorbateCMC.__name__,
            "calc_kwargs": calc_kwargs,
            "mc_kwargs": mc_kwargs,
            "atoms_template": atoms.copy(),
        }
        n_gpus, workers_per_gpu = self._pt_pool_config()
        swap_interval = int(self.local_config.get("swap_interval", 10))
        total_sweeps = int(self.local_config.get("n_cycles", 500))
        pt_cycles = self.local_config.get("pt_n_cycles")
        if pt_cycles is None:
            pt_cycles = max(1, int(np.ceil(total_sweeps / max(1, swap_interval))))
        pt_equilibration = self.local_config.get("pt_equilibration_cycles")
        if pt_equilibration is None:
            eq_sweeps = int(self.local_config.get("equilibration_cycles", 100))
            pt_equilibration = int(np.ceil(eq_sweeps / max(1, swap_interval)))

        pt = ReplicaExchange(
            n_gpus=n_gpus,
            workers_per_gpu=workers_per_gpu,
            replica_states=replica_states,
            swap_interval=swap_interval,
            report_interval=int(self.local_config.get("report_interval", 1)),
            sampling_interval=max(1, int(self.local_config.get("sample_interval", 25))),
            checkpoint_interval=int(self.local_config.get("checkpoint_interval", 0)),
            swap_stride=int(self.local_config.get("swap_stride", 1)),
            local_eq_fraction=float(self.local_config.get("pt_local_eq_fraction", 0.2)),
            stats_file=str(pt_dir / "replica_stats.csv"),
            results_file=str(pt_dir / "results.csv"),
            checkpoint_file=str(pt_dir / "pt_state.pkl"),
            resume=bool(self.local_config.get("pt_resume", False)),
            worker_init_info=worker_init_info,
            seed=int(self.local_config.get("seed", 81)),
            seed_nonce=seed_index * 1000,
            execution_backend=str(self.local_config.get("backend", "multiprocessing")),
            backend_kwargs=self._pt_backend_kwargs(),
        )
        pt.run(n_cycles=int(pt_cycles), equilibration_cycles=int(pt_equilibration))

        target_state = min(
            pt.replica_states,
            key=lambda state: abs(float(state["T"]) - target_temperature),
        )
        frames = (
            self._read_atoms_list(target_state["traj_file"])
            if Path(str(target_state["traj_file"])).exists()
            else []
        )
        if not frames:
            frames = [target_state["atoms"].copy()]

        return self._annotate_local_frames(
            frames,
            seed_row,
            seed_index,
            center,
            prefix,
            candidate_kind="local_cmc_pt_sample",
            suffix="localpt",
            extra_metadata={
                "local_cmc_pt_dir": str(pt_dir),
                "local_cmc_pt_temperature_K": float(target_state["T"]),
            },
        )

    def _annotate_local_frames(
        self,
        frames: Sequence[Atoms],
        seed_row: dict[str, object],
        seed_index: int,
        center: np.ndarray,
        prefix: str,
        *,
        candidate_kind: str,
        suffix: str,
        extra_metadata: dict[str, object],
    ) -> list[tuple[Atoms, dict[str, object]]]:
        output: list[tuple[Atoms, dict[str, object]]] = []
        for sample_index, frame in enumerate(frames):
            metadata = dict(seed_row)
            metadata.update(
                {
                    "candidate_id": (
                        f"{seed_row.get('candidate_id', prefix)}"
                        f"_{suffix}{sample_index:03d}"
                    ),
                    "candidate_kind": candidate_kind,
                    "local_cmc_seed_index": seed_index,
                    "local_cmc_sample_index": sample_index,
                    "local_cmc_center_A": " ".join(f"{value:.8f}" for value in center),
                    "local_cmc_radius_A": float(self.local_config.get("radius_A", 3.0)),
                    "local_cmc_distance_metric": self.local_config.get(
                        "distance_metric",
                        "xy",
                    ),
                    "anchor_index": self._anchor_index(frame),
                }
            )
            metadata.update(extra_metadata)
            ReactionCandidateGenerator._annotate_candidate(frame, metadata)
            output.append((frame, metadata))
        return output

    def _pt_mc_kwargs(
        self,
        *,
        template: Atoms,
        anchor_local_index: int,
        center: np.ndarray,
    ) -> dict[str, object]:
        return {
            "adsorbate": template,
            "adsorbate_anchor_index": anchor_local_index,
            "substrate_elements": tuple(self.config.substrate_elements),
            "functional_elements": tuple(self.config.functional_elements),
            "site_elements": tuple(self.config.site_elements),
            "surface_side": str(self.config.surface_side),
            "site_type": self.local_config.get("site_types", None)
            or self.config.site_types,
            "move_mode": str(self.local_config.get("move_mode", "hybrid")),
            "site_hop_prob": float(self.local_config.get("site_hop_prob", 0.6)),
            "reorientation_prob": float(self.local_config.get("reorientation_prob", 0.3)),
            "hop_reorientation_prob": float(
                self.local_config.get("hop_reorientation_prob", 0.0)
            ),
            "hop_puckering_prob": float(
                self.local_config.get("hop_puckering_prob", 0.0)
            ),
            "hop_puckering_reorientation_prob": float(
                self.local_config.get("hop_puckering_reorientation_prob", 0.0)
            ),
            "puckering_prob": float(self.local_config.get("puckering_prob", 0.0)),
            "puckering_hop_prob": float(
                self.local_config.get("puckering_hop_prob", 0.0)
            ),
            "puckering_elements": (
                self._parse_element_list(self.local_config.get("puckering_elements"))
                or tuple(self.config.site_elements)
            ),
            "puckering_height_A": float(
                self.local_config.get("puckering_height_A", 0.15)
            ),
            "puckering_height_jitter_A": (
                None
                if self.local_config.get("puckering_height_jitter_A") is None
                else float(self.local_config.get("puckering_height_jitter_A", 0.0))
            ),
            "displacement_sigma": float(self.local_config.get("displacement_sigma", 0.2)),
            "max_displacement_trials": int(
                self.local_config.get("max_displacement_trials", 20)
            ),
            "max_hop_reorientation_trials": (
                None
                if self.local_config.get("max_hop_reorientation_trials") is None
                else int(self.local_config.get("max_hop_reorientation_trials", 20))
            ),
            "max_puckering_trials": (
                None
                if self.local_config.get("max_puckering_trials") is None
                else int(self.local_config.get("max_puckering_trials", 20))
            ),
            "rotation_max_angle_deg": float(
                self.local_config.get("rotation_max_angle_deg", 35.0)
            ),
            "hop_reorientation_angle_deg": float(
                self.local_config.get("hop_reorientation_angle_deg", 180.0)
            ),
            "site_match_tol": float(self.config.site_match_tol),
            "support_xy_tol": float(self.config.support_xy_tol),
            "adsorbate_surface_clearance_A": float(
                self.local_config.get("adsorbate_surface_clearance_A", 0.0)
            ),
            "adsorbate_surface_xy_tol_A": (
                None
                if self.local_config.get("adsorbate_surface_xy_tol_A") is None
                else float(self.local_config.get("adsorbate_surface_xy_tol_A", 0.0))
            ),
            "termination_site_xy_tol": self.config.termination_site_xy_tol,
            "surface_layer_tol": float(self.config.surface_layer_tol),
            "termination_clearance": float(self.config.termination_clearance),
            "vertical_offset": float(self.config.vertical_offset),
            "relax": bool(self.local_config.get("relax", False)),
            "relax_steps": int(self.local_config.get("relax_steps", 20)),
            "fmax": float(self.local_config.get("fmax", 0.05)),
            "debug_traj_interval": int(self.local_config.get("debug_traj_interval", 1)),
            "checkpoint_interval": 0,
            "enable_hybrid_md": bool(self.local_config.get("enable_hybrid_md", False)),
            "md_move_prob": float(self.local_config.get("md_move_prob", 0.05)),
            "md_steps": int(self.local_config.get("md_steps", 20)),
            "md_timestep_fs": float(self.local_config.get("md_timestep_fs", 0.5)),
            "md_ensemble": str(self.local_config.get("md_ensemble", "nve")),
            "md_accept_mode": str(self.local_config.get("md_accept_mode", "potential")),
            "enforce_molecular_integrity": True,
            "site_region_center_A": center,
            "site_region_radius_A": float(self.local_config.get("radius_A", 3.0)),
            "site_region_distance_metric": str(
                self.local_config.get("distance_metric", "xy")
            ),
        }

    def _pt_replica_states(
        self,
        atoms: Atoms,
        temperatures: Sequence[float],
        pt_dir: Path,
        mc_kwargs: dict[str, object],
    ) -> list[dict[str, object]]:
        states = []
        for replica_id, temperature in enumerate(temperatures):
            label = self._temperature_label(float(temperature))
            replica_atoms = atoms.copy()
            replica_atoms.calc = None
            replica_mc_kwargs = dict(mc_kwargs)
            attempted_path = self._debug_traj_path(pt_dir, f"replica_{label}", "attempted")
            accepted_path = self._debug_traj_path(pt_dir, f"replica_{label}", "accepted")
            rejected_path = self._debug_traj_path(pt_dir, f"replica_{label}", "rejected")
            replica_mc_kwargs.update(
                {
                    "attempted_traj_file": str(attempted_path) if attempted_path else None,
                    "accepted_traj_file": str(accepted_path) if accepted_path else None,
                    "rejected_traj_file": str(rejected_path) if rejected_path else None,
                }
            )
            states.append(
                {
                    "id": replica_id,
                    "T": float(temperature),
                    "atoms": replica_atoms,
                    "e_old": 0.0,
                    "sweep": 0,
                    "cum_sum_E": 0.0,
                    "cum_sum_E_sq": 0.0,
                    "cum_n_samples": 0,
                    "traj_file": str(pt_dir / f"replica_{label}.traj"),
                    "thermo_file": str(pt_dir / f"replica_{label}.dat"),
                    "checkpoint_file": str(pt_dir / f"checkpoint_{label}.pkl"),
                    "mc_kwargs": replica_mc_kwargs,
                }
            )
        return states

    def _replica_calculator_spec(self) -> tuple[type, dict]:
        values = vars(self.config).copy()
        for key in ("calculator", "model", "model_file", "device", "use_kokkos"):
            value = self.local_config.get(key)
            if value not in (None, ""):
                values[key] = value
        return build_replica_calculator_spec(SimpleNamespace(**values))

    def _pt_temperatures(self) -> list[float]:
        values = self.local_config.get("temperatures_K")
        if values in (None, ""):
            return [float(self.local_config.get("temperature_K", 303.0))]
        if isinstance(values, str):
            tokens = [token for token in values.replace(",", " ").split() if token]
            temperatures = [float(token) for token in tokens]
        else:
            temperatures = [float(value) for value in values]
        if not temperatures:
            raise ValueError("local_cmc.temperatures_K must not be empty when provided.")
        return temperatures

    def _pt_target_temperature(self, temperatures: Sequence[float]) -> float:
        value = self.local_config.get("target_temperature_K")
        target = (
            float(value)
            if value not in (None, "")
            else float(self.local_config.get("temperature_K", temperatures[0]))
        )
        return float(min(temperatures, key=lambda temp: abs(float(temp) - target)))

    def _pt_pool_config(self) -> tuple[int, int]:
        n_gpus = self.local_config.get("n_gpus")
        if n_gpus is None:
            n_gpus = self._infer_total_gpus_from_environment() or 1
        workers_per_gpu = self.local_config.get("workers_per_gpu")
        if workers_per_gpu is None:
            workers_per_gpu = 1
        return int(n_gpus), int(workers_per_gpu)

    def _pt_backend_kwargs(self) -> dict | None:
        if str(self.local_config.get("backend", "multiprocessing")).lower() != "ray":
            return None
        actor_options = {
            "num_cpus": float(self.local_config.get("ray_num_cpus_per_task", 1)),
        }
        num_gpus = self.local_config.get("ray_num_gpus_per_task")
        if num_gpus is not None:
            actor_options["num_gpus"] = float(num_gpus)
        return {
            "init_kwargs": {
                "address": self.local_config.get("ray_address") or "auto",
                "log_to_driver": bool(
                    self.local_config.get("ray_log_to_driver", False)
                ),
            },
            "actor_options": actor_options,
            "max_restarts": int(self.local_config.get("ray_actor_max_restarts", 0)),
            "max_task_retries": int(
                self.local_config.get("ray_actor_max_task_retries", 0)
            ),
            "get_result_timeout_s": self.local_config.get("ray_get_timeout_s"),
            "use_placement_group": bool(
                self.local_config.get("use_placement_group", False)
            ),
            "placement_group_strategy": str(
                self.local_config.get("placement_group_strategy", "SPREAD")
            ),
            "remove_placement_group_on_stop": bool(
                self.local_config.get("remove_placement_group_on_stop", True)
            ),
            "shutdown_on_stop": bool(self.local_config.get("shutdown_on_stop", False)),
        }

    @staticmethod
    def _temperature_label(temperature: float) -> str:
        text = f"{float(temperature):.6g}".replace(".", "p")
        return f"{text}K"

    @staticmethod
    def _infer_total_gpus_from_environment() -> int | None:
        for name in ("SLURM_GPUS", "SLURM_GPUS_ON_NODE"):
            value = os.getenv(name)
            if value:
                try:
                    parsed = int(str(value).strip())
                except ValueError:
                    continue
                if parsed > 0:
                    return parsed
        visible = os.getenv("CUDA_VISIBLE_DEVICES")
        if visible:
            tokens = [token.strip() for token in visible.split(",") if token.strip()]
            if tokens and tokens != ["NoDevFiles"]:
                return len(tokens)
        return None

    def _debug_traj_path(self, local_dir: Path, prefix: str, kind: str) -> Path | None:
        if bool(self.local_config.get("write_debug_trajs", False)) or bool(
            self.local_config.get(f"write_{kind}_traj", False)
        ):
            return local_dir / f"{prefix}_{kind}.traj"
        return None

    def _build_calculator(self):
        values = vars(self.config).copy()
        for key in ("calculator", "model", "model_file", "device", "use_kokkos"):
            value = self.local_config.get(key)
            if value not in (None, ""):
                values[key] = value
        cfg = SimpleNamespace(**values)
        return build_adsorbate_gcmc_calculator(
            cfg,
            {"device": values.get("device") or getattr(self.config, "device", None)},
        )

    def _local_center_for_seed(
        self,
        manifest_row: dict[str, object],
        seed_row: dict[str, object],
        *,
        fallback: np.ndarray,
    ) -> np.ndarray:
        center_state = str(self.local_config.get("center_state", "01_OH"))
        state_path = Path(str(manifest_row.get("candidates_traj", ""))).parent
        center_path = state_path.parent / center_state
        center_traj = center_path / "candidates.traj"
        center_csv = center_path / "candidates.csv"
        if not center_traj.exists() or not center_csv.exists():
            return np.asarray(fallback, dtype=float)

        center_rows = self._read_csv(center_csv)
        center_atoms = self._read_atoms_list(center_traj)
        if not center_rows or not center_atoms:
            return np.asarray(fallback, dtype=float)
        seed_rank = str(seed_row.get("source_representative_rank", ""))
        selected_index = 0
        for idx, row in enumerate(center_rows):
            if seed_rank and str(row.get("source_representative_rank", "")) == seed_rank:
                selected_index = idx
                break
        selected_index = min(selected_index, len(center_atoms) - 1)
        atoms = center_atoms[selected_index]
        anchor_idx = int(float(center_rows[selected_index].get("anchor_index", -1)))
        if not (0 <= anchor_idx < len(atoms)):
            anchor_idx = self._anchor_index(atoms)
        return np.asarray(atoms.positions[anchor_idx], dtype=float)

    @staticmethod
    def _adsorbate_template(atoms: Atoms) -> tuple[Atoms, int]:
        tags = np.asarray(atoms.get_tags(), dtype=int)
        groups = [
            np.where(tags == tag)[0]
            for tag in sorted(np.unique(tags))
            if tag >= ADSORBATE_TAG_OFFSET
        ]
        if not groups:
            raise ValueError("Local CMC candidates must contain a tagged adsorbate.")
        group = np.asarray(groups[0], dtype=int)
        template = atoms[group].copy()
        template.positions -= template.positions[0].copy()
        return template, 0

    @staticmethod
    def _anchor_index(atoms: Atoms) -> int:
        tags = np.asarray(atoms.get_tags(), dtype=int)
        for tag in sorted(np.unique(tags)):
            if tag >= ADSORBATE_TAG_OFFSET:
                return int(np.where(tags == tag)[0][0])
        oxygen = [idx for idx, atom in enumerate(atoms) if atom.symbol == "O"]
        if oxygen:
            return int(oxygen[-1])
        return len(atoms) - 1

    def _selected_states(self) -> set[str]:
        states = self.local_config.get("states") or []
        if isinstance(states, str):
            states = [states]
        return {str(state) for state in states}

    def _select_output_samples(
        self,
        samples: Sequence[tuple[Atoms, dict[str, object]]],
        max_samples: int,
    ) -> list[tuple[Atoms, dict[str, object]]]:
        if max_samples < 0:
            raise ValueError("local_cmc.max_output_candidates_per_state must be >= 0.")
        if max_samples == 0:
            return []
        if len(samples) <= max_samples:
            return list(samples)

        mode = self._output_selection_mode()
        if mode == "first":
            indices = list(range(max_samples))
        elif mode == "last":
            indices = list(range(len(samples) - max_samples, len(samples)))
        elif mode == "stride":
            indices = self._evenly_spaced_indices(len(samples), max_samples)
        elif mode == "random":
            seed = int(self.local_config.get("seed", 81))
            rng = np.random.default_rng(seed)
            indices = sorted(int(i) for i in rng.choice(len(samples), max_samples, replace=False))
        elif mode == "diverse":
            indices = self._diverse_sample_indices(samples, max_samples)
        elif mode == "motif_diverse":
            indices = self._motif_diverse_sample_indices(samples, max_samples)
        else:
            raise ValueError(
                "local_cmc.output_selection must be one of "
                "'first', 'last', 'stride', 'random', 'diverse', "
                "or 'motif_diverse'."
            )

        return [samples[index] for index in indices]

    def _output_selection_mode(self) -> str:
        raw = str(self.local_config.get("output_selection", "diverse")).lower()
        aliases = {
            "first": "first",
            "head": "first",
            "initial": "first",
            "last": "last",
            "tail": "last",
            "final": "last",
            "stride": "stride",
            "strided": "stride",
            "even": "stride",
            "evenly_spaced": "stride",
            "uniform": "stride",
            "random": "random",
            "shuffle": "random",
            "diverse": "diverse",
            "diversity": "diverse",
            "farthest": "diverse",
            "fps": "diverse",
            "motif_diverse": "motif_diverse",
            "motif-diverse": "motif_diverse",
            "motif": "motif_diverse",
            "motifs": "motif_diverse",
            "site_motif": "motif_diverse",
            "site-motif": "motif_diverse",
            "motif_balanced": "motif_diverse",
            "motif-balanced": "motif_diverse",
        }
        mode = aliases.get(raw)
        if mode is None:
            raise ValueError(
                "local_cmc.output_selection must be one of "
                "'first', 'last', 'stride', 'random', 'diverse', "
                "or 'motif_diverse'."
            )
        return mode

    @staticmethod
    def _evenly_spaced_indices(n_items: int, n_select: int) -> list[int]:
        if n_select >= n_items:
            return list(range(n_items))
        raw = np.linspace(0, n_items - 1, n_select)
        indices: list[int] = []
        seen: set[int] = set()
        for value in raw:
            index = int(round(float(value)))
            index = min(max(index, 0), n_items - 1)
            if index not in seen:
                indices.append(index)
                seen.add(index)
        for index in range(n_items):
            if len(indices) >= n_select:
                break
            if index not in seen:
                indices.append(index)
                seen.add(index)
        return sorted(indices)

    def _diverse_sample_indices(
        self,
        samples: Sequence[tuple[Atoms, dict[str, object]]],
        n_select: int,
        initial_indices: Sequence[int] = (),
    ) -> list[int]:
        features = [self._sample_feature_vector(atoms) for atoms, _ in samples]
        if any(feature is None for feature in features):
            return self._evenly_spaced_indices(len(samples), n_select)

        selected: list[int] = []
        seen: set[int] = set()
        for index in initial_indices:
            index = int(index)
            if 0 <= index < len(samples) and index not in seen:
                selected.append(index)
                seen.add(index)
                if len(selected) >= n_select:
                    return sorted(selected)
        if not selected:
            selected = [0]
            seen.add(0)
        remaining = set(range(len(samples))) - seen
        min_dist = np.full(len(samples), np.inf, dtype=float)
        while remaining and len(selected) < n_select:
            latest = selected[-1]
            latest_feature = features[latest]
            for index in remaining:
                dist = self._feature_distance(features[index], latest_feature)
                if dist < min_dist[index]:
                    min_dist[index] = dist
            next_index = max(remaining, key=lambda index: (min_dist[index], -index))
            selected.append(int(next_index))
            remaining.remove(next_index)

        return sorted(selected)

    def _motif_diverse_sample_indices(
        self,
        samples: Sequence[tuple[Atoms, dict[str, object]]],
        n_select: int,
    ) -> list[int]:
        groups: dict[str, list[int]] = {}
        for index, (atoms, _) in enumerate(samples):
            key = self._sample_motif_key(atoms)
            groups.setdefault(key, []).append(index)
        if len(groups) <= 1:
            return self._diverse_sample_indices(samples, n_select)

        reserve: list[int] = []
        for _, group in sorted(
            groups.items(),
            key=lambda item: (len(item[1]), item[1][0], item[0]),
        ):
            reserve.append(self._motif_group_representative_index(samples, group))
            if len(reserve) >= n_select:
                return sorted(reserve)
        return self._diverse_sample_indices(samples, n_select, initial_indices=reserve)

    def _sample_motif_key(self, atoms: Atoms) -> str:
        ads_indices = set(self._adsorbate_indices(atoms))
        anchor_idx = self._anchor_index(atoms)
        if not ads_indices or not (0 <= anchor_idx < len(atoms)):
            return "unknown"

        support_indices = self._motif_support_indices(atoms, ads_indices)
        if len(support_indices) == 0:
            return "unknown"

        vectors = np.asarray(
            atoms.get_distances(anchor_idx, support_indices, mic=True, vector=True),
            dtype=float,
        )
        lateral = np.linalg.norm(vectors[:, :2], axis=1)
        order = np.argsort(lateral)
        sorted_distances = [float(lateral[int(idx)]) for idx in order]
        sorted_indices = [int(support_indices[int(idx)]) for idx in order]
        sorted_symbols = [atoms[index].symbol for index in sorted_indices]

        d1 = sorted_distances[0]
        d2 = sorted_distances[1] if len(sorted_distances) > 1 else float("inf")
        d3 = sorted_distances[2] if len(sorted_distances) > 2 else float("inf")
        atop_cutoff = float(self.local_config.get("motif_atop_distance_A", 1.25))
        atop_ratio = float(self.local_config.get("motif_atop_ratio", 0.65))
        bridge_ratio = float(self.local_config.get("motif_bridge_ratio", 0.75))

        if d1 <= atop_cutoff or (np.isfinite(d2) and d1 <= atop_ratio * d2):
            return f"atop:{sorted_symbols[0]}"
        if np.isfinite(d2) and (not np.isfinite(d3) or d2 <= bridge_ratio * d3):
            symbols = "-".join(sorted(sorted_symbols[:2]))
            return f"bridge:{symbols}"
        symbols = "-".join(sorted(sorted_symbols[: min(3, len(sorted_symbols))]))
        return f"hollow:{symbols}"

    def _motif_support_indices(
        self,
        atoms: Atoms,
        ads_indices: set[int],
    ) -> np.ndarray:
        elements: set[str] = set()
        elements.update(str(value) for value in getattr(self.config, "site_elements", ()) or ())
        elements.update(
            str(value) for value in getattr(self.config, "substrate_elements", ()) or ()
        )
        if elements:
            indices = [
                idx
                for idx, atom in enumerate(atoms)
                if idx not in ads_indices and atom.symbol in elements
            ]
        else:
            indices = [idx for idx in range(len(atoms)) if idx not in ads_indices]
        if indices:
            side = str(getattr(self.config, "surface_side", "top")).strip().lower()
            if side in {"top", "bottom"}:
                z_values = np.asarray([atoms.positions[idx, 2] for idx in indices], dtype=float)
                extreme = float(np.max(z_values) if side == "top" else np.min(z_values))
                default_tol = max(
                    1.2,
                    float(getattr(self.config, "surface_layer_tol", 0.5)),
                )
                layer_tol = float(
                    self.local_config.get("motif_surface_layer_tol_A", default_tol)
                )
                if side == "top":
                    filtered = [
                        idx for idx in indices if atoms.positions[idx, 2] >= extreme - layer_tol
                    ]
                else:
                    filtered = [
                        idx for idx in indices if atoms.positions[idx, 2] <= extreme + layer_tol
                    ]
                if filtered:
                    indices = filtered
        return np.asarray(indices, dtype=int)

    def _motif_group_representative_index(
        self,
        samples: Sequence[tuple[Atoms, dict[str, object]]],
        group: Sequence[int],
    ) -> int:
        if len(group) == 1:
            return int(group[0])
        features = [self._sample_feature_vector(samples[int(index)][0]) for index in group]
        if any(feature is None for feature in features):
            return int(group[len(group) // 2])

        best_index = int(group[0])
        best_mean = float("inf")
        for local_index, feature in enumerate(features):
            distances = [
                self._feature_distance(feature, other)
                for other_index, other in enumerate(features)
                if other_index != local_index
            ]
            mean_distance = float(np.mean(distances)) if distances else 0.0
            candidate_index = int(group[local_index])
            if mean_distance < best_mean:
                best_mean = mean_distance
                best_index = candidate_index
        return best_index

    @staticmethod
    def _feature_distance(
        left: np.ndarray | None,
        right: np.ndarray | None,
    ) -> float:
        if left is None or right is None:
            return float("inf")
        if left.shape != right.shape:
            return float("inf")
        delta = left - right
        return float(np.sqrt(np.mean(delta * delta)))

    @classmethod
    def _sample_feature_vector(cls, atoms: Atoms) -> np.ndarray | None:
        ads_indices = cls._adsorbate_indices(atoms)
        if not ads_indices:
            return None
        anchor_idx = cls._anchor_index(atoms)
        if not (0 <= anchor_idx < len(atoms)):
            return None
        rel = np.asarray(
            atoms.get_distances(anchor_idx, ads_indices, mic=True, vector=True),
            dtype=float,
        )
        order = sorted(
            range(len(ads_indices)),
            key=lambda idx: (
                atoms[int(ads_indices[idx])].symbol,
                float(rel[idx, 0]),
                float(rel[idx, 1]),
                float(rel[idx, 2]),
            ),
        )
        rel = rel[np.asarray(order, dtype=int)]
        anchor_position = np.asarray(atoms.positions[anchor_idx], dtype=float)
        return np.concatenate([anchor_position, rel.reshape(-1)])

    @staticmethod
    def _adsorbate_indices(atoms: Atoms) -> list[int]:
        if "reaction_is_adsorbate" in atoms.arrays:
            mask = np.asarray(atoms.arrays["reaction_is_adsorbate"], dtype=bool)
            return [int(idx) for idx, value in enumerate(mask) if value]
        tags = np.asarray(atoms.get_tags(), dtype=int)
        if tags.size:
            return [int(idx) for idx, value in enumerate(tags) if value != 0]
        return []

    @staticmethod
    def _row_selected(
        row: dict[str, object],
        selected_states: set[str],
    ) -> bool:
        if not selected_states:
            return True
        return (
            str(row.get("state_dir", "")) in selected_states
            or str(row.get("species", "")) in selected_states
        )

    @staticmethod
    def _parse_element_list(value: object) -> tuple[str, ...]:
        if value in (None, ""):
            return ()
        if isinstance(value, str):
            return tuple(token for token in value.replace(",", " ").split() if token)
        if isinstance(value, Sequence):
            return tuple(str(token) for token in value if str(token))
        return (str(value),)

    @staticmethod
    def _read_csv(path: str | Path) -> list[dict[str, str]]:
        if not Path(path).exists():
            return []
        with Path(path).open(newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]

    @staticmethod
    def _read_atoms_list(path: Path) -> list[Atoms]:
        atoms = read(str(path), index=":")
        if isinstance(atoms, Atoms):
            return [atoms]
        return list(atoms)

    @staticmethod
    def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fields: list[str] = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    @classmethod
    def _write_candidates(
        cls,
        candidates: Sequence[tuple[Atoms, dict[str, object]]],
        traj_path: Path,
        csv_path: Path,
    ) -> None:
        traj_path.parent.mkdir(parents=True, exist_ok=True)
        if traj_path.exists():
            traj_path.unlink()
        if csv_path.exists():
            csv_path.unlink()
        if candidates:
            write(str(traj_path), [atoms for atoms, _ in candidates])
        rows = [metadata for _, metadata in candidates]
        cls._write_csv(csv_path, rows)

    def _progress(self, message: str) -> None:
        line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
        if bool(self.local_config.get("progress_stdout", True)):
            print(line, flush=True)
        progress_log = self.local_config.get("progress_log")
        if progress_log in (None, ""):
            return
        path = Path(str(progress_log))
        if not path.is_absolute():
            path = Path(str(self.config.output_dir)) / path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as handle:
            handle.write(line + "\n")
