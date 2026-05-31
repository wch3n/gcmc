"""High-level reaction post-processing workflow."""

from __future__ import annotations

import csv
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import numpy as np
from ase import Atoms
from ase.io import read, write

from gcmc.analysis import LocalAdsorptionMotifAnalyzer

from .che import OERCHESummarizer
from .config import load_reaction_postprocess_config
from .local_cmc import ReactionLocalCMCWorkflow
from .output_files import output_path
from .parent_sites import (
    aggregate_parent_site_rows,
    canonical_parent_site_id,
    representative_score,
    safe_float,
    site_directory_name,
    temperature_label,
    write_csv,
)
from .reference_thermo import run_reference_thermo_stage
from .relax import ReactionStateRelaxer
from .states import ReactionCandidateGenerator
from .vibrations import ReactionStateVibrationWorkflow


_REACTION_STATE_NOTES = {
    "00_clean": (
        "Clean parent-conditioned slab. The parent adsorbate is removed before "
        "MLIP relaxation so CHE steps can reference G(*)."
    ),
    "01_OH": (
        "Parent OH* state. Use this directory for MLIP relaxation of the "
        "selected OH* representative structures."
    ),
    "02_O": (
        "O* state derived from the parent OH* site. Include the direct "
        "deprotonated candidate and any nearby/site-shifted O* basins."
    ),
    "03_OOH": (
        "OOH* state derived from the parent-conditioned O* ensemble. Include "
        "nearby/site-shifted OOH* candidates when the local basin can move."
    ),
}


_SUMMARY_COLUMNS = (
    "site_id",
    "site_population_rank",
    "site_type",
    "support_indices_sorted",
    "support_key_mode",
    "samples_total",
    "population_total",
    "shell1_key_mode",
    "shell2_key_mode",
    "functional_count_mode",
    "motif_key_mode",
    "anchor_site_xy_dist_A_mean",
    "anchor_support_min_dist_A_mean",
    "anchor_z_offset_A_mean",
)

_REPRESENTATIVE_COLUMNS = (
    "site_id",
    "site_population_rank",
    "population_total",
    "site_type",
    "support_indices_sorted",
    "support_key_mode",
    "shell1_key_mode",
    "shell2_key_mode",
    "functional_count_mode",
    "representative_rank_within_group",
    "representative_score",
    "representative_traj",
    "representative_frame",
    "representative_temperature_K",
    "representative_anchor_index",
    "representative_anchor_site_xy_dist_A",
)

_SITE_REPRESENTATIVE_COLUMNS = _REPRESENTATIVE_COLUMNS


def _select_columns(
    rows: Sequence[dict[str, object]],
    columns: Sequence[str],
) -> list[dict[str, object]]:
    return [
        {column: row.get(column, "") for column in columns if column in row}
        for row in rows
    ]


class ReactionPostProcessingWorkflow:
    """Reusable post-processing workflow for MC-derived reaction ensembles.

    The first implemented stage identifies populated parent adsorbate sites
    from one or more MC/PT trajectories and writes representative structures.
    The next stage generates and optionally relaxes parent-conditioned reaction
    candidates for OER and related mechanisms.
    """

    def __init__(self, config: SimpleNamespace):
        self.config = config

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "ReactionPostProcessingWorkflow":
        return cls(load_reaction_postprocess_config(config_path))

    def _build_analyzer(self) -> LocalAdsorptionMotifAnalyzer:
        cfg = self.config
        return LocalAdsorptionMotifAnalyzer(
            site_elements=cfg.site_elements,
            substrate_elements=cfg.substrate_elements,
            functional_elements=cfg.functional_elements,
            site_types=cfg.site_types,
            surface_side=cfg.surface_side,
            layer_tol=float(cfg.surface_layer_tol),
            xy_tol=float(cfg.site_match_tol),
            support_xy_tol=float(cfg.support_xy_tol),
            termination_site_xy_tol=cfg.termination_site_xy_tol,
            vertical_offset=float(cfg.vertical_offset),
            min_termination_dist=float(cfg.termination_clearance),
            anchor_element=cfg.anchor_element,
            shell1_size=int(cfg.shell1_size),
            shell2_size=int(cfg.shell2_size),
            functional_cutoff=float(cfg.functional_cutoff),
        )

    def _analyze_parent_trajectories(
        self,
    ) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        cfg = self.config
        analyzer = self._build_analyzer()
        output_dir = Path(cfg.output_dir)
        per_trajectory_results: list[dict[str, object]] = []
        frame_rows: list[dict[str, object]] = []

        for traj in cfg.parent_traj:
            traj_path = Path(traj)
            result = analyzer.analyze_trajectory(
                traj_path,
                reference=cfg.reference,
                start=int(cfg.start),
                stop=cfg.stop,
                step=int(cfg.step),
            )
            per_trajectory_results.append(result)
            temp_label = temperature_label(result.get("temperature_K"))
            for row in result.get("local_motif_frames", []):
                row = dict(row)
                row["_temperature_label"] = temp_label
                row["_site_id"] = canonical_parent_site_id(row)
                frame_rows.append(row)

            if bool(cfg.write_per_trajectory):
                out_prefix = output_dir / traj_path.stem / str(cfg.out_prefix)
                out_prefix.parent.mkdir(parents=True, exist_ok=True)
                analyzer.export_csv(result, out_prefix)
                if bool(cfg.write_representatives):
                    analyzer.export_representatives(
                        result,
                        out_prefix,
                        group_by=(
                            "site"
                            if cfg.aggregate_group_by == "site"
                            else "motif"
                        ),
                        top_k=int(cfg.representative_top_k),
                        n_per_group=int(cfg.representative_n_per_group),
                    )

        return per_trajectory_results, frame_rows

    def _write_aggregate_outputs(
        self,
        frame_rows: Sequence[dict[str, object]],
    ) -> dict[str, str]:
        cfg = self.config
        output_dir = Path(cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        temperature_labels = [
            temperature_label(result_temperature)
            for result_temperature in sorted(
                {
                    safe_float(row.get("temperature_K"))
                    for row in frame_rows
                    if np.isfinite(safe_float(row.get("temperature_K")))
                }
            )
        ]
        summary_rows = aggregate_parent_site_rows(
            frame_rows,
            temperature_labels=temperature_labels,
        )
        summary_rows = self._annotate_site_directories(summary_rows)
        summary_path = output_path(cfg, "summary_csv")

        representative_csv = output_path(cfg, "representatives_csv")
        representative_traj = output_path(cfg, "representatives_traj")
        representative_rows = self._select_aggregate_representatives(
            frame_rows,
            summary_rows,
        )
        site_manifest_path = output_path(cfg, "site_manifest_csv")
        if bool(getattr(cfg, "write_site_directories", True)):
            self._write_site_directories(representative_rows, site_manifest_path)

        temperature_columns: list[str] = []
        for label in temperature_labels:
            temperature_columns.extend([f"samples_{label}", f"population_{label}"])
        summary_columns = (*_SUMMARY_COLUMNS, *temperature_columns)

        write_csv(summary_path, _select_columns(summary_rows, summary_columns))
        write_csv(
            representative_csv,
            _select_columns(representative_rows, _REPRESENTATIVE_COLUMNS),
        )
        self._write_representative_traj(representative_rows, representative_traj)

        paths = {
            "summary_csv": str(summary_path),
            "representatives_csv": str(representative_csv),
            "representatives_traj": str(representative_traj),
        }
        if bool(getattr(cfg, "write_site_directories", True)):
            paths["site_manifest_csv"] = str(site_manifest_path)
            generator = ReactionCandidateGenerator(cfg)
            candidate_paths = generator.generate(site_manifest_path)
            paths.update(candidate_paths)
            candidate_manifest = candidate_paths.get(
                "candidate_manifest_csv",
                str(output_path(cfg, "candidate_manifest_csv")),
            )
            screen_outputs = self._run_parent_stability_screen(candidate_manifest)
            paths.update(screen_outputs)
            candidate_manifest = screen_outputs.get(
                "parent_stability_candidate_manifest_csv",
                candidate_manifest,
            )
            if screen_outputs and not self._manifest_has_rows(candidate_manifest):
                return paths
            local_cmc = ReactionLocalCMCWorkflow(cfg)
            paths.update(local_cmc.run(candidate_manifest))
            relaxer = ReactionStateRelaxer(cfg)
            relax_outputs = relaxer.relax(candidate_manifest)
            paths.update(relax_outputs)
            vibrations = ReactionStateVibrationWorkflow(cfg)
            paths.update(
                vibrations.run(
                    relax_outputs.get(
                        "state_relaxation_manifest_csv",
                        str(output_path(cfg, "state_relaxation_manifest_csv")),
                    ),
                )
            )
            paths.update(run_reference_thermo_stage(cfg))
            che = OERCHESummarizer(cfg)
            paths.update(che.summarize(candidate_manifest))
        return paths

    def _run_parent_stability_screen(
        self,
        candidate_manifest: str | Path,
    ) -> dict[str, str]:
        cfg = self.config
        screen = getattr(cfg, "parent_stability_screen", {}) or {}
        if not isinstance(screen, dict) or not bool(screen.get("enabled", False)):
            return {}

        state = str(screen.get("state", "01_OH"))
        screen_cfg = self._screen_config(state, screen)
        relax_outputs = ReactionStateRelaxer(screen_cfg).relax(candidate_manifest)
        vibration_outputs = ReactionStateVibrationWorkflow(screen_cfg).run(
            relax_outputs.get(
                "state_relaxation_manifest_csv",
                str(output_path(screen_cfg, "state_relaxation_manifest_csv")),
            )
        )
        vibration_summary = vibration_outputs.get(
            "vibration_summary_csv",
            str(output_path(screen_cfg, "vibration_summary_csv")),
        )
        filtered_manifest = self._filter_candidate_manifest_by_parent_vibrations(
            candidate_manifest,
            vibration_summary,
            state,
            screen,
        )
        outputs = {
            "parent_stability_candidate_manifest_csv": str(filtered_manifest),
        }
        outputs.update(
            {
                f"parent_stability_{key}": value
                for key, value in relax_outputs.items()
            }
        )
        outputs.update(
            {
                f"parent_stability_{key}": value
                for key, value in vibration_outputs.items()
            }
        )
        return outputs

    def _screen_config(self, state: str, screen: dict[str, object]) -> SimpleNamespace:
        values = deepcopy(vars(self.config))
        screen_states = ["00_clean", state]
        state_relaxation = deepcopy(values.get("state_relaxation", {}) or {})
        state_relaxation["enabled"] = True
        state_relaxation["states"] = screen_states
        if "skip_existing" in screen:
            state_relaxation["skip_existing"] = bool(screen["skip_existing"])
        values["state_relaxation"] = state_relaxation

        vibrations = deepcopy(values.get("vibrations", {}) or {})
        vibrations["enabled"] = True
        vibrations["states"] = screen_states
        if "skip_existing" in screen:
            vibrations["skip_existing"] = bool(screen["skip_existing"])
        values["vibrations"] = vibrations
        return SimpleNamespace(**values)

    def _filter_candidate_manifest_by_parent_vibrations(
        self,
        candidate_manifest: str | Path,
        vibration_summary: str | Path,
        state: str,
        screen: dict[str, object],
    ) -> Path:
        candidate_manifest = Path(candidate_manifest)
        rows = self._read_csv(candidate_manifest)
        vibration_rows = self._read_csv(vibration_summary)
        ready_sites = {
            str(row.get("site_id", ""))
            for row in vibration_rows
            if str(row.get("state_dir", "")) == state
            and self._as_bool(row.get("ready"))
        }
        filtered = [
            row
            for row in rows
            if str(row.get("site_id", "")) in ready_sites
        ]
        output_manifest = Path(
            str(screen.get("output_manifest", "candidate_manifest_parent_stable.csv"))
        )
        if not output_manifest.is_absolute():
            output_manifest = Path(str(self.config.output_dir)) / output_manifest
        self._write_manifest_csv(output_manifest, filtered, rows)
        return output_manifest

    @staticmethod
    def _read_csv(path: str | Path) -> list[dict[str, str]]:
        path = Path(path)
        if not path.exists():
            return []
        with path.open(newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]

    @staticmethod
    def _write_manifest_csv(
        path: Path,
        rows: Sequence[dict[str, object]],
        template_rows: Sequence[dict[str, object]],
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fields: list[str] = []
        for row in list(template_rows) + list(rows):
            for key in row:
                if key not in fields:
                    fields.append(key)
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    @classmethod
    def _manifest_has_rows(cls, path: str | Path) -> bool:
        return bool(cls._read_csv(path))

    @staticmethod
    def _as_bool(value: object) -> bool:
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "y"}

    def _site_paths(self, site_id: object) -> dict[str, str]:
        cfg = self.config
        sites_root = Path(str(getattr(cfg, "sites_dir", "sites")))
        if not sites_root.is_absolute():
            sites_root = Path(cfg.output_dir) / sites_root
        site_dir = sites_root / site_directory_name(site_id)
        reactions_dir = site_dir / str(getattr(cfg, "reactions_dir", "reactions"))
        return {
            "site_dir_name": site_dir.name,
            "site_dir": str(site_dir),
            "reactions_dir": str(reactions_dir),
        }

    def _reaction_state_dirs(self) -> tuple[str, ...]:
        values = getattr(self.config, "reaction_state_dirs", None)
        if not values:
            return ("00_clean", "01_OH", "02_O", "03_OOH")
        if isinstance(values, str):
            return (values,)
        return tuple(str(value) for value in values)

    def _annotate_site_directories(
        self,
        summary_rows: Sequence[dict[str, object]],
    ) -> list[dict[str, object]]:
        annotated: list[dict[str, object]] = []
        for rank, row in enumerate(summary_rows, start=1):
            output = dict(row)
            output["site_population_rank"] = int(rank)
            if bool(getattr(self.config, "write_site_directories", True)):
                output.update(self._site_paths(output["site_id"]))
            annotated.append(output)
        return annotated

    def _select_aggregate_representatives(
        self,
        frame_rows: Sequence[dict[str, object]],
        summary_rows: Sequence[dict[str, object]],
    ) -> list[dict[str, object]]:
        grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in frame_rows:
            grouped[str(row["_site_id"])].append(dict(row))

        selected: list[dict[str, object]] = []
        top_k = int(self.config.representative_top_k)
        n_per_group = int(self.config.representative_n_per_group)
        for rank, summary in enumerate(summary_rows[:top_k], start=1):
            site_id = str(summary["site_id"])
            rows = grouped.get(site_id, [])
            scored = sorted(
                ((representative_score(row, rows), row) for row in rows),
                key=lambda item: (item[0], int(item[1].get("frame", 0))),
            )
            used_frames: set[tuple[str, int]] = set()
            chosen = 0
            for score, row in scored:
                frame_key = (str(row["traj"]), int(row["frame"]))
                if frame_key in used_frames:
                    continue
                output = dict(summary)
                output.update(
                    {
                        "rank": int(rank),
                        "representative_rank_within_group": int(chosen),
                        "representative_score": float(score),
                        "representative_traj": str(row["traj"]),
                        "representative_frame": int(row["frame"]),
                        "representative_temperature_K": safe_float(
                            row.get("temperature_K")
                        ),
                        "representative_anchor_index": int(row["anchor_index"]),
                        "representative_anchor_site_xy_dist_A": safe_float(
                            row.get("anchor_site_xy_dist_A")
                        ),
                    }
                )
                selected.append(output)
                used_frames.add(frame_key)
                chosen += 1
                if chosen >= n_per_group:
                    break
        return selected

    def _write_site_directories(
        self,
        representative_rows: Sequence[dict[str, object]],
        manifest_path: Path,
    ) -> None:
        if not representative_rows:
            return

        grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in representative_rows:
            grouped[str(row["site_id"])].append(dict(row))

        manifest_rows: list[dict[str, object]] = []
        for site_id, rows in grouped.items():
            first = rows[0]
            site_dir = Path(str(first["site_dir"]))
            reactions_dir = Path(str(first["reactions_dir"]))
            site_dir.mkdir(parents=True, exist_ok=True)
            reactions_dir.mkdir(parents=True, exist_ok=True)

            site_representatives_csv = site_dir / "representatives.csv"
            site_representatives_traj = site_dir / "representatives.traj"
            site_metadata = site_dir / "site.yaml"
            reaction_readme = reactions_dir / "README.md"
            reaction_state_dirs = self._reaction_state_dirs()
            reaction_state_paths = [
                reactions_dir / state_dir for state_dir in reaction_state_dirs
            ]
            for row in rows:
                row["site_representatives_csv"] = str(site_representatives_csv)
                row["site_representatives_traj"] = str(site_representatives_traj)
                row["site_metadata"] = str(site_metadata)
                row["reaction_state_dirs"] = " ".join(reaction_state_dirs)

            write_csv(
                site_representatives_csv,
                _select_columns(rows, _SITE_REPRESENTATIVE_COLUMNS),
            )
            self._write_representative_traj(rows, site_representatives_traj)
            for state_path in reaction_state_paths:
                state_path.mkdir(parents=True, exist_ok=True)
                self._write_reaction_state_readme(state_path / "README.md", first)
            self._write_site_metadata(
                site_metadata,
                first,
                rows,
                reaction_state_dirs,
            )
            self._write_reaction_readme(
                reaction_readme,
                first,
                reaction_state_dirs,
            )
            manifest_rows.append(
                {
                    "site_id": site_id,
                    "site_population_rank": first.get("site_population_rank", ""),
                    "site_type": first.get("site_type", ""),
                    "support_indices_sorted": first.get(
                        "support_indices_sorted",
                        "",
                    ),
                    "population_total": first.get("population_total", ""),
                    "site_dir": str(site_dir),
                    "reactions_dir": str(reactions_dir),
                    "site_representatives_csv": str(site_representatives_csv),
                    "site_representatives_traj": str(site_representatives_traj),
                    "reaction_state_dirs": " ".join(reaction_state_dirs),
                }
            )

        write_csv(manifest_path, manifest_rows)

    @staticmethod
    def _metadata_value(value: object) -> object:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        return value

    @classmethod
    def _metadata_dict(cls, row: dict[str, object]) -> dict[str, object]:
        return {key: cls._metadata_value(value) for key, value in row.items()}

    @classmethod
    def _site_summary_dict(cls, row: dict[str, object]) -> dict[str, object]:
        excluded_prefixes = ("representative_", "site_representatives_")
        excluded_keys = {"rank", "site_metadata"}
        return {
            key: cls._metadata_value(value)
            for key, value in row.items()
            if key not in excluded_keys
            and not any(key.startswith(prefix) for prefix in excluded_prefixes)
        }

    @classmethod
    def _write_site_metadata(
        cls,
        path: Path,
        summary_row: dict[str, object],
        representative_rows: Sequence[dict[str, object]],
        reaction_state_dirs: Sequence[str],
    ) -> None:
        import yaml

        representative_fields = (
            "rank",
            "representative_rank_within_group",
            "representative_score",
            "representative_traj",
            "representative_frame",
            "representative_temperature_K",
            "representative_anchor_index",
        )
        metadata = {
            "site": cls._site_summary_dict(summary_row),
            "parent_representatives": {
                "representatives_csv": summary_row.get(
                    "site_representatives_csv",
                    "",
                ),
                "representatives_traj": summary_row.get(
                    "site_representatives_traj",
                    "",
                ),
                "n_representatives": len(representative_rows),
            },
            "reactions": {
                "directory": summary_row.get("reactions_dir", ""),
                "state_directories": [
                    {
                        "name": state_dir,
                        "directory": str(
                            Path(str(summary_row.get("reactions_dir", "")))
                            / state_dir
                        ),
                        "note": _REACTION_STATE_NOTES.get(state_dir, ""),
                    }
                    for state_dir in reaction_state_dirs
                ],
            },
            "representatives": [
                {
                    key: cls._metadata_value(row.get(key, ""))
                    for key in representative_fields
                }
                for row in representative_rows
            ],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(metadata, sort_keys=False))

    @staticmethod
    def _write_reaction_readme(
        path: Path,
        row: dict[str, object],
        reaction_state_dirs: Sequence[str],
    ) -> None:
        site_id = row.get("site_id", "")
        state_lines = "\n".join(f"- `{state_dir}/`" for state_dir in reaction_state_dirs)
        text = f"""# Reaction Children for {site_id}

Place downstream site-conditioned reaction calculations for this parent site
under this directory.

OER state layout:

{state_lines}

Use each state directory for MLIP relaxations of that intermediate. For `02_O/`
and `03_OOH/`, include nearby or site-shifted candidate basins when the
deprotonated/intermediate structure is not confined to the original OH* anchor.

The parent representative structures are in `../representatives.traj`.
"""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    @staticmethod
    def _write_reaction_state_readme(path: Path, row: dict[str, object]) -> None:
        site_id = row.get("site_id", "")
        state_name = path.parent.name
        note = _REACTION_STATE_NOTES.get(
            state_name,
            "Use this directory for MLIP relaxations of this intermediate state.",
        )
        text = f"""# {state_name} for {site_id}

{note}

Recommended contents:

- input structures or candidate subdirectories;
- relaxation configs and run scripts;
- relaxed structures and energies;
- provenance linking back to `../../representatives.traj`.
"""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    @staticmethod
    def _write_representative_traj(
        representative_rows: Sequence[dict[str, object]],
        output_path: Path,
    ) -> None:
        if not representative_rows:
            return
        atoms_list: list[Atoms] = []
        for row in representative_rows:
            atoms = read(
                str(row["representative_traj"]),
                index=int(row["representative_frame"]),
            )
            atoms.info["parent_site_rank"] = int(row["rank"])
            atoms.info["parent_site_id"] = str(row["site_id"])
            atoms.info["parent_site_population"] = float(row["population_total"])
            atoms.info["source_traj"] = str(row["representative_traj"])
            atoms.info["source_frame"] = int(row["representative_frame"])
            for row_key, info_key in (
                ("site_dir", "parent_site_dir"),
                ("reactions_dir", "parent_reactions_dir"),
                ("site_dir_name", "parent_site_dir_name"),
            ):
                if row_key in row:
                    atoms.info[info_key] = str(row[row_key])
            atoms_list.append(atoms)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write(str(output_path), atoms_list)

    def run(self) -> dict[str, object]:
        _, frame_rows = self._analyze_parent_trajectories()
        if not frame_rows:
            raise RuntimeError("No parent adsorbate frames were identified.")
        paths = self._write_aggregate_outputs(frame_rows)
        return {
            "n_parent_samples": int(len(frame_rows)),
            "output_paths": paths,
        }
