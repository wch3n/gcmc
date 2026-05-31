"""Configuration loading for reaction post-processing workflows."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from .che import default_che_config
from .local_cmc import default_local_cmc_config
from .reference_thermo import default_reference_thermo_stage_config
from .relax import default_state_relaxation_config
from .states import default_candidate_generation_config
from .vibrations import default_vibration_config


DEFAULT_REACTION_POSTPROCESS_CONFIG = {
    "parent_traj": [],
    "reference": None,
    "start": 0,
    "stop": None,
    "step": 1,
    "site_elements": [],
    "substrate_elements": [],
    "functional_elements": ["O"],
    "site_types": ["atop", "fcc", "hcp"],
    "surface_side": "top",
    "anchor_element": "O",
    "shell1_size": 6,
    "shell2_size": 6,
    "functional_cutoff": 3.0,
    "surface_layer_tol": 0.5,
    "site_match_tol": 0.6,
    "support_xy_tol": 1.2,
    "termination_site_xy_tol": 2.2,
    "vertical_offset": 1.5,
    "termination_clearance": 0.8,
    "output_dir": "reaction_postprocess",
    "out_prefix": "parent_sites",
    "write_per_trajectory": True,
    "write_representatives": True,
    "write_site_directories": True,
    "sites_dir": "sites",
    "reactions_dir": "reactions",
    "reaction_state_dirs": ["00_clean", "01_OH", "02_O", "03_OOH"],
    "representative_top_k": 10,
    "representative_n_per_group": 1,
    "aggregate_group_by": "site",
    "calculator": "lj",
    "lj_cutoff": 6.0,
    "model": None,
    "model_file": None,
    "device": "cpu",
    "use_kokkos": True,
    "candidate_generation": default_candidate_generation_config(),
    "parent_stability_screen": {
        "enabled": False,
        "state": "01_OH",
        "output_manifest": "candidate_manifest_parent_stable.csv",
        "skip_existing": False,
    },
    "local_cmc": default_local_cmc_config(),
    "state_relaxation": default_state_relaxation_config(),
    "vibrations": default_vibration_config(),
    "reference_thermo": default_reference_thermo_stage_config(),
    "che": default_che_config(),
}


def deep_update(base: dict, updates: dict) -> dict:
    merged = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def resolve_optional_path(value, base_dir: Path) -> str | None:
    if value is None:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return str(path)


def resolve_path_list(values, base_dir: Path) -> list[str]:
    paths: list[str] = []
    for value in as_list(values):
        path = Path(value)
        if not path.is_absolute():
            path = (base_dir / path).resolve()
        paths.append(str(path))
    return paths


def load_reaction_postprocess_config(config_path: str | Path) -> SimpleNamespace:
    import yaml

    config_path = Path(config_path).resolve()
    raw = yaml.safe_load(config_path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError("Reaction post-processing config must be a mapping.")

    if any(
        key in raw
        for key in (
            "parent",
            "site_analysis",
            "selection",
            "calculator",
            "output",
            "candidate_generation",
            "parent_stability_screen",
            "local_cmc",
            "state_relaxation",
            "vibrations",
            "reference_thermo",
            "che",
        )
    ):
        merged = deep_update(
            {
                "parent": {},
                "site_analysis": {},
                "selection": {},
                "calculator": {},
                "output": {},
                "candidate_generation": {},
                "parent_stability_screen": {},
                "local_cmc": {},
                "state_relaxation": {},
                "vibrations": {},
                "reference_thermo": {},
                "che": {},
            },
            raw,
        )
        flat = dict(DEFAULT_REACTION_POSTPROCESS_CONFIG)
        flat["candidate_generation"] = deep_update(
            default_candidate_generation_config(),
            merged["candidate_generation"],
        )
        flat["state_relaxation"] = deep_update(
            default_state_relaxation_config(),
            merged["state_relaxation"],
        )
        flat["parent_stability_screen"] = deep_update(
            DEFAULT_REACTION_POSTPROCESS_CONFIG["parent_stability_screen"],
            merged["parent_stability_screen"],
        )
        flat["local_cmc"] = deep_update(
            default_local_cmc_config(),
            merged["local_cmc"],
        )
        flat["vibrations"] = deep_update(
            default_vibration_config(),
            merged["vibrations"],
        )
        flat["reference_thermo"] = deep_update(
            default_reference_thermo_stage_config(),
            merged["reference_thermo"],
        )
        flat["che"] = deep_update(
            default_che_config(),
            merged["che"],
        )
        parent = merged["parent"]
        if "traj" in parent:
            parent = dict(parent)
            parent["parent_traj"] = parent.pop("traj")
        for section in (
            parent,
            merged["site_analysis"],
            merged["selection"],
            merged["calculator"],
            merged["output"],
        ):
            flat.update(section)
    else:
        flat = dict(DEFAULT_REACTION_POSTPROCESS_CONFIG)
        flat["candidate_generation"] = default_candidate_generation_config()
        flat["parent_stability_screen"] = dict(
            DEFAULT_REACTION_POSTPROCESS_CONFIG["parent_stability_screen"]
        )
        flat["local_cmc"] = default_local_cmc_config()
        flat["state_relaxation"] = default_state_relaxation_config()
        flat["vibrations"] = default_vibration_config()
        flat["reference_thermo"] = default_reference_thermo_stage_config()
        flat["che"] = default_che_config()
        flat.update(raw)
        if isinstance(raw.get("candidate_generation"), dict):
            flat["candidate_generation"] = deep_update(
                default_candidate_generation_config(),
                raw["candidate_generation"],
            )
        if isinstance(raw.get("parent_stability_screen"), dict):
            flat["parent_stability_screen"] = deep_update(
                DEFAULT_REACTION_POSTPROCESS_CONFIG["parent_stability_screen"],
                raw["parent_stability_screen"],
            )
        if isinstance(raw.get("local_cmc"), dict):
            flat["local_cmc"] = deep_update(
                default_local_cmc_config(),
                raw["local_cmc"],
            )
        if isinstance(raw.get("state_relaxation"), dict):
            flat["state_relaxation"] = deep_update(
                default_state_relaxation_config(),
                raw["state_relaxation"],
            )
        if isinstance(raw.get("vibrations"), dict):
            flat["vibrations"] = deep_update(
                default_vibration_config(),
                raw["vibrations"],
            )
        if isinstance(raw.get("reference_thermo"), dict):
            flat["reference_thermo"] = deep_update(
                default_reference_thermo_stage_config(),
                raw["reference_thermo"],
            )
        if isinstance(raw.get("che"), dict):
            flat["che"] = deep_update(
                default_che_config(),
                raw["che"],
            )

    if "site_type" in flat and "site_types" not in flat:
        flat["site_types"] = flat.pop("site_type")
    if "prefix" in flat and "out_prefix" not in flat:
        flat["out_prefix"] = flat.pop("prefix")
    if "min_termination_dist" in flat and "termination_clearance" not in flat:
        flat["termination_clearance"] = flat.pop("min_termination_dist")

    flat["parent_traj"] = resolve_path_list(flat.get("parent_traj"), config_path.parent)
    if not flat["parent_traj"]:
        raise ValueError("Reaction post-processing requires parent.traj.")
    flat["reference"] = resolve_optional_path(flat.get("reference"), config_path.parent)
    flat["output_dir"] = resolve_optional_path(flat.get("output_dir"), config_path.parent)
    for key in ("model", "model_file"):
        flat[key] = resolve_optional_path(flat.get(key), config_path.parent)
    for section_key in (
        "candidate_generation",
        "local_cmc",
        "state_relaxation",
        "vibrations",
    ):
        section = flat.get(section_key)
        if isinstance(section, dict):
            for path_key in ("model", "model_file"):
                if path_key in section and section[path_key] not in (None, ""):
                    section[path_key] = resolve_optional_path(
                        section[path_key],
                        config_path.parent,
                    )
            log_file = section.get("log_file")
            if section_key == "state_relaxation" and log_file not in (None, ""):
                log_path = Path(str(log_file))
                if log_path.parent != Path("."):
                    section["log_file"] = resolve_optional_path(
                        log_file,
                        config_path.parent,
                    )
            progress_log = section.get("progress_log")
            if section_key in {"local_cmc", "state_relaxation", "vibrations"} and progress_log not in (None, ""):
                progress_path = Path(str(progress_log))
                if progress_path.parent != Path("."):
                    section["progress_log"] = resolve_optional_path(
                        progress_log,
                        config_path.parent,
                    )

    reference_thermo = flat.get("reference_thermo")
    if isinstance(reference_thermo, dict):
        for molecule_key in ("h2", "h2o", "o2"):
            molecule = reference_thermo.get(molecule_key)
            if isinstance(molecule, dict):
                atoms_path = molecule.get("atoms")
                if atoms_path not in (None, ""):
                    molecule["atoms"] = resolve_optional_path(
                        atoms_path,
                        config_path.parent,
                    )
        calculator = reference_thermo.get("calculator")
        if isinstance(calculator, dict):
            for path_key in ("model", "model_file"):
                if calculator.get(path_key) not in (None, ""):
                    calculator[path_key] = resolve_optional_path(
                        calculator[path_key],
                        config_path.parent,
                    )

    che_section = flat.get("che")
    if isinstance(che_section, dict):
        vibration_summary = che_section.get("vibration_summary_csv")
        if vibration_summary not in (None, ""):
            che_section["vibration_summary_csv"] = resolve_optional_path(
                vibration_summary,
                config_path.parent,
            )
        for thermo_key in ("h2_thermo", "h2o_thermo", "o2_thermo"):
            thermo = che_section.get(thermo_key)
            if isinstance(thermo, dict):
                atoms_path = thermo.get("atoms")
                if atoms_path not in (None, ""):
                    thermo["atoms"] = resolve_optional_path(
                        atoms_path,
                        config_path.parent,
                    )

    flat["site_elements"] = tuple(flat.get("site_elements") or ())
    flat["substrate_elements"] = tuple(flat.get("substrate_elements") or ())
    flat["functional_elements"] = tuple(flat.get("functional_elements") or ())
    flat["site_types"] = tuple(flat.get("site_types") or ())
    flat["reaction_state_dirs"] = tuple(
        str(value) for value in as_list(flat.get("reaction_state_dirs"))
    )
    if not flat["site_elements"]:
        raise ValueError("Reaction post-processing requires site_elements.")
    if not flat["substrate_elements"]:
        raise ValueError("Reaction post-processing requires substrate_elements.")
    if not flat["site_types"]:
        raise ValueError("Reaction post-processing requires site_types.")
    if str(flat.get("aggregate_group_by", "site")) != "site":
        raise ValueError("Only aggregate_group_by='site' is currently supported.")

    return SimpleNamespace(**flat)
