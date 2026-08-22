#!/usr/bin/env python3
"""Prepare canonical adsorbate-CMC input directories from slab snapshots."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import yaml
from ase.io import iread, read, write
from ase.io.trajectory import Trajectory


def _deep_update(base: dict, updates: dict) -> dict:
    result = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = value
    return result


def _parse_int_set(text: str) -> list[int]:
    if isinstance(text, (list, tuple)):
        values = [int(value) for value in text]
        if not values:
            raise ValueError("At least one integer value is required.")
        return list(dict.fromkeys(values))
    values: list[int] = []
    for token in text.replace(",", " ").split():
        if ":" in token:
            parts = token.split(":")
            if len(parts) not in {2, 3}:
                raise ValueError(f"Invalid range token: {token!r}")
            start = int(parts[0])
            stop = int(parts[1])
            step = int(parts[2]) if len(parts) == 3 and parts[2] else 1
            values.extend(range(start, stop, step))
        else:
            values.append(int(token))
    if not values:
        raise ValueError("At least one integer value is required.")
    return list(dict.fromkeys(values))


def _parse_symbols(text: str | Iterable[str]) -> list[str]:
    if text is None:
        return []
    if isinstance(text, str):
        return [token for token in text.replace(",", " ").split() if token]
    symbols: list[str] = []
    for value in text:
        symbols.extend(_parse_symbols(value))
    return symbols


def _parse_yaml_mapping(value: str | dict) -> dict:
    if isinstance(value, dict):
        return value
    parsed = yaml.safe_load(value)
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("Expected a YAML mapping.")
    return parsed


def _parse_dat(path: Path | None) -> list[tuple[float, float]]:
    if path is None:
        return []
    rows: list[tuple[float, float]] = []
    with path.open() as handle:
        for line in handle:
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                rows.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue
    return rows


def _count_frames(path: Path) -> int:
    if path.suffix == ".traj":
        with Trajectory(path) as traj:
            return len(traj)
    count = 0
    for _ in iread(path, index=":"):
        count += 1
    return max(count, 1)


def _select_distinct_frames(
    *,
    total_frames: int,
    n_frames: int,
    frame_start: int = 0,
    frame_stop: int | None = None,
) -> list[int]:
    if n_frames <= 0:
        raise ValueError("n_frames must be positive.")
    if total_frames <= 0:
        raise ValueError("Input trajectory does not contain any frames.")
    start = int(frame_start)
    if start < 0:
        start += total_frames
    stop = total_frames - 1 if frame_stop is None else int(frame_stop)
    if stop < 0:
        stop += total_frames
    if start < 0 or stop < 0 or start >= total_frames or stop >= total_frames:
        raise ValueError(
            f"Frame window [{start}, {stop}] is outside trajectory with "
            f"{total_frames} frame(s)."
        )
    if stop < start:
        raise ValueError(f"frame_stop ({stop}) must be >= frame_start ({start}).")
    available = stop - start + 1
    if n_frames > available:
        raise ValueError(
            f"Cannot select {n_frames} distinct frame(s) from a window with "
            f"only {available} frame(s)."
        )
    if n_frames == 1:
        return [start]
    frames = [
        start + round(i * (available - 1) / (n_frames - 1))
        for i in range(n_frames)
    ]
    return list(dict.fromkeys(frames))


def _normalize_frame_window(
    *,
    total_frames: int,
    frame_start: int = 0,
    frame_stop: int | None = None,
) -> tuple[int, int]:
    if total_frames <= 0:
        raise ValueError("Input trajectory does not contain any frames.")
    start = int(frame_start)
    if start < 0:
        start += total_frames
    stop = total_frames - 1 if frame_stop is None else int(frame_stop)
    if stop < 0:
        stop += total_frames
    if start < 0 or stop < 0 or start >= total_frames or stop >= total_frames:
        raise ValueError(
            f"Frame window [{start}, {stop}] is outside trajectory with "
            f"{total_frames} frame(s)."
        )
    if stop < start:
        raise ValueError(f"frame_stop ({stop}) must be >= frame_start ({start}).")
    return start, stop


def _soap_descriptor(atoms, args: argparse.Namespace) -> np.ndarray:
    try:
        from dscribe.descriptors import SOAP
    except ImportError as exc:
        raise RuntimeError(
            "SOAP frame selection requires the optional 'dscribe' package. "
            "Install dscribe or use source.frame_selection.method: evenly_spaced."
        ) from exc

    species = _parse_symbols(args.soap_species)
    if not species:
        species = sorted(set(atoms.get_chemical_symbols()))
    descriptor = SOAP(
        species=species,
        periodic=bool(args.soap_periodic),
        r_cut=float(args.soap_r_cut),
        n_max=int(args.soap_n_max),
        l_max=int(args.soap_l_max),
        sigma=float(args.soap_sigma),
        average=args.soap_average,
        sparse=False,
    ).create(atoms)
    vector = np.asarray(descriptor, dtype=float)
    if vector.ndim > 1:
        vector = vector.mean(axis=0)
    vector = vector.ravel()
    norm = float(np.linalg.norm(vector))
    if norm > 0.0:
        vector = vector / norm
    return vector


def _frame_descriptor(atoms, args: argparse.Namespace) -> np.ndarray:
    descriptor = str(args.frame_descriptor).lower()
    if descriptor == "soap":
        return _soap_descriptor(atoms, args)
    raise ValueError(f"Unsupported frame descriptor: {args.frame_descriptor!r}")


def _farthest_point_order(descriptors: np.ndarray, n_select: int) -> list[int]:
    if n_select <= 0:
        raise ValueError("n_frames must be positive.")
    n_candidates = len(descriptors)
    if n_select > n_candidates:
        raise ValueError(
            f"Cannot select {n_select} frame(s) from only {n_candidates} "
            "candidate frame(s)."
        )
    centroid = descriptors.mean(axis=0)
    selected = [int(np.argmin(np.linalg.norm(descriptors - centroid, axis=1)))]
    min_dist = np.linalg.norm(descriptors - descriptors[selected[0]], axis=1)
    while len(selected) < n_select:
        min_dist[selected] = -1.0
        next_idx = int(np.argmax(min_dist))
        selected.append(next_idx)
        dist = np.linalg.norm(descriptors - descriptors[next_idx], axis=1)
        min_dist = np.minimum(min_dist, dist)
    return selected


def _select_diverse_frames(
    *,
    source_snapshot: Path,
    total_frames: int,
    n_frames: int,
    frame_start: int,
    frame_stop: int | None,
    args: argparse.Namespace,
) -> list[int]:
    start, stop = _normalize_frame_window(
        total_frames=total_frames,
        frame_start=frame_start,
        frame_stop=frame_stop,
    )
    stride = int(args.frame_candidate_stride)
    if stride <= 0:
        raise ValueError("frame_candidate_stride must be positive.")
    candidate_frames = list(range(start, stop + 1, stride))
    if candidate_frames[-1] != stop:
        candidate_frames.append(stop)
    if n_frames > len(candidate_frames):
        raise ValueError(
            f"Cannot select {n_frames} frame(s) from {len(candidate_frames)} "
            "candidate frame(s). Decrease n_frames or frame_candidate_stride."
        )

    descriptors = []
    for frame in candidate_frames:
        atoms = read(source_snapshot, index=frame)
        descriptors.append(_frame_descriptor(atoms, args))
    matrix = np.vstack(descriptors)
    selected_local = _farthest_point_order(matrix, n_frames)
    return sorted(candidate_frames[idx] for idx in selected_local)


def _resolve_yaml_path(value, base_dir: Path):
    if value is None:
        return None
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else (base_dir / path).resolve()


def _defaults_from_prepare_yaml(path: Path) -> dict[str, object]:
    path = path.expanduser().resolve()
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Preparation config must be a YAML mapping: {path}")
    base_dir = path.parent

    source = raw.get("source", {}) or {}
    adsorbate = raw.get("adsorbate", {}) or {}
    sites = raw.get("sites", {}) or {}
    slab = raw.get("slab", {}) or {}
    cmc = raw.get("cmc", {}) or {}
    moves = raw.get("moves", {}) or {}
    pt = raw.get("pt", {}) or {}
    calculator = raw.get("calculator", {}) or {}
    backend = raw.get("backend", {}) or {}
    output = raw.get("output", {}) or {}
    slurm = raw.get("slurm", {}) or {}

    defaults: dict[str, object] = {}
    defaults["workflow"] = raw.get(
        "workflow", raw.get("run_mode", source.get("workflow", "cmc"))
    )
    defaults["snapshot"] = _resolve_yaml_path(source.get("snapshot"), base_dir)
    defaults["source_dat"] = _resolve_yaml_path(source.get("dat", source.get("source_dat")), base_dir)
    if source.get("out_dir") is not None:
        defaults["out_dir"] = _resolve_yaml_path(source.get("out_dir"), base_dir)
    elif output.get("root") is not None:
        defaults["out_dir"] = _resolve_yaml_path(output.get("root"), base_dir)
    if "frames" in source:
        defaults["frames"] = source["frames"]
    if "n_frames" in source:
        defaults["n_frames"] = source["n_frames"]
    elif "num_frames" in source:
        defaults["n_frames"] = source["num_frames"]
    elif "distinct_frames" in source:
        defaults["n_frames"] = source["distinct_frames"]
    if "frame_start" in source:
        defaults["frame_start"] = source["frame_start"]
    if "frame_stop" in source:
        defaults["frame_stop"] = source["frame_stop"]
    frame_selection = source.get("frame_selection", {}) or {}
    if isinstance(frame_selection, str):
        frame_selection = {"method": frame_selection}
    defaults["frame_selection_method"] = frame_selection.get("method", "evenly_spaced")
    defaults["frame_descriptor"] = frame_selection.get("descriptor", "soap")
    defaults["frame_candidate_stride"] = frame_selection.get("candidate_stride", 1)
    soap = frame_selection.get("soap", {}) or {}
    defaults["soap_species"] = soap.get("species")
    defaults["soap_r_cut"] = soap.get("r_cut", 5.0)
    defaults["soap_n_max"] = soap.get("n_max", 6)
    defaults["soap_l_max"] = soap.get("l_max", 4)
    defaults["soap_sigma"] = soap.get("sigma", 0.5)
    defaults["soap_average"] = soap.get("average", "inner")
    defaults["soap_periodic"] = soap.get("periodic", True)
    defaults["seeds"] = source.get("seeds", "67,211")

    defaults["adsorbate"] = adsorbate.get("species", adsorbate.get("adsorbate", "OH"))
    defaults["adsorbate_anchor_index"] = adsorbate.get("anchor_index", 0)
    defaults["n_adsorbates"] = adsorbate.get("n_adsorbates", 1)
    defaults["adsorbate_templates"] = adsorbate.get("templates")
    defaults["vertical_offset"] = adsorbate.get("vertical_offset", 1.5)

    defaults["site_elements"] = sites.get("elements", ["Ti", "Zr"])
    defaults["site_types"] = sites.get("types", ["atop", "fcc", "hcp"])
    defaults["surface_side"] = sites.get("surface_side", "top")
    defaults["substrate_elements"] = slab.get("substrate_elements", ["Ti", "Zr", "C"])
    defaults["functional_elements"] = slab.get("functional_elements", ["O"])

    for key in (
        "temperature",
        "nsweeps",
        "equilibration",
        "write_interval",
        "sample_interval",
        "write_debug_trajs",
        "write_attempted_traj",
        "write_accepted_traj",
        "write_rejected_traj",
        "debug_traj_interval",
        "checkpoint_interval",
        "displacement_sigma",
        "max_displacement_trials",
        "min_clearance",
        "site_match_tol",
        "support_xy_tol",
        "termination_site_xy_tol",
        "surface_layer_tol",
        "termination_clearance",
        "vertical_adjust_step",
        "max_vertical_adjust",
        "z_max_support",
        "detach_tol",
        "relax",
        "relax_steps",
        "fmax",
        "enable_hybrid_md",
        "md_move_prob",
        "md_steps",
        "md_timestep_fs",
        "md_ensemble",
        "md_accept_mode",
        "md_friction",
        "adsorbate_surface_clearance_A",
        "adsorbate_surface_xy_tol_A",
        "diagnostics",
        "diagnostics_log",
        "molecular_upright_atom_indices",
        "molecular_upright_min_z_A",
    ):
        if key in cmc:
            defaults[key] = cmc[key]
            if key == "checkpoint_interval":
                defaults.setdefault("pt_worker_checkpoint_interval", cmc[key])
    defaults["cmc_resume"] = bool(cmc.get("resume", False))

    hop = moves.get("hop", {}) or {}
    reorientation = moves.get("reorientation", {}) or {}
    puckering = hop.get("puckering", {}) or {}
    hop_reorient = hop.get("reorient", {}) or {}
    if "prob" in hop:
        defaults["hop_prob"] = hop["prob"]
    if "prob" in hop_reorient:
        defaults["hop_reorient_prob"] = hop_reorient["prob"]
    if "angle_deg" in hop_reorient:
        defaults["reorientation_angle_deg"] = hop_reorient["angle_deg"]
    if "max_trials" in hop_reorient:
        defaults["max_reorientation_trials"] = hop_reorient["max_trials"]
    if "prob" in reorientation:
        defaults["reorientation_prob"] = reorientation["prob"]
    if "angle_deg" in reorientation:
        defaults["reorientation_angle_deg"] = reorientation["angle_deg"]
    if "max_trials" in reorientation:
        defaults["max_reorientation_trials"] = reorientation["max_trials"]
    if puckering:
        defaults["puckering_prob"] = (
            puckering.get("prob", 1.0)
            if bool(puckering.get("enabled", True))
            else 0.0
        )
    if "elements" in puckering:
        defaults["puckering_elements"] = puckering["elements"]
    if "height_A" in puckering:
        defaults["puckering_height_A"] = puckering["height_A"]
    if "height_jitter_A" in puckering:
        defaults["puckering_height_jitter_A"] = puckering["height_jitter_A"]
    if "heights" in puckering:
        defaults["puckering_heights"] = puckering["heights"]
    if "max_trials" in puckering:
        defaults["max_puckering_trials"] = puckering["max_trials"]

    for key in ("calculator", "model_file", "model", "device", "use_kokkos", "lj_cutoff"):
        if key in calculator:
            defaults[key] = calculator[key]
    for key in ("output_prefix",):
        if key in output:
            defaults[key] = output[key]
    defaults["run_output_dir"] = output.get(
        "run_output_dir", output.get("task_output_dir", output.get("output_dir", "results"))
    )
    for key in ("stats_file", "results_file", "checkpoint_file", "initial_traj_file"):
        if key in output:
            defaults[key] = output[key]
    for key in (
        "T_start",
        "T_end",
        "T_step",
        "n_replicas",
        "fine_grid_temps",
        "fine_grid_weights",
        "fine_grid_strength",
        "fine_grid_width",
        "grid_space",
        "swap_stride",
        "swap_interval",
        "report_interval",
        "sampling_interval",
        "local_eq_fraction",
        "checkpoint_interval",
        "worker_checkpoint_interval",
        "seed_nonce",
        "n_cycles",
        "equilibration_cycles",
        "resume",
    ):
        if key in pt:
            defaults[f"pt_{key}"] = pt[key]
    for key in (
        "backend",
        "n_gpus",
        "workers_per_gpu",
        "ray_address",
        "ray_log_to_driver",
        "ray_num_cpus_per_task",
        "ray_num_gpus_per_task",
        "ray_actor_max_restarts",
        "ray_actor_max_task_retries",
        "ray_get_timeout_s",
        "use_placement_group",
        "placement_group_strategy",
        "remove_placement_group_on_stop",
        "shutdown_on_stop",
    ):
        if key in backend:
            defaults[key] = backend[key]
    for key in (
        "nodes",
        "repo_root",
        "partition",
        "account",
        "time",
        "gres",
        "cpus_per_task",
        "mem_per_cpu",
        "array_concurrency",
        "job_name",
        "signal_seconds_before_timeout",
        "auto_requeue",
    ):
        if key in slurm:
            defaults[key] = slurm[key]

    defaults["system_extra"] = raw.get("system_extra", {})
    defaults["cmc_extra"] = raw.get("cmc_extra", {})
    defaults["pt_extra"] = raw.get("pt_extra", {})
    defaults["backend_extra"] = raw.get("backend_extra", {})
    defaults["calculator_extra"] = raw.get("calculator_extra", {})
    defaults["output_extra"] = raw.get("output_extra", {})
    return defaults


def _snapshot_name(snapshot_id: int, frame: int) -> str:
    return f"snapshot_{snapshot_id:03d}_frame{frame:04d}.traj"


def _config_for_task(
    *,
    snapshot_path: Path,
    run_dir: Path,
    source_snapshot: Path,
    source_frame: int,
    source_cycle: float | str,
    seed: int,
    args: argparse.Namespace,
) -> dict[str, object]:
    rel_snapshot = os.path.relpath(snapshot_path, start=run_dir)
    cmc_moves: dict[str, object] = {
        "mode": "hybrid",
        "hop": {
            "prob": args.hop_prob,
            "reorient": {
                "prob": args.hop_reorient_prob,
                "angle_deg": args.reorientation_angle_deg,
                "max_trials": args.max_reorientation_trials,
            },
        },
        "reorientation": {
            "prob": args.reorientation_prob,
            "angle_deg": args.reorientation_angle_deg,
            "max_trials": args.max_reorientation_trials,
        },
    }
    if args.puckering_prob > 0.0:
        puckering_config = {
            "prob": args.puckering_prob,
            "elements": _parse_symbols(args.puckering_elements),
            "height_A": args.puckering_height_A,
            "height_jitter_A": args.puckering_height_jitter_A,
            "max_trials": args.max_puckering_trials,
        }
        if args.puckering_heights:
            puckering_config["heights"] = args.puckering_heights
        cmc_moves["hop"]["puckering"] = puckering_config

    system_config: dict[str, object] = {
        "snapshot": rel_snapshot,
        "frame": 0,
        "clear_loaded_constraints": False,
        "adsorbate": args.adsorbate,
        "adsorbate_anchor_index": args.adsorbate_anchor_index,
        "initialization_mode": "fixed_count",
        "n_adsorbates": args.n_adsorbates,
        "site_elements": _parse_symbols(args.site_elements),
        "surface_side": args.surface_side,
        "site_type": _parse_symbols(args.site_types),
        "substrate_elements": _parse_symbols(args.substrate_elements),
        "functional_elements": _parse_symbols(args.functional_elements),
        "vertical_offset": args.vertical_offset,
        "write_site_overlay": True,
        "site_overlay_include_blocked": False,
        "site_overlay_z_field": "suggested_z_A",
    }
    if getattr(args, "adsorbate_templates", None):
        system_config["adsorbate_templates"] = args.adsorbate_templates
    system_config = _deep_update(system_config, getattr(args, "system_extra", {}) or {})

    cmc_config: dict[str, object] = {
        "temperature": args.temperature,
        "nsweeps": args.nsweeps,
        "equilibration": args.equilibration,
        "write_interval": args.write_interval,
        "sample_interval": args.sample_interval,
        "write_debug_trajs": bool(args.write_debug_trajs),
        "write_attempted_traj": bool(args.write_attempted_traj),
        "write_accepted_traj": bool(args.write_accepted_traj),
        "write_rejected_traj": bool(args.write_rejected_traj),
        "debug_traj_interval": args.debug_traj_interval,
        "seed": int(seed),
        "resume": bool(args.cmc_resume),
        "checkpoint_interval": args.checkpoint_interval,
        "displacement_sigma": args.displacement_sigma,
        "max_displacement_trials": args.max_displacement_trials,
        "min_clearance": args.min_clearance,
        "site_match_tol": args.site_match_tol,
        "support_xy_tol": args.support_xy_tol,
        "termination_site_xy_tol": args.termination_site_xy_tol,
        "surface_layer_tol": args.surface_layer_tol,
        "termination_clearance": args.termination_clearance,
        "vertical_adjust_step": args.vertical_adjust_step,
        "max_vertical_adjust": args.max_vertical_adjust,
        "z_max_support": args.z_max_support,
        "detach_tol": args.detach_tol,
        "relax": bool(args.relax),
        "relax_steps": args.relax_steps,
        "relax_z_only": False,
        "fmax": args.fmax,
        "verbose_relax": False,
        "enable_hybrid_md": bool(args.enable_hybrid_md),
        "md_move_prob": args.md_move_prob,
        "md_steps": args.md_steps,
        "md_timestep_fs": args.md_timestep_fs,
        "md_ensemble": args.md_ensemble,
        "md_accept_mode": args.md_accept_mode,
        "md_friction": args.md_friction,
        "md_planar": False,
        "md_planar_axis": 2,
        "md_init_momenta": True,
        "md_remove_drift": True,
        "moves": cmc_moves,
        "adsorbate_surface_clearance_A": args.adsorbate_surface_clearance_A,
        "adsorbate_surface_xy_tol_A": args.adsorbate_surface_xy_tol_A,
        "diagnostics_enabled": bool(args.diagnostics),
        "diagnostics_log": bool(args.diagnostics_log),
    }
    if getattr(args, "molecular_upright_atom_indices", None) is not None:
        cmc_config["molecular_upright_atom_indices"] = args.molecular_upright_atom_indices
    if getattr(args, "molecular_upright_min_z_A", None) is not None:
        cmc_config["molecular_upright_min_z_A"] = args.molecular_upright_min_z_A
    cmc_config = _deep_update(cmc_config, getattr(args, "cmc_extra", {}) or {})

    calculator_config: dict[str, object] = {
        "calculator": args.calculator,
        "device": args.device,
        "use_kokkos": bool(args.use_kokkos),
    }
    if args.model_file:
        calculator_config["model_file"] = str(Path(args.model_file).expanduser())
    if args.model:
        calculator_config["model"] = str(Path(args.model).expanduser())
    if args.calculator == "lj":
        calculator_config["lj_cutoff"] = args.lj_cutoff
    calculator_config = _deep_update(
        calculator_config, getattr(args, "calculator_extra", {}) or {}
    )

    output_config = _deep_update(
        {
            "output_dir": args.run_output_dir,
            "output_prefix": args.output_prefix,
        },
        getattr(args, "output_extra", {}) or {},
    )

    config: dict[str, object] = {
        "metadata": {
            "source_snapshot": str(source_snapshot),
            "source_frame": int(source_frame),
            "source_cycle": source_cycle,
            "seed": int(seed),
        },
        "system": system_config,
        "cmc": cmc_config,
        "calculator": calculator_config,
        "output": output_config,
    }
    return config


def _pt_config_for_task(
    *,
    snapshot_path: Path,
    run_dir: Path,
    source_snapshot: Path,
    source_frame: int,
    source_cycle: float | str,
    seed: int,
    args: argparse.Namespace,
) -> dict[str, object]:
    config = _config_for_task(
        snapshot_path=snapshot_path,
        run_dir=run_dir,
        source_snapshot=source_snapshot,
        source_frame=source_frame,
        source_cycle=source_cycle,
        seed=seed,
        args=args,
    )
    cmc_config = dict(config["cmc"])
    for key in (
        "temperature",
        "nsweeps",
        "write_interval",
        "sample_interval",
        "equilibration",
        "seed",
        "resume",
    ):
        cmc_config.pop(key, None)
    cmc_config["checkpoint_interval"] = int(args.pt_worker_checkpoint_interval)

    pt_config: dict[str, object] = {
        "T_start": args.pt_T_start,
        "T_end": args.pt_T_end,
        "T_step": args.pt_T_step,
        "n_replicas": args.pt_n_replicas,
        "fine_grid_temps": args.pt_fine_grid_temps,
        "fine_grid_weights": args.pt_fine_grid_weights,
        "fine_grid_strength": args.pt_fine_grid_strength,
        "fine_grid_width": args.pt_fine_grid_width,
        "grid_space": args.pt_grid_space,
        "swap_stride": args.pt_swap_stride,
        "swap_interval": args.pt_swap_interval,
        "report_interval": args.pt_report_interval,
        "sampling_interval": args.pt_sampling_interval,
        "local_eq_fraction": args.pt_local_eq_fraction,
        "checkpoint_interval": args.pt_checkpoint_interval,
        "seed_nonce": int(seed),
        "n_cycles": args.pt_n_cycles,
        "equilibration_cycles": args.pt_equilibration_cycles,
        "resume": bool(args.pt_resume),
    }
    pt_config = _deep_update(pt_config, getattr(args, "pt_extra", {}) or {})

    backend_config: dict[str, object] = {
        "backend": args.backend,
        "n_gpus": args.n_gpus,
        "workers_per_gpu": args.workers_per_gpu,
        "ray_address": args.ray_address,
        "ray_log_to_driver": bool(args.ray_log_to_driver),
        "ray_num_cpus_per_task": args.ray_num_cpus_per_task,
        "ray_num_gpus_per_task": args.ray_num_gpus_per_task,
        "ray_actor_max_restarts": args.ray_actor_max_restarts,
        "ray_actor_max_task_retries": args.ray_actor_max_task_retries,
        "ray_get_timeout_s": args.ray_get_timeout_s,
        "use_placement_group": bool(args.use_placement_group),
        "placement_group_strategy": args.placement_group_strategy,
        "remove_placement_group_on_stop": bool(args.remove_placement_group_on_stop),
        "shutdown_on_stop": bool(args.shutdown_on_stop),
    }
    backend_config = _deep_update(
        backend_config, getattr(args, "backend_extra", {}) or {}
    )

    output_config = _deep_update(
        {
            "output_dir": args.run_output_dir,
            "stats_file": args.stats_file,
            "results_file": args.results_file,
            "checkpoint_file": args.pt_checkpoint_file,
            "initial_traj_file": args.initial_traj_file,
        },
        getattr(args, "output_extra", {}) or {},
    )

    return {
        "metadata": {
            **config["metadata"],
            "workflow": "pt",
        },
        "system": {
            **config["system"],
            "initialization_seed": int(seed),
        },
        "pt": pt_config,
        "cmc": cmc_config,
        "calculator": config["calculator"],
        "backend": backend_config,
        "output": output_config,
    }


def _write_slurm(root: Path, n_tasks: int, args: argparse.Namespace) -> None:
    array_spec = f"0-{n_tasks - 1}%{args.array_concurrency}"
    workflow = str(args.workflow).lower()
    backend = str(getattr(args, "backend", "")).lower()
    module = "gcmc.run_adsorbate_pt" if workflow == "pt" else "gcmc.run_adsorbate_cmc"
    label = "adsorbate PT" if workflow == "pt" else "adsorbate CMC"
    nodes = int(getattr(args, "nodes", 1) or 1)
    signal_seconds = int(getattr(args, "signal_seconds_before_timeout", 0) or 0)
    if signal_seconds < 0:
        raise ValueError("signal_seconds_before_timeout must be >= 0")
    graceful_pt_stop = workflow == "pt" and signal_seconds > 0
    auto_requeue = bool(getattr(args, "auto_requeue", False))
    if auto_requeue and not graceful_pt_stop:
        raise ValueError(
            "auto_requeue requires a PT workflow with "
            "signal_seconds_before_timeout > 0"
        )
    signal_directive = (
        f"#SBATCH --signal=B:USR1@{signal_seconds}\n" if graceful_pt_stop else ""
    )
    requeue_directives = (
        "#SBATCH --requeue\n#SBATCH --open-mode=append\n"
        if auto_requeue
        else ""
    )
    signal_handler = (
        """
driver_pid=""
requeue_requested=0

forward_stop_signal() {
  requeue_requested=1
  if [[ -n "${driver_pid}" ]] && kill -0 "${driver_pid}" 2>/dev/null; then
    echo "Received SIGUSR1; requesting a graceful PT checkpoint and stop."
    kill -USR1 "${driver_pid}"
  fi
}
trap forward_stop_signal USR1
"""
        if graceful_pt_stop
        else ""
    )
    requeue_block = (
        """
if [[ "${requeue_requested}" -eq 1 ]]; then
  if [[ "${driver_status}" -ne 0 ]]; then
    echo "PT driver failed after SIGUSR1; refusing automatic requeue." >&2
    exit "${driver_status}"
  fi
  if declare -F cleanup >/dev/null 2>&1; then
    cleanup
    trap - EXIT
  fi
  requeue_target="${SLURM_JOB_ID}"
  if [[ -n "${SLURM_ARRAY_JOB_ID:-}" && -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    requeue_target="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
  fi
  echo "Graceful checkpoint complete; requeueing ${requeue_target}."
  scontrol requeue "${requeue_target}"
  exit 0
fi
"""
        if auto_requeue
        else ""
    )
    if graceful_pt_stop:
        driver_command = f"""python3 -u -m {module} --config "${{CONFIG}}" &
driver_pid=$!

driver_status=0
while true; do
  set +e
  wait "${{driver_pid}}"
  wait_status=$?
  set -e
  if kill -0 "${{driver_pid}}" 2>/dev/null; then
    continue
  fi
  driver_status=${{wait_status}}
  break
done
{requeue_block}exit "${{driver_status}}"
"""
    else:
        driver_command = f"""python3 -u -m {module} --config "${{CONFIG}}"
"""

    if workflow == "pt" and backend == "ray":
        gres = str(getattr(args, "gres", ""))
        gres_tail = gres.rsplit(":", 1)[-1]
        default_gpus_per_node = gres_tail if gres_tail.isdigit() else "1"
        workers_per_gpu = getattr(args, "workers_per_gpu", None)
        default_workers_per_gpu = int(workers_per_gpu) if workers_per_gpu is not None else 1
        text = f"""#!/bin/bash

#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node=1
#SBATCH --partition={args.partition}
#SBATCH --job-name={args.job_name}
#SBATCH --output=logs/%A_%a.log
#SBATCH --account={args.account}
#SBATCH --time={args.time}
{requeue_directives}{signal_directive}#SBATCH --gres={args.gres}
#SBATCH --cpus-per-task={args.cpus_per_task}
#SBATCH --mem-per-cpu={args.mem_per_cpu}
#SBATCH --array={array_spec}

set -euo pipefail

module --force purge
module load EasyBuild/2024a Python LAMMPS
ulimit -s unlimited

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PT_GPUS_PER_NODE="${{PT_GPUS_PER_NODE:-${{SLURM_GPUS_ON_NODE:-{default_gpus_per_node}}}}}"
export PT_WORKERS_PER_GPU="${{PT_WORKERS_PER_GPU:-{default_workers_per_gpu}}}"
export PYTHONPATH="{args.repo_root}${{PYTHONPATH:+:${{PYTHONPATH}}}}"

ROOT="{root}"
CONFIG_LIST="${{ROOT}}/configs.txt"
mkdir -p "${{ROOT}}/logs"

CONFIG=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${{CONFIG_LIST}}")
if [[ -z "${{CONFIG}}" || ! -f "${{CONFIG}}" ]]; then
  echo "ERROR: no config for SLURM_ARRAY_TASK_ID=${{SLURM_ARRAY_TASK_ID}}" >&2
  exit 1
fi

nodes=($(scontrol show hostnames "${{SLURM_JOB_NODELIST}}"))
head_node=${{nodes[0]}}
ray_port=${{RAY_PORT:-6379}}
head_ip=$(getent ahostsv4 "${{head_node}}" | awk 'NR==1{{print $1}}')
if [[ -z "${{head_ip}}" ]]; then
  echo "Failed to resolve head node IP for ${{head_node}}" >&2
  exit 1
fi
head_addr="${{head_ip}}:${{ray_port}}"
export RAY_ADDRESS="${{head_addr}}"

pids=()

cleanup() {{
  for n in "${{nodes[@]}}"; do
    srun -N1 -n1 -w "${{n}}" ray stop --force >/dev/null 2>&1 || true
  done
  for pid in "${{pids[@]:-}}"; do
    kill "${{pid}}" >/dev/null 2>&1 || true
  done
}}
trap cleanup EXIT
{signal_handler}

for n in "${{nodes[@]}}"; do
  srun -N1 -n1 -w "${{n}}" ray stop --force >/dev/null 2>&1 || true
done

echo "Starting Ray head at ${{head_addr}}"
srun --overlap -N1 -n1 -w "${{head_node}}" \\
  ray start --head \\
    --node-ip-address="${{head_ip}}" \\
    --port="${{ray_port}}" \\
    --num-gpus="${{PT_GPUS_PER_NODE}}" \\
    --num-cpus="${{SLURM_CPUS_PER_TASK}}" \\
    --disable-usage-stats \\
    --block &
pids+=("$!")

sleep 5

for n in "${{nodes[@]:1}}"; do
  echo "Starting Ray worker on ${{n}}"
  srun --overlap -N1 -n1 -w "${{n}}" \\
    ray start \\
      --address "${{head_addr}}" \\
      --num-gpus="${{PT_GPUS_PER_NODE}}" \\
      --num-cpus="${{SLURM_CPUS_PER_TASK}}" \\
      --disable-usage-stats \\
      --block &
  pids+=("$!")
done

for i in $(seq 1 30); do
  if ray status --address "${{head_addr}}" >/dev/null 2>&1; then
    echo "Ray cluster is ready at ${{head_addr}}"
    break
  fi
  sleep 2
  if [[ "${{i}}" -eq 30 ]]; then
    echo "ERROR: Ray GCS did not become reachable at ${{head_addr}}" >&2
    exit 1
  fi
done

cd "$(dirname "${{CONFIG}}")"
echo "Running {label} config: ${{CONFIG}}"
echo "Ray address: ${{RAY_ADDRESS}}"
{driver_command}
"""
    else:
        text = f"""#!/bin/bash

#SBATCH --nodes={nodes}
#SBATCH --ntasks=1
#SBATCH --partition={args.partition}
#SBATCH --job-name={args.job_name}
#SBATCH --output=logs/%A_%a.log
#SBATCH --account={args.account}
#SBATCH --time={args.time}
{requeue_directives}{signal_directive}#SBATCH --gres={args.gres}
#SBATCH --cpus-per-task={args.cpus_per_task}
#SBATCH --mem-per-cpu={args.mem_per_cpu}
#SBATCH --array={array_spec}

set -euo pipefail

module --force purge
module load EasyBuild/2024a Python LAMMPS
ulimit -s unlimited

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="{args.repo_root}${{PYTHONPATH:+:${{PYTHONPATH}}}}"

ROOT="{root}"
CONFIG_LIST="${{ROOT}}/configs.txt"
mkdir -p "${{ROOT}}/logs"

CONFIG=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${{CONFIG_LIST}}")
if [[ -z "${{CONFIG}}" || ! -f "${{CONFIG}}" ]]; then
  echo "ERROR: no config for SLURM_ARRAY_TASK_ID=${{SLURM_ARRAY_TASK_ID}}" >&2
  exit 1
fi
{signal_handler}

cd "$(dirname "${{CONFIG}}")"
echo "Running {label} config: ${{CONFIG}}"
{driver_command}
"""
    path = root / "run_array.slurm"
    path.write_text(text)
    path.chmod(0o775)


def _write_readme(root: Path, n_snapshots: int, n_configs: int, args: argparse.Namespace) -> None:
    workflow = str(args.workflow).lower()
    workflow_label = "adsorbate PT" if workflow == "pt" else "adsorbate CMC"
    selected_frames = getattr(args, "selected_frames", None)
    frame_source = (
        str(args.frames)
        if args.frames is not None
        else (
            f"{args.n_frames} frame(s) selected by {args.frame_selection_method} "
            f"from {args.frame_start}..{args.frame_stop}"
        )
        if args.n_frames is not None
        else "0"
    )
    selected_text = (
        f"- selected frames: `{selected_frames}`" if selected_frames is not None else ""
    )
    frame_lines = f"- frames: `{frame_source}`"
    if selected_text:
        frame_lines += f"\n{selected_text}"
    run_lines = ""
    if workflow == "pt":
        run_lines = (
            f"- PT range: `{args.pt_T_start:g}` to `{args.pt_T_end:g}` K"
            f" | grid: `{args.pt_grid_space}`"
            f" | cycles: `{args.pt_n_cycles}`"
            f" | backend: `{args.backend}`\n"
        )
    else:
        run_lines = (
            f"- temperature: `{args.temperature:g} K`\n"
            f"- sweeps: `{args.nsweeps}` with `{args.equilibration}` equilibration sweeps\n"
        )
    text = f"""# {workflow_label} inputs

Source snapshot/trajectory:

```text
{Path(args.snapshot).expanduser()}
```

This directory contains `{n_snapshots}` extracted slab snapshot(s) and
`{n_configs}` {workflow_label} config(s).

Layout:

- `snapshots/`: extracted clean slab snapshots.
- `runs/snapshot_*/seed_*/config.yaml`: one {workflow_label} input per task.
- `snapshot_manifest.csv`: source frame, source cycle, and optional source energy.
- `configs.txt`: task list consumed by the Slurm array.
- `run_array.slurm`: one config per Slurm array task.

Submit:

```bash
cd {root}
sbatch run_array.slurm
```

Main settings:

{frame_lines}
- adsorbate: `{args.adsorbate}`
- site elements: `{','.join(_parse_symbols(args.site_elements))}`
- site types: `{','.join(_parse_symbols(args.site_types))}`
{run_lines}
- relax: `{bool(args.relax)}`, `relax_steps={args.relax_steps}`, `fmax={args.fmax}`
"""
    (root / "README.md").write_text(text)


def build_parser(defaults: dict[str, object] | None = None) -> argparse.ArgumentParser:
    defaults = defaults or {}

    def default(name: str, fallback):
        return defaults.get(name, fallback)

    parser = argparse.ArgumentParser(
        description="Generate canonical adsorbate-CMC inputs from pristine slab snapshots.",
    )
    parser.add_argument("--config", type=Path, default=None, help="YAML preparation config.")
    parser.add_argument(
        "--workflow",
        default=default("workflow", "cmc"),
        choices=["cmc", "pt"],
        help="Generated workflow type: single-temperature CMC or adsorbate PT.",
    )
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=default("snapshot", None),
        help="ASE-readable slab snapshot or trajectory.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=default("out_dir", None),
        help="Output directory to create.",
    )
    parser.add_argument(
        "--frames",
        default=default("frames", None),
        help=(
            "Explicit frame indices to extract. Supports comma/space values and "
            "ranges like 700:1125:60. Overrides --n-frames."
        ),
    )
    parser.add_argument(
        "--n-frames",
        type=int,
        default=default("n_frames", None),
        help="Number of evenly spaced distinct frames to extract when --frames is omitted.",
    )
    parser.add_argument(
        "--frame-start",
        type=int,
        default=default("frame_start", 0),
        help="First frame allowed for automatic frame selection.",
    )
    parser.add_argument(
        "--frame-stop",
        type=int,
        default=default("frame_stop", None),
        help=(
            "Last frame allowed for automatic frame selection, inclusive. "
            "Negative values are relative to the trajectory end."
        ),
    )
    parser.add_argument(
        "--frame-selection-method",
        default=default("frame_selection_method", "evenly_spaced"),
        choices=["evenly_spaced", "uniform", "diverse", "soap_diverse"],
        help="How to select frames when --frames is omitted.",
    )
    parser.add_argument(
        "--frame-descriptor",
        default=default("frame_descriptor", "soap"),
        choices=["soap"],
        help="Descriptor used by diverse frame selection.",
    )
    parser.add_argument(
        "--frame-candidate-stride",
        type=int,
        default=default("frame_candidate_stride", 1),
        help="Evaluate every Nth frame as a candidate for diverse selection.",
    )
    parser.add_argument("--soap-species", nargs="+", default=default("soap_species", None))
    parser.add_argument("--soap-r-cut", type=float, default=default("soap_r_cut", 5.0))
    parser.add_argument("--soap-n-max", type=int, default=default("soap_n_max", 6))
    parser.add_argument("--soap-l-max", type=int, default=default("soap_l_max", 4))
    parser.add_argument("--soap-sigma", type=float, default=default("soap_sigma", 0.5))
    parser.add_argument("--soap-average", default=default("soap_average", "inner"))
    parser.add_argument("--soap-periodic", dest="soap_periodic", action="store_true", default=default("soap_periodic", True))
    parser.add_argument("--soap-nonperiodic", dest="soap_periodic", action="store_false")
    parser.add_argument(
        "--source-dat",
        type=Path,
        default=default("source_dat", None),
        help="Optional .dat file aligned with trajectory frames.",
    )
    parser.add_argument("--seeds", default=default("seeds", "67,211"), help="CMC seeds, e.g. '67,211'.")

    parser.add_argument("--adsorbate", default=default("adsorbate", "OH"), help="Adsorbate preset or ASE-readable file.")
    parser.add_argument("--adsorbate-anchor-index", type=int, default=default("adsorbate_anchor_index", 0))
    parser.add_argument("--n-adsorbates", type=int, default=default("n_adsorbates", 1))
    parser.add_argument("--site-elements", nargs="+", default=default("site_elements", ["Ti", "Zr"]))
    parser.add_argument("--site-types", nargs="+", default=default("site_types", ["atop", "fcc", "hcp"]))
    parser.add_argument("--substrate-elements", nargs="+", default=default("substrate_elements", ["Ti", "Zr", "C"]))
    parser.add_argument("--functional-elements", nargs="+", default=default("functional_elements", ["O"]))
    parser.add_argument("--surface-side", default=default("surface_side", "top"), choices=["top", "bottom"])
    parser.add_argument("--vertical-offset", type=float, default=default("vertical_offset", 1.5))

    parser.add_argument("--temperature", type=float, default=default("temperature", 300.0))
    parser.add_argument("--nsweeps", type=int, default=default("nsweeps", 800))
    parser.add_argument("--equilibration", type=int, default=default("equilibration", 200))
    parser.add_argument("--write-interval", type=int, default=default("write_interval", 10))
    parser.add_argument("--sample-interval", type=int, default=default("sample_interval", 5))
    parser.add_argument(
        "--write-debug-trajs",
        dest="write_debug_trajs",
        action="store_true",
        default=default("write_debug_trajs", False),
    )
    parser.add_argument(
        "--no-write-debug-trajs",
        dest="write_debug_trajs",
        action="store_false",
    )
    for option, description in (
        ("attempted", "materialized proposals before validation"),
        ("accepted", "accepted energy-evaluated proposals"),
        ("rejected", "rejected validation and Metropolis proposals"),
    ):
        parser.add_argument(
            f"--write-{option}-traj",
            dest=f"write_{option}_traj",
            action=argparse.BooleanOptionalAction,
            default=default(f"write_{option}_traj", False),
            help=f"Write {description} to a per-replica debug trajectory.",
        )
    parser.add_argument(
        "--debug-traj-interval",
        type=int,
        default=default("debug_traj_interval", 1),
    )
    parser.add_argument("--checkpoint-interval", type=int, default=default("checkpoint_interval", 100))
    parser.add_argument(
        "--cmc-resume",
        dest="cmc_resume",
        action="store_true",
        default=default("cmc_resume", False),
        help="Resume generated single-temperature CMC runs from their checkpoints.",
    )
    parser.add_argument(
        "--no-cmc-resume",
        dest="cmc_resume",
        action="store_false",
    )

    parser.add_argument("--displacement-sigma", type=float, default=default("displacement_sigma", 0.25))
    parser.add_argument("--max-displacement-trials", type=int, default=default("max_displacement_trials", 20))
    parser.add_argument("--min-clearance", type=float, default=default("min_clearance", 0.8))
    parser.add_argument("--site-match-tol", type=float, default=default("site_match_tol", 0.6))
    parser.add_argument("--support-xy-tol", type=float, default=default("support_xy_tol", 1.2))
    parser.add_argument("--termination-site-xy-tol", type=float, default=default("termination_site_xy_tol", 2.2))
    parser.add_argument("--surface-layer-tol", type=float, default=default("surface_layer_tol", 0.5))
    parser.add_argument("--termination-clearance", type=float, default=default("termination_clearance", 0.8))
    parser.add_argument("--vertical-adjust-step", type=float, default=default("vertical_adjust_step", 0.25))
    parser.add_argument("--max-vertical-adjust", type=float, default=default("max_vertical_adjust", 1.5))
    parser.add_argument("--z-max-support", type=float, default=default("z_max_support", 3.5))
    parser.add_argument("--detach-tol", type=float, default=default("detach_tol", 3.0))

    parser.add_argument("--relax", dest="relax", action="store_true", default=default("relax", True))
    parser.add_argument("--no-relax", dest="relax", action="store_false")
    parser.add_argument("--relax-steps", type=int, default=default("relax_steps", 20))
    parser.add_argument("--fmax", type=float, default=default("fmax", 0.2))

    parser.add_argument("--enable-hybrid-md", dest="enable_hybrid_md", action="store_true", default=default("enable_hybrid_md", False))
    parser.add_argument("--no-hybrid-md", dest="enable_hybrid_md", action="store_false")
    parser.add_argument("--md-move-prob", type=float, default=default("md_move_prob", 0.0))
    parser.add_argument("--md-steps", type=int, default=default("md_steps", 10))
    parser.add_argument("--md-timestep-fs", type=float, default=default("md_timestep_fs", 0.5))
    parser.add_argument("--md-ensemble", default=default("md_ensemble", "nve"))
    parser.add_argument("--md-accept-mode", default=default("md_accept_mode", "hamiltonian"))
    parser.add_argument("--md-friction", type=float, default=default("md_friction", 0.01))

    parser.add_argument("--hop-prob", type=float, default=default("hop_prob", 0.80))
    parser.add_argument("--hop-reorient-prob", type=float, default=default("hop_reorient_prob", 0.50))
    parser.add_argument("--reorientation-prob", type=float, default=default("reorientation_prob", 0.20))
    parser.add_argument("--reorientation-angle-deg", type=float, default=default("reorientation_angle_deg", 45.0))
    parser.add_argument("--max-reorientation-trials", type=int, default=default("max_reorientation_trials", 20))
    parser.add_argument("--puckering-prob", type=float, default=default("puckering_prob", 0.0))
    parser.add_argument("--puckering-elements", nargs="+", default=default("puckering_elements", ["Ti", "Zr"]))
    parser.add_argument("--puckering-height-A", type=float, default=default("puckering_height_A", 0.5))
    parser.add_argument("--puckering-height-jitter-A", type=float, default=default("puckering_height_jitter_A", 0.1))
    parser.add_argument(
        "--puckering-heights",
        type=_parse_yaml_mapping,
        default=default("puckering_heights", None),
    )
    parser.add_argument("--max-puckering-trials", type=int, default=default("max_puckering_trials", 20))
    parser.add_argument("--adsorbate-surface-clearance-A", type=float, default=default("adsorbate_surface_clearance_A", 0.2))
    parser.add_argument("--adsorbate-surface-xy-tol-A", type=float, default=default("adsorbate_surface_xy_tol_A", 2.5))
    parser.add_argument("--molecular-upright-atom-indices", nargs="+", type=int, default=default("molecular_upright_atom_indices", None))
    parser.add_argument("--molecular-upright-min-z-A", type=float, default=default("molecular_upright_min_z_A", None))
    parser.add_argument("--diagnostics", dest="diagnostics", action="store_true", default=default("diagnostics", False))
    parser.add_argument("--no-diagnostics", dest="diagnostics", action="store_false")
    parser.add_argument("--diagnostics-log", dest="diagnostics_log", action="store_true", default=default("diagnostics_log", False))
    parser.add_argument("--no-diagnostics-log", dest="diagnostics_log", action="store_false")

    parser.add_argument("--calculator", default=default("calculator", "symmetrix"), choices=["symmetrix", "mace", "lj"])
    parser.add_argument("--model-file", default=default("model_file", None))
    parser.add_argument("--model", default=default("model", None))
    parser.add_argument("--device", default=default("device", "cuda"))
    parser.add_argument("--use-kokkos", action="store_true", default=default("use_kokkos", True))
    parser.add_argument("--no-kokkos", dest="use_kokkos", action="store_false")
    parser.add_argument("--lj-cutoff", type=float, default=default("lj_cutoff", 6.0))

    parser.add_argument("--output-prefix", default=default("output_prefix", "adsorbate_cmc"))
    parser.add_argument("--run-output-dir", default=default("run_output_dir", "results"))

    parser.add_argument("--pt-t-start", dest="pt_T_start", type=float, default=default("pt_T_start", 800.0))
    parser.add_argument("--pt-t-end", dest="pt_T_end", type=float, default=default("pt_T_end", 300.0))
    parser.add_argument("--pt-t-step", dest="pt_T_step", type=float, default=default("pt_T_step", 50.0))
    parser.add_argument("--pt-n-replicas", dest="pt_n_replicas", type=int, default=default("pt_n_replicas", None))
    parser.add_argument("--pt-grid-space", dest="pt_grid_space", default=default("pt_grid_space", "temperature"))
    parser.add_argument("--pt-swap-interval", dest="pt_swap_interval", type=int, default=default("pt_swap_interval", 20))
    parser.add_argument("--pt-swap-stride", dest="pt_swap_stride", type=int, default=default("pt_swap_stride", 1))
    parser.add_argument("--pt-report-interval", dest="pt_report_interval", type=int, default=default("pt_report_interval", 5))
    parser.add_argument("--pt-sampling-interval", dest="pt_sampling_interval", type=int, default=default("pt_sampling_interval", 1))
    parser.add_argument("--pt-local-eq-fraction", dest="pt_local_eq_fraction", type=float, default=default("pt_local_eq_fraction", 0.2))
    parser.add_argument("--pt-checkpoint-interval", dest="pt_checkpoint_interval", type=int, default=default("pt_checkpoint_interval", 10))
    parser.add_argument("--pt-worker-checkpoint-interval", dest="pt_worker_checkpoint_interval", type=int, default=default("pt_worker_checkpoint_interval", 0))
    parser.add_argument("--pt-n-cycles", dest="pt_n_cycles", type=int, default=default("pt_n_cycles", 2))
    parser.add_argument("--pt-equilibration-cycles", dest="pt_equilibration_cycles", type=int, default=default("pt_equilibration_cycles", 0))
    parser.add_argument("--pt-resume", dest="pt_resume", action="store_true", default=default("pt_resume", False))
    parser.add_argument("--no-pt-resume", dest="pt_resume", action="store_false")

    parser.add_argument("--backend", default=default("backend", "multiprocessing"), choices=["multiprocessing", "ray"])
    parser.add_argument("--n-gpus", type=int, default=default("n_gpus", None))
    parser.add_argument("--workers-per-gpu", type=int, default=default("workers_per_gpu", None))
    parser.add_argument("--ray-address", default=default("ray_address", None))
    parser.add_argument("--ray-log-to-driver", dest="ray_log_to_driver", action="store_true", default=default("ray_log_to_driver", False))
    parser.add_argument("--ray-no-log-to-driver", dest="ray_log_to_driver", action="store_false")
    parser.add_argument("--ray-num-cpus-per-task", type=float, default=default("ray_num_cpus_per_task", 1))
    parser.add_argument("--ray-num-gpus-per-task", type=float, default=default("ray_num_gpus_per_task", None))
    parser.add_argument("--ray-actor-max-restarts", type=int, default=default("ray_actor_max_restarts", 0))
    parser.add_argument("--ray-actor-max-task-retries", type=int, default=default("ray_actor_max_task_retries", 0))
    parser.add_argument("--ray-get-timeout-s", type=float, default=default("ray_get_timeout_s", None))
    parser.add_argument("--use-placement-group", action="store_true", default=default("use_placement_group", False))
    parser.add_argument("--no-placement-group", dest="use_placement_group", action="store_false")
    parser.add_argument("--placement-group-strategy", default=default("placement_group_strategy", "SPREAD"))
    parser.add_argument("--remove-placement-group-on-stop", dest="remove_placement_group_on_stop", action="store_true", default=default("remove_placement_group_on_stop", True))
    parser.add_argument("--keep-placement-group-on-stop", dest="remove_placement_group_on_stop", action="store_false")
    parser.add_argument("--shutdown-on-stop", action="store_true", default=default("shutdown_on_stop", False))
    parser.add_argument("--no-shutdown-on-stop", dest="shutdown_on_stop", action="store_false")

    parser.add_argument("--repo-root", default=default("repo_root", "/gpfs/home/acad/ucl-modl/wchen/mxene_proj/gcmc"))
    parser.add_argument("--nodes", type=int, default=default("nodes", 1))
    parser.add_argument("--partition", default=default("partition", "gpu"))
    parser.add_argument("--account", default=default("account", "htbase"))
    parser.add_argument("--time", default=default("time", "12:00:00"))
    parser.add_argument("--gres", default=default("gres", "gpu:1"))
    parser.add_argument("--cpus-per-task", type=int, default=default("cpus_per_task", 8))
    parser.add_argument("--mem-per-cpu", default=default("mem_per_cpu", "8gb"))
    parser.add_argument("--array-concurrency", type=int, default=default("array_concurrency", 4))
    parser.add_argument("--job-name", default=default("job_name", "adsorbate_cmc"))
    parser.add_argument(
        "--signal-seconds-before-timeout",
        type=int,
        default=default("signal_seconds_before_timeout", 0),
        help=(
            "For PT workflows, ask Slurm to send SIGUSR1 this many seconds "
            "before walltime so the driver checkpoints and stops cleanly; 0 disables it."
        ),
    )
    parser.add_argument(
        "--auto-requeue",
        dest="auto_requeue",
        action="store_true",
        default=default("auto_requeue", False),
        help=(
            "After a pre-timeout PT checkpoint, requeue the current Slurm "
            "array task automatically."
        ),
    )
    parser.add_argument(
        "--no-auto-requeue",
        dest="auto_requeue",
        action="store_false",
    )
    parser.set_defaults(
        adsorbate_templates=default("adsorbate_templates", None),
        system_extra=default("system_extra", {}),
        cmc_extra=default("cmc_extra", {}),
        calculator_extra=default("calculator_extra", {}),
        output_extra=default("output_extra", {}),
        pt_extra=default("pt_extra", {}),
        backend_extra=default("backend_extra", {}),
        pt_fine_grid_temps=default("pt_fine_grid_temps", []),
        pt_fine_grid_weights=default("pt_fine_grid_weights", []),
        pt_fine_grid_strength=default("pt_fine_grid_strength", 4.0),
        pt_fine_grid_width=default("pt_fine_grid_width", None),
        stats_file=default("stats_file", "replica_stats.csv"),
        results_file=default("results_file", "results.csv"),
        pt_checkpoint_file=default("checkpoint_file", "pt_state.pkl"),
        initial_traj_file=default("initial_traj_file", "adsorbate_pt_initial.traj"),
    )
    return parser


def main() -> None:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path, default=None)
    pre_args, _ = pre_parser.parse_known_args()
    defaults = (
        _defaults_from_prepare_yaml(pre_args.config)
        if pre_args.config is not None
        else {}
    )

    parser = build_parser(defaults)
    args = parser.parse_args()
    if args.snapshot is None:
        parser.error("--snapshot is required unless provided by --config")
    if args.out_dir is None:
        parser.error("--out-dir is required unless provided by --config")
    args.workflow = str(args.workflow).lower()
    if args.calculator != "lj" and not (args.model_file or args.model):
        parser.error("--model-file or --model is required unless --calculator=lj")

    root = args.out_dir.expanduser().resolve()
    snapshot_dir = root / "snapshots"
    runs_dir = root / "runs"
    logs_dir = root / "logs"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    source_snapshot = args.snapshot.expanduser().resolve()
    if args.frames is not None:
        frames = _parse_int_set(args.frames)
    elif args.n_frames is not None:
        total_frames = _count_frames(source_snapshot)
        method = str(args.frame_selection_method).lower()
        if method in {"evenly_spaced", "uniform"}:
            frames = _select_distinct_frames(
                total_frames=total_frames,
                n_frames=args.n_frames,
                frame_start=args.frame_start,
                frame_stop=args.frame_stop,
            )
        elif method in {"diverse", "soap_diverse"}:
            frames = _select_diverse_frames(
                source_snapshot=source_snapshot,
                total_frames=total_frames,
                n_frames=args.n_frames,
                frame_start=args.frame_start,
                frame_stop=args.frame_stop,
                args=args,
            )
        else:
            raise ValueError(f"Unsupported frame selection method: {method!r}")
    else:
        frames = [0]
    args.selected_frames = frames
    seeds = _parse_int_set(args.seeds)
    dat_rows = _parse_dat(args.source_dat.expanduser().resolve() if args.source_dat else None)

    manifest: list[dict[str, object]] = []
    config_paths: list[Path] = []

    for snapshot_id, frame in enumerate(frames):
        atoms = read(source_snapshot, index=frame)
        snapshot_path = snapshot_dir / _snapshot_name(snapshot_id, frame)
        write(snapshot_path, atoms)
        symbols = atoms.get_chemical_symbols()
        counts = {symbol: symbols.count(symbol) for symbol in sorted(set(symbols))}
        source_cycle: float | str = ""
        source_energy: float | str = ""
        if frame < len(dat_rows):
            source_cycle, source_energy = dat_rows[frame]
        row = {
            "snapshot_id": snapshot_id,
            "source_frame": frame,
            "source_cycle": source_cycle,
            "source_energy_eV": source_energy,
            "snapshot": str(snapshot_path),
            "formula": atoms.get_chemical_formula(),
            "n_atoms": len(atoms),
        }
        for symbol in sorted(counts):
            row[f"n_{symbol}"] = counts[symbol]
        manifest.append(row)

        for seed in seeds:
            run_dir = runs_dir / f"snapshot_{snapshot_id:03d}" / f"seed_{seed:03d}"
            run_dir.mkdir(parents=True, exist_ok=True)
            config_builder = (
                _pt_config_for_task if args.workflow == "pt" else _config_for_task
            )
            config = config_builder(
                snapshot_path=snapshot_path,
                run_dir=run_dir,
                source_snapshot=source_snapshot,
                source_frame=frame,
                source_cycle=source_cycle,
                seed=seed,
                args=args,
            )
            config_path = run_dir / "config.yaml"
            config_path.write_text(yaml.safe_dump(config, sort_keys=False))
            config_paths.append(config_path)

    if manifest:
        fieldnames: list[str] = []
        seen: set[str] = set()
        for row in manifest:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
        with (root / "snapshot_manifest.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(manifest)

    with (root / "configs.txt").open("w") as handle:
        for path in config_paths:
            handle.write(str(path) + "\n")

    _write_slurm(root, len(config_paths), args)
    _write_readme(root, len(manifest), len(config_paths), args)

    label = "adsorbate PT" if args.workflow == "pt" else "adsorbate CMC"
    print(f"Wrote {label} inputs to: {root}")
    print(f"  snapshots: {len(manifest)}")
    print(f"  configs: {len(config_paths)}")
    print(f"  submit: cd {root} && sbatch run_array.slurm")


if __name__ == "__main__":
    main()
