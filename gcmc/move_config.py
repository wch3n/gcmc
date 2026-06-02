"""Shared adsorbate CMC move configuration normalization."""

from __future__ import annotations


def _as_probability(value: object, *, name: str) -> float:
    probability = float(value)
    if not (0.0 <= probability <= 1.0):
        raise ValueError(f"{name} must be in [0, 1].")
    return probability


def normalize_adsorbate_move_config(flat_config: dict) -> dict:
    """Apply nested ``moves`` syntax to flat AdsorbateCMC keyword config."""

    moves = flat_config.get("moves")
    if not isinstance(moves, dict):
        return flat_config

    _apply_move_mode(flat_config, moves)
    _apply_displacement_config(flat_config, moves.get("displacement"))
    _apply_reorientation_config(flat_config, moves.get("reorientation"))
    _apply_hop_config(flat_config, moves.get("hop"))
    _apply_standalone_puckering_config(flat_config, moves.get("puckering"))
    _apply_surface_filter_config(flat_config, moves)
    return flat_config


def _apply_move_mode(flat_config: dict, moves: dict) -> None:
    if "mode" in moves:
        flat_config["move_mode"] = moves["mode"]
    if "move_mode" in moves:
        flat_config["move_mode"] = moves["move_mode"]


def _apply_displacement_config(flat_config: dict, displacement: object) -> None:
    if not isinstance(displacement, dict):
        return
    if "sigma_A" in displacement:
        flat_config["displacement_sigma"] = displacement["sigma_A"]
    if "sigma" in displacement:
        flat_config["displacement_sigma"] = displacement["sigma"]
    if "max_trials" in displacement:
        flat_config["max_displacement_trials"] = displacement["max_trials"]


def _apply_reorientation_config(flat_config: dict, reorientation: object) -> None:
    if not isinstance(reorientation, dict):
        return
    if "prob" in reorientation:
        flat_config["reorientation_prob"] = reorientation["prob"]
    if "angle_deg" in reorientation:
        flat_config["rotation_max_angle_deg"] = reorientation["angle_deg"]
    if "max_trials" in reorientation:
        flat_config["max_reorientation_trials"] = reorientation["max_trials"]


def _apply_hop_config(flat_config: dict, hop: object) -> None:
    if not isinstance(hop, dict):
        return

    if "max_trials" in hop:
        flat_config["max_displacement_trials"] = hop["max_trials"]

    reorient_prob = _apply_hop_reorientation_config(flat_config, hop.get("reorient"))
    puckering_prob = _apply_hop_puckering_config(flat_config, hop.get("puckering"))
    if "prob" not in hop:
        return

    hop_prob = _as_probability(hop["prob"], name="moves.hop.prob")
    plain_hop_prob = hop_prob * (1.0 - puckering_prob)
    puckered_hop_prob = hop_prob * puckering_prob
    flat_config["site_hop_prob"] = plain_hop_prob * (1.0 - reorient_prob)
    flat_config["hop_reorientation_prob"] = plain_hop_prob * reorient_prob
    flat_config["hop_puckering_prob"] = puckered_hop_prob * (1.0 - reorient_prob)
    flat_config["hop_puckering_reorientation_prob"] = (
        puckered_hop_prob * reorient_prob
    )


def _apply_hop_reorientation_config(flat_config: dict, reorient: object) -> float:
    if not isinstance(reorient, dict):
        return 0.0
    reorient_prob = 0.0
    if bool(reorient.get("enabled", True)):
        reorient_prob = _as_probability(
            reorient.get("prob", 1.0),
            name="moves.hop.reorient.prob",
        )
    if "angle_deg" in reorient:
        flat_config["hop_reorientation_angle_deg"] = reorient["angle_deg"]
    if "max_trials" in reorient:
        flat_config["max_hop_reorientation_trials"] = reorient["max_trials"]
    return reorient_prob


def _apply_hop_puckering_config(flat_config: dict, puckering: object) -> float:
    if not isinstance(puckering, dict):
        return 0.0
    puckering_prob = 0.0
    if bool(puckering.get("enabled", True)):
        puckering_prob = _as_probability(
            puckering.get("prob", 1.0),
            name="moves.hop.puckering.prob",
        )
    _apply_puckering_geometry_config(flat_config, puckering)
    return puckering_prob


def _apply_standalone_puckering_config(flat_config: dict, puckering: object) -> None:
    if not isinstance(puckering, dict):
        return
    if "prob" in puckering:
        flat_config["puckering_prob"] = puckering["prob"]
    if "hop_prob" in puckering:
        flat_config["puckering_hop_prob"] = puckering["hop_prob"]
    _apply_puckering_geometry_config(flat_config, puckering)


def _apply_puckering_geometry_config(flat_config: dict, puckering: dict) -> None:
    if "elements" in puckering:
        flat_config["puckering_elements"] = puckering["elements"]
    if "height_A" in puckering:
        flat_config["puckering_height_A"] = puckering["height_A"]
    if "height_jitter_A" in puckering:
        flat_config["puckering_height_jitter_A"] = puckering["height_jitter_A"]
    if "max_trials" in puckering:
        flat_config["max_puckering_trials"] = puckering["max_trials"]


def _apply_surface_filter_config(flat_config: dict, moves: dict) -> None:
    aliases = {
        "surface_clearance_A": "adsorbate_surface_clearance_A",
        "adsorbate_surface_clearance_A": "adsorbate_surface_clearance_A",
        "surface_xy_tol_A": "adsorbate_surface_xy_tol_A",
        "adsorbate_surface_xy_tol_A": "adsorbate_surface_xy_tol_A",
    }
    for source_key, target_key in aliases.items():
        if source_key in moves:
            flat_config[target_key] = moves[source_key]
