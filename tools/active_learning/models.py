"""Shared committee-model discovery and calculator construction."""

from __future__ import annotations

import argparse
import gc
import re
from pathlib import Path


MODEL_SUFFIXES = {".model": "mace", ".json": "symmetrix"}


def discover_model_paths(values: list[str], calculator: str) -> list[Path]:
    """Resolve model files from explicit files or non-recursive directories."""
    discovered: list[Path] = []
    for value in values:
        path = Path(value).expanduser().resolve()
        if path.is_file():
            if path.suffix.lower() not in MODEL_SUFFIXES:
                raise ValueError(f"Unsupported model file extension: {path}")
            discovered.append(path)
        elif path.is_dir():
            discovered.extend(sorted(path.glob("*.model")))
            discovered.extend(sorted(path.glob("*.json")))
        else:
            raise FileNotFoundError(path)

    discovered = list(dict.fromkeys(path.resolve() for path in discovered))
    if calculator == "mace":
        discovered = [
            path for path in discovered if path.suffix.lower() == ".model"
        ]
    elif calculator == "symmetrix":
        discovered = [
            path for path in discovered if path.suffix.lower() == ".json"
        ]
    else:
        suffixes = {path.suffix.lower() for path in discovered}
        if ".model" in suffixes and ".json" in suffixes:
            raise ValueError(
                "--model-path found both native MACE (*.model) and Symmetrix "
                "(*.json) files. Select one with --calculator mace or "
                "--calculator symmetrix."
            )

    if not discovered:
        raise FileNotFoundError(
            "No compatible top-level *.model or *.json files were found in "
            "the supplied --model-path locations."
        )
    return discovered


def resolve_models(
    search_paths: list[str] | None,
    calculator: str,
    *,
    minimum: int,
) -> list[Path]:
    """Resolve and validate an explicitly supplied model committee."""
    if not search_paths:
        raise ValueError(
            "--model-path is required; provide a model file or directory."
        )
    models = discover_model_paths(search_paths, calculator)
    if len(models) < minimum:
        raise ValueError(
            f"At least {minimum} committee model(s) are required; "
            f"--model-path resolved {len(models)}."
        )

    print(f"Committee ({len(models)} models):")
    for model in models:
        backend = resolve_calculator_backend(model, calculator)
        print(f"  [{backend}] {model}")
    return models


def model_labels(models: list[Path]) -> list[str]:
    """Return stable, unique labels for per-model output columns."""
    labels: list[str] = []
    used: set[str] = set()
    for index, model in enumerate(models):
        match = re.search(r"multi-(\d+)", model.stem)
        label = f"model_{match.group(1)}" if match else f"model_{index:02d}"
        if label in used:
            label = f"model_{index:02d}"
        labels.append(label)
        used.add(label)
    return labels


def resolve_calculator_backend(model: Path, calculator: str) -> str:
    """Resolve the ASE calculator backend from the option and file suffix."""
    calculator = str(calculator).lower()
    if calculator == "auto":
        suffix = model.suffix.lower()
        if suffix not in MODEL_SUFFIXES:
            raise ValueError(f"Unsupported model file extension: {model}")
        return MODEL_SUFFIXES[suffix]
    if calculator not in {"mace", "symmetrix"}:
        raise ValueError(f"Unsupported calculator backend: {calculator!r}")
    expected_suffix = ".json" if calculator == "symmetrix" else ".model"
    if model.suffix.lower() != expected_suffix:
        raise ValueError(
            f"{calculator} requires {expected_suffix} models, received {model}"
        )
    return calculator


def build_calculator(
    model: Path,
    *,
    backend: str,
    device: str,
    default_dtype: str | None,
    use_kokkos: bool,
):
    """Build one ASE calculator for a committee member."""
    if backend == "symmetrix":
        from symmetrix import Symmetrix

        return Symmetrix(
            model_file=str(model),
            dtype=str(default_dtype or "float64"),
            use_kokkos=bool(use_kokkos),
        )

    from mace.calculators import MACECalculator

    kwargs: dict[str, object] = {
        "model_paths": [str(model)],
        "device": str(device),
    }
    if default_dtype:
        kwargs["default_dtype"] = str(default_dtype)
    return MACECalculator(**kwargs)


def release_calculator(calc, device: str) -> None:
    """Release calculator resources between committee members."""
    del calc
    gc.collect()
    if str(device).startswith("cuda"):
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass


def add_model_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_device: str,
) -> None:
    """Add the common model-discovery and calculator CLI options."""
    parser.add_argument(
        "--model-path",
        "--model_path",
        dest="model_path",
        action="append",
        required=True,
        help=(
            "Model file or directory. Repeat as needed. Directories are "
            "searched non-recursively for top-level *.model or *.json files."
        ),
    )
    parser.add_argument(
        "--calculator",
        default="auto",
        choices=("auto", "mace", "symmetrix"),
        help=(
            "Calculator backend. Auto infers native MACE from .model and "
            "Symmetrix from .json."
        ),
    )
    parser.add_argument("--device", default=default_device)
    parser.add_argument(
        "--default-dtype", choices=("float32", "float64"), default=None
    )
    parser.add_argument(
        "--use-kokkos",
        dest="use_kokkos",
        action="store_true",
        default=True,
        help="Use the Kokkos Symmetrix evaluator (default).",
    )
    parser.add_argument(
        "--no-kokkos",
        dest="use_kokkos",
        action="store_false",
        help="Use the non-Kokkos Symmetrix evaluator.",
    )
