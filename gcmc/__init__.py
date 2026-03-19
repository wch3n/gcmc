"""
gcmc package: Monte Carlo simulation tools for surface/adsorbate and solid systems.

Exports:
- BaseMC: Generic MC infrastructure.
- SurfaceMCBase: Slab/adsorbate-specific MC infrastructure.
- GCMC: Legacy Grand Canonical Monte Carlo.
- AdsorbateGCMC: Site-based grand-canonical MC for adsorbates.
- AdsorbateCMC: Replica-compatible canonical MC for fixed-loading adsorbates.
- AlloyCMC: Canonical Monte Carlo for solids.
- SemiGrandAlloyMC: Semi-Grand Canonical Monte Carlo for solids.
- ReplicaExchange: Replica exchange module (ensemble- and system-neutral).
- utils: Adsorbate configuration generation, site registry, alloy lattice initialization, etc.
"""

from .base import BaseMC, SurfaceMCBase
from .gcmc import GCMC
from .adsorbate_cmc import AdsorbateCMC
from .adsorbate_gcmc import AdsorbateGCMC
from .alloy_cmc import AlloyCMC
from .sgcmc import SemiGrandAlloyMC
from .replica import ReplicaExchange
from .cluster_analysis import analyze_and_plot, find_cu_clusters
from .analysis import (
    MXeneAdsorptionSiteAnalyzer,
    MXeneOrderingAnalyzer,
    MXeneSROAnalyzer,
    MXeneSurfaceMotifAnalyzer,
)
from . import utils

__all__ = [
    "BaseMC",
    "SurfaceMCBase",
    "GCMC",
    "AdsorbateGCMC",
    "AdsorbateCMC",
    "AlloyCMC",
    "SemiGrandAlloyMC",
    "ReplicaExchange",
    "MXeneAdsorptionSiteAnalyzer",
    "MXeneOrderingAnalyzer",
    "MXeneSROAnalyzer",
    "MXeneSurfaceMotifAnalyzer",
    "analyze_and_plot",
    "find_cu_clusters",
    "utils",
]
