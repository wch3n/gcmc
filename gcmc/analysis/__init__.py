"""Analysis utilities for ordering and structure metrics."""

from .local_motifs import LocalAdsorptionMotifAnalyzer
from .ordering import MXeneOrderingAnalyzer
from .motifs import MXeneSurfaceMotifAnalyzer
from .sites import MXeneAdsorptionSiteAnalyzer
from .site_viz import build_site_marker_atoms, overlay_site_markers, summarize_site_registry
from .sro import MXeneSROAnalyzer


def plot_adsorption_motif_distributions(*args, **kwargs):
    """Lazily import and run the adsorption-motif plotting function."""
    from .adsorption_motif_plots import plot_adsorption_motif_distributions as plot

    return plot(*args, **kwargs)

__all__ = [
    "LocalAdsorptionMotifAnalyzer",
    "plot_adsorption_motif_distributions",
    "MXeneOrderingAnalyzer",
    "MXeneSurfaceMotifAnalyzer",
    "MXeneAdsorptionSiteAnalyzer",
    "build_site_marker_atoms",
    "overlay_site_markers",
    "summarize_site_registry",
    "MXeneSROAnalyzer",
]
