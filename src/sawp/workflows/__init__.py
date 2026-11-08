"""High-level scientific workflows built from the reusable SAWP core."""

from .cooling import build_event_table, cooling_composite
from .correlation import correlation_matrix, pairwise_statistics
from .fields import build_moisture_fields, build_theta_e
from .fluxes import FluxCompositeSpec, FluxKind, compute_flux_composites

__all__ = [
    "FluxCompositeSpec",
    "FluxKind",
    "build_event_table",
    "build_moisture_fields",
    "build_theta_e",
    "compute_flux_composites",
    "cooling_composite",
    "correlation_matrix",
    "pairwise_statistics",
]
