"""Utilities for resolving parameter specifications in configs.

Supports two types of parameter variation:
- sweep: varies BETWEEN trajectories (e.g., density varies 0.5→1.0 across trajectories)
- dist: varies WITHIN a trajectory (e.g., each cell's radius drawn from uniform distribution)
"""

import numpy as np
from typing import Any
from omegaconf import DictConfig
import torch


def _resolve_steps_repeats(
    param_spec: dict,
    n_total: int,
) -> tuple[int, int]:
    """Resolve steps and repeats from a parameter spec.

    If only 'steps' is provided, auto-calculates repeats = n_total / steps.
    If only 'repeats' is provided, auto-calculates steps = n_total / repeats.
    If neither is provided, defaults to steps = n_total, repeats = 1.
    """
    steps = param_spec.get("steps")
    repeats = param_spec.get("repeats")

    if steps is not None and repeats is None:
        if n_total % steps != 0:
            raise ValueError(
                f"n_total={n_total} must be divisible by steps={steps}"
            )
        repeats = n_total // steps
    elif repeats is not None and steps is None:
        if n_total % repeats != 0:
            raise ValueError(
                f"n_total={n_total} must be divisible by repeats={repeats}"
            )
        steps = n_total // repeats
    elif steps is None and repeats is None:
        steps = n_total
        repeats = 1

    if steps * repeats != n_total:
        raise ValueError(
            f"steps={steps} * repeats={repeats} = {steps * repeats} must equal n_total={n_total}"
        )

    return steps, repeats


def _tile_or_repeat(
    base_values: np.ndarray,
    repeats: int,
    interleaved: bool,
    block: int | None = None,
) -> np.ndarray:
    """Tile or repeat base values.

    interleaved: true -> [v1,v2,v3,v1,v2,v3,...] (tile)
    interleaved: false -> [v1,v1,v1,v2,v2,v2,...] (repeat)

    block (overrides interleaved when present): tile-of-repeat, i.e.
    np.tile(np.repeat(base, block), repeats // block). This expresses an
    intermediate stride so several factors can compose into a balanced
    factorial grid (each factor gets a different `block`). Requires `block`
    to divide `repeats`. Special cases recover the two modes above:
      block=1 -> [v1,v2,v3,v1,v2,v3,...] (same as interleaved: true)
      block=repeats -> [v1,...,v1,v2,...,v2,...] (same as interleaved: false)
    """
    if block is not None:
        if repeats % block != 0:
            raise ValueError(f"block={block} must divide repeats={repeats}")
        outer = repeats // block
        return np.tile(np.repeat(base_values, block), outer)
    if interleaved:
        return np.tile(base_values, repeats)
    else:
        return np.repeat(base_values, repeats)


def _resolve_linspace(
    param_spec: dict,
    n_total: int,
) -> np.ndarray:
    """Resolve a linspace specification into an array of values.

    Args:
        param_spec: Dict with 'min', 'max', and optionally 'steps', 'repeats', 'interleaved'
        n_total: Total number of values needed

    Returns:
        Array of shape (n_total,) with parameter values

    interleaved: true (default) -> [v1,v2,v3,v1,v2,v3,...] (tile)
    interleaved: false -> [v1,v1,v1,v2,v2,v2,...] (repeat)
    """
    steps, repeats = _resolve_steps_repeats(param_spec, n_total)
    base_values = np.linspace(param_spec["min"], param_spec["max"], steps)
    return _tile_or_repeat(base_values, repeats, param_spec.get("interleaved", True),
                           block=param_spec.get("block"))


def _resolve_logspace(
    param_spec: dict,
    n_total: int,
) -> np.ndarray:
    """Resolve a logspace specification into an array of values.

    Spaces values exponentially (by orders of magnitude), e.g.:
        min: 1e3, max: 1e7, steps: 5
        -> [1e3, 1e4, 1e5, 1e6, 1e7]

    Args:
        param_spec: Dict with 'min', 'max', and optionally 'steps', 'repeats', 'interleaved'
            min and max must be positive.
        n_total: Total number of values needed

    Returns:
        Array of shape (n_total,) with parameter values
    """
    steps, repeats = _resolve_steps_repeats(param_spec, n_total)
    vmin, vmax = param_spec["min"], param_spec["max"]
    if vmin <= 0 or vmax <= 0:
        raise ValueError("logspace requires positive min and max")
    base_values = np.logspace(np.log10(vmin), np.log10(vmax), steps)
    return _tile_or_repeat(base_values, repeats, param_spec.get("interleaved", True),
                           block=param_spec.get("block"))


def _resolve_inverse_linspace(
    param_spec: dict,
    n_total: int,
) -> np.ndarray:
    """Resolve an inverse_linspace specification into an array of values.

    Linearly spaces values in reciprocal (1/x) space, then converts back.
    Useful for sampling periods that are evenly spaced in frequency, e.g.:
        min: 0.1, max: 1.0, steps: 10
        -> frequencies [1, 2, ..., 10] -> periods [1.0, 0.5, ..., 0.1]

    Args:
        param_spec: Dict with 'min', 'max', and optionally 'steps', 'repeats', 'interleaved'
            min and max must be nonzero.
        n_total: Total number of values needed

    Returns:
        Array of shape (n_total,) with parameter values
    """
    steps, repeats = _resolve_steps_repeats(param_spec, n_total)
    vmin, vmax = param_spec["min"], param_spec["max"]
    if vmin == 0 or vmax == 0:
        raise ValueError("inverse_linspace requires nonzero min and max")
    # Linspace in reciprocal space, then invert back
    reciprocal_values = np.linspace(1.0 / vmax, 1.0 / vmin, steps)
    base_values = 1.0 / reciprocal_values
    return _tile_or_repeat(base_values, repeats, param_spec.get("interleaved", True),
                           block=param_spec.get("block"))


def resolve_sweep_param(
    param_spec: Any,
    trajectory_idx: int,
    n_trajectories: int
) -> Any:
    """Resolve a sweep parameter for a specific trajectory.

    Args:
        param_spec: Either a fixed value or a dict with 'sweep' key
        trajectory_idx: Index of the current trajectory (0-indexed)
        n_trajectories: Total number of trajectories

    Returns:
        The resolved parameter value for this trajectory

    Examples:
        # Fixed value - returns as-is
        resolve_sweep_param(1.0, 0, 10) -> 1.0

        # Linspace sweep
        resolve_sweep_param({'sweep': 'linspace', 'min': 0.5, 'max': 1.0}, 0, 10) -> 0.5
        resolve_sweep_param({'sweep': 'linspace', 'min': 0.5, 'max': 1.0}, 9, 10) -> 1.0

        # Choice sweep (round-robin)
        resolve_sweep_param({'sweep': 'choice', 'values': [0, 10]}, 0, 10) -> 0
        resolve_sweep_param({'sweep': 'choice', 'values': [0, 10]}, 1, 10) -> 10
    """
    # Handle OmegaConf DictConfig
    if isinstance(param_spec, DictConfig):
        param_spec = dict(param_spec)

    if not isinstance(param_spec, dict) or "sweep" not in param_spec:
        return param_spec  # Fixed value

    sweep_type = param_spec["sweep"]

    if sweep_type == "linspace":
        values = _resolve_linspace(param_spec, n_trajectories)
        return float(values[trajectory_idx])

    elif sweep_type == "inverse_linspace":
        values = _resolve_inverse_linspace(param_spec, n_trajectories)
        return float(values[trajectory_idx])

    elif sweep_type == "logspace":
        values = _resolve_logspace(param_spec, n_trajectories)
        return float(values[trajectory_idx])

    elif sweep_type == "choice":
        # Round-robin: cycles through values [0, 10, 0, 10, ...]
        values = param_spec["values"]
        return values[trajectory_idx % len(values)]

    else:
        raise ValueError(f"Unknown sweep type: {sweep_type}. Supported: linspace, choice")


def resolve_dist_param(
    param_spec: Any,
    n_objects: int,
) -> np.ndarray:
    """Resolve a distribution parameter for all objects in a trajectory.

    Args:
        param_spec: Either a fixed value or a dict with 'dist' key
        n_objects: Number of objects to generate values for

    Returns:
        Array of shape (n_objects,) with parameter values

    Examples:
        # Fixed value - broadcasts to array
        resolve_dist_param(1.0, 100) -> array([1.0, 1.0, ..., 1.0])

        # Linspace (deterministic, evenly spaced)
        resolve_dist_param({'dist': 'linspace', 'min': 0.5, 'max': 1.5}, 100)
            -> array of 100 evenly spaced values from 0.5 to 1.5

        # Linspace with steps/repeats (e.g., 10 unique values, each repeated 10 times)
        resolve_dist_param({'dist': 'linspace', 'min': 0.5, 'max': 1.5, 'steps': 10, 'repeats': 10}, 100)
            -> array of 100 values: [0.5, 0.5, ..., 0.61, 0.61, ..., 1.5, 1.5]
    """
    # Handle OmegaConf DictConfig
    if isinstance(param_spec, DictConfig):
        param_spec = dict(param_spec)

    if not isinstance(param_spec, dict) or "dist" not in param_spec:
        # Fixed value - broadcast to array
        return np.full(n_objects, param_spec, dtype=np.float64)

    dist_type = param_spec["dist"]

    if dist_type == "linspace":
        return _resolve_linspace(param_spec, n_objects)

    elif dist_type == "inverse_linspace":
        return _resolve_inverse_linspace(param_spec, n_objects)

    elif dist_type == "logspace":
        return _resolve_logspace(param_spec, n_objects)

    else:
        raise ValueError(f"Unknown dist type: {dist_type}. Supported: linspace, inverse_linspace, logspace")


def is_sweep_param(param_spec: Any) -> bool:
    """Check if a parameter spec is a sweep parameter."""
    if isinstance(param_spec, DictConfig):
        param_spec = dict(param_spec)
    return isinstance(param_spec, dict) and "sweep" in param_spec


def is_dist_param(param_spec: Any) -> bool:
    """Check if a parameter spec is a distribution parameter."""
    if isinstance(param_spec, DictConfig):
        param_spec = dict(param_spec)
    return isinstance(param_spec, dict) and "dist" in param_spec

def resolve_growth_rates(cfg: DictConfig, rs: torch.Tensor | None = None) -> torch.Tensor:
    """Resolve growth rates for all particles based on config and simulation type.

    Args:
        cfg: Full config with data.params and data sections
        rs: Optional tensor of radii. If None, will be computed from config.

    Returns:
        Tensor of growth rates (n_objects,)
        - sphere_interaction: growth_rate * radius (per-particle)
        - polyhoop: alpha_mu (uniform area growth rate)
        - other: zeros
    """
    si = cfg.data.params
    data = cfg.data
    n_objects = data.n_objects

    if rs is None:
        radius_spec = si.get("radius")
        rs = torch.from_numpy(resolve_dist_param(radius_spec, n_objects)).float()

    if si.name == "sphere_interaction":
        growth_rate_spec = si.get("growth_rate", 0.0)
        growth_rate_pct = resolve_dist_param(growth_rate_spec, n_objects)
        growth_rates = torch.from_numpy(growth_rate_pct).float() * rs

    elif si.name == "polyhoop":
        # Check for per-cell growth rate dist syntax first
        growth_rate_spec = si.get("growth_rate")
        if growth_rate_spec is not None and is_dist_param(growth_rate_spec):
            growth_rates_np = resolve_dist_param(growth_rate_spec, n_objects)
            growth_rates = torch.from_numpy(growth_rates_np).float()
        else:
            # Legacy behavior: uniform alpha_mu
            alpha_mu = si.get("alpha_mu", 0.0)
            growth_rates = torch.full((n_objects,), alpha_mu, dtype=torch.float32)

    else:
        growth_rates = torch.zeros(n_objects, dtype=torch.float32)

    return growth_rates


def resolve_max_radius(cfg: DictConfig) -> tuple[float, torch.Tensor]:
    """Resolve maximum radius for normalization based on config.

    Handles both fixed values and distribution specs for radius and growth_rate.

    Args:
        cfg: Full config with data.params and data sections

    Returns:
        Tuple of (max_radius, initial_radii_tensor)
    """
    si = cfg.data.params
    data = cfg.data
    n_objects = data.n_objects

    # Resolve radius - can be fixed value or dist spec
    radius_spec = si.get("radius")
    rs_full = resolve_dist_param(radius_spec, n_objects)

    # init_steps extends the simulation before data capture; cells keep
    # growing during warm-up so it has to be counted toward max radius.
    init_steps = data.generator.get("init_steps", 0)
    total_steps = data.generator.trajectory_length + init_steps

    if si.name == "sphere_interaction":
        # Resolve growth rate - can be fixed value or dist spec
        growth_rate_spec = si.get("growth_rate", 0.0)
        rates_full = resolve_dist_param(growth_rate_spec, n_objects)

        r_max = rs_full.max() + rates_full.max() * data.dt * total_steps

    elif si.name == "polyhoop":
        import math

        max_initial_radius = rs_full.max()
        total_time = data.dt * total_steps

        # Check for per-cell growth rate dist syntax first
        growth_rate_spec = si.get("growth_rate")
        if growth_rate_spec is not None and is_dist_param(growth_rate_spec):
            growth_rates_np = resolve_dist_param(growth_rate_spec, n_objects)
            alpha_max = growth_rates_np.max()
        else:
            # Legacy behavior: use alpha_mu/alpha_CV
            alpha_mu = si.get("alpha_mu", 0.0)
            alpha_CV = si.get("alpha_CV", 0.0)

            if alpha_CV > 0:
                sigma = math.sqrt(math.log(1 + alpha_CV**2))
                alpha_max = alpha_mu * math.exp(2 * sigma)
            else:
                alpha_max = alpha_mu

        initial_area = math.pi * max_initial_radius**2
        max_area = initial_area + alpha_max * total_time
        r_max = (max_area / math.pi)**0.5 * 1.5

    else:
        # Default: no growth, max is just the largest initial radius
        r_max = rs_full.max()

    # Convert to torch tensor for compatibility
    rs_full = torch.from_numpy(rs_full).float()

    return r_max, rs_full