# %%
import sys
import os
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from concurrent.futures import ThreadPoolExecutor
from omegaconf import DictConfig, OmegaConf
from typing import List, Tuple, Optional
from skimage.draw import polygon2mask
from scipy.ndimage import shift
from graph_dyno.data.param_utils import resolve_dist_param, is_dist_param
from graph_dyno.data.raster import rasterize_batch

# Add the repo-root _polyhoop/ to path for the C++ extension
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "_polyhoop"))
import polyhoop


def _rasterize_polygon_from_snapshot(verts_np, image_size, r_max):
    """Rasterize a single polygon from pre-extracted numpy vertices.

    This is a standalone function safe to call from worker threads.

    Args:
        verts_np: (n_vertices, 2) numpy array of vertices centered at origin
        image_size: Output image size
        r_max: Maximum radius for scaling

    Returns:
        mask_np: (image_size, image_size) numpy binary mask
    """
    scale = (image_size - 1) / (2 * r_max)
    center = (image_size - 1) / 2
    scaled = np.stack((verts_np[:, 0] * scale + center,
                       verts_np[:, 1] * scale + center), axis=-1)
    return polygon2mask((image_size, image_size), scaled)


def _snapshot_to_polys(snapshot, image_size, r_max):
    """Center, scale and stack a snapshot into the list of polygons to rasterize.

    Returns a list of (n_vertices, 2) float64 arrays in (row, col) image coords.
    """
    scale = (image_size - 1) / (2 * r_max)
    center = (image_size - 1) / 2
    polys = []
    for cx, cy, verts_np in snapshot:
        polys.append(np.stack(((verts_np[:, 0] - cx) * scale + center,
                               (verts_np[:, 1] - cy) * scale + center), axis=-1))
    return polys


def _generate_image_from_snapshot(snapshot, image_size, r_max, device,
                                  executor=None, n_threads=1, out=None):
    """Generate channeled images from a pre-extracted polygon snapshot.

    This function does not access the ensemble and is safe to call from
    worker threads. Unlike skimage's ``polygon2mask`` (which holds the GIL),
    ``rasterize_batch`` batches the crossing test across polygons so the numpy
    ops are large enough to release it — which is what makes ``executor``
    give real parallelism rather than time-slicing.

    Args:
        snapshot: List of (centroid_x, centroid_y, vertices_np) per polygon
        image_size: Output image size
        r_max: Maximum radius for scaling
        device: Torch device for output tensor
        executor: Optional ThreadPoolExecutor to split the polygons over
        n_threads: Number of strides to split the polygons into
        out: Optional (n_polygons, image_size, image_size) tensor to fill

    Returns:
        channeled_image: (n_polygons, image_size, image_size) tensor
    """
    polys = _snapshot_to_polys(snapshot, image_size, r_max)
    n_polygons = len(polys)
    masks = np.zeros((n_polygons, image_size, image_size), dtype=np.float32)

    if executor is not None and n_threads > 1 and n_polygons > 1:
        # Stride the polygons across threads so each gets a mix of sizes.
        # masks[k::n_threads] is a view, so the workers fill `masks` in place;
        # .result() is only awaited to propagate exceptions.
        futures = [
            executor.submit(rasterize_batch, polys[k::n_threads],
                            (image_size, image_size), out=masks[k::n_threads])
            for k in range(min(n_threads, n_polygons))
        ]
        for fut in futures:
            fut.result()
    else:
        rasterize_batch(polys, (image_size, image_size), out=masks)

    if out is not None:
        out.copy_(torch.from_numpy(masks))
        return out
    return torch.tensor(masks, device=device, dtype=torch.float32)


def generate_combined_image(positions, channeled_image, domain_size, r_max, device=None):
    """Generates a combined 2D image representation of the polygons.

    Args:
        positions: Tensor of polygon centroid positions (n_polygons, 2)
        channeled_image: Multi-channel image (n_polygons, image_size, image_size)
        domain_size: Size of the simulation domain
        r_max: Maximum radius for scaling the image
        device: Device to use for tensors (defaults to channeled_image device)

    Returns:
        combined_image: Single-channel image (combined_image_size, combined_image_size)
    """
    if device is None:
        device = channeled_image.device

    n_polygons = positions.shape[0]
    channeled_image_width = channeled_image.shape[1]
    channeled_image_dx = 2 * r_max / (channeled_image_width - 1)
    combined_image_size = int(domain_size / channeled_image_dx)
    combined_image = torch.zeros((combined_image_size, combined_image_size), device=device)

    for i in range(n_polygons):
        pos = positions[i]
        exact_center_x = pos[0] / channeled_image_dx
        exact_center_y = pos[1] / channeled_image_dx
        center_x = int(exact_center_x)
        center_y = int(exact_center_y)

        # Subpixel shift amount
        shift_x = exact_center_x - center_x
        shift_y = exact_center_y - center_y

        half_width = channeled_image_width // 2

        x_start = max(center_x - half_width, 0)
        x_end = min(center_x + half_width, combined_image_size)
        y_start = max(center_y - half_width, 0)
        y_end = min(center_y + half_width, combined_image_size)

        ch_x_start = max(half_width - center_x, 0)
        ch_x_end = channeled_image_width - max(
            center_x + half_width - combined_image_size, 0
        )
        ch_y_start = max(half_width - center_y, 0)
        ch_y_end = channeled_image_width - max(
            center_y + half_width - combined_image_size, 0
        )

        if x_start < x_end and y_start < y_end:
            # Shift the channeled image by the subpixel amount
            shifted_image = shift(channeled_image[i].cpu().numpy(), (shift_x, shift_y), order=1)
            shifted_image = torch.tensor(shifted_image, device=device, dtype=channeled_image.dtype)

            combined_image[x_start:x_end, y_start:y_end] += shifted_image[
                ch_x_start:ch_x_end, ch_y_start:ch_y_end
            ]
    return combined_image


def generate_random_centers(
    num_cells: int,
    bounds: Tuple[float, float, float, float],
    radii: np.ndarray,
    seed: Optional[int] = None
) -> List[Tuple[float, float]]:
    """Generate randomly positioned cell centers with minimum separation.

    Args:
        num_cells: Number of cells to place
        bounds: (xmin, xmax, ymin, ymax) bounds for placement
        radii: Array of radii for each cell (length num_cells)
        seed: Random seed for reproducibility
    """
    if seed is not None:
        np.random.seed(seed)

    xmin, xmax, ymin, ymax = bounds
    centers = []

    for i in range(num_cells):
        radius = radii[i]
        placed = False
        for _ in range(1000):
            x = np.random.uniform(xmin + radius, xmax - radius)
            y = np.random.uniform(ymin + radius, ymax - radius)

            # Check separation against all existing centers (sum of radii)
            valid = True
            for j, (cx, cy) in enumerate(centers):
                min_sep = radius + radii[j]
                if math.sqrt((x - cx)**2 + (y - cy)**2) < min_sep:
                    valid = False
                    break

            if valid:
                centers.append((x, y))
                placed = True
                break

        if not placed:
            print(f"Warning: Could only place {len(centers)} of {num_cells} cells")
            break

    return centers


def _grid_centers(
    num_cells: int,
    bounds: Tuple[float, float, float, float],
    radius: float,
    dx: float,
    dy: float,
    hex_offset: bool,
) -> List[Tuple[float, float]]:
    """Shared body for hex/square grid placement.

    Always emits exactly num_cells centers laid out on a roughly-square grid
    centered in `bounds`. If the grid would extend beyond bounds, prints a
    warning and continues (cells outside are still emitted — caller should
    increase domain_size if containment matters).
    """
    xmin, xmax, ymin, ymax = bounds
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2

    aspect = (xmax - xmin) / max(ymax - ymin, 1e-9)
    cols = max(1, int(round(math.sqrt(num_cells * aspect))))
    rows = max(1, math.ceil(num_cells / cols))

    grid_w = (cols - 1) * dx + (dx / 2 if hex_offset and rows > 1 else 0)
    grid_h = (rows - 1) * dy
    x0 = cx - grid_w / 2
    y0 = cy - grid_h / 2

    centers = []
    for i in range(rows):
        row_offset = dx / 2 if (hex_offset and i % 2 == 1) else 0
        for j in range(cols):
            centers.append((x0 + j * dx + row_offset, y0 + i * dy))
            if len(centers) == num_cells:
                break
        if len(centers) == num_cells:
            break

    xs = [c[0] for c in centers]
    ys = [c[1] for c in centers]
    if (min(xs) < xmin + radius or max(xs) > xmax - radius
            or min(ys) < ymin + radius or max(ys) > ymax - radius):
        print(f"Warning: grid placement extends beyond domain bounds "
              f"(x:[{min(xs):.2f},{max(xs):.2f}], y:[{min(ys):.2f},{max(ys):.2f}], "
              f"bounds:[{xmin},{xmax}]x[{ymin},{ymax}]). "
              f"Consider increasing domain_size.")
    return centers


def generate_hex_grid_centers(
    num_cells: int,
    bounds: Tuple[float, float, float, float],
    radius: float,
    edge_thickness: float = 0.0,
) -> List[Tuple[float, float]]:
    """Hexagonal lattice: every cell has 6 neighbors at contact distance
    `2*radius + edge_thickness`. Densest packing for equal circles (~90.7%).
    Always emits exactly num_cells (warns if grid spills past bounds).
    """
    dx = 2 * radius + edge_thickness
    dy = dx * math.sqrt(3) / 2
    return _grid_centers(num_cells, bounds, radius, dx, dy, hex_offset=True)


def generate_square_grid_centers(
    num_cells: int,
    bounds: Tuple[float, float, float, float],
    radius: float,
    edge_thickness: float = 0.0,
) -> List[Tuple[float, float]]:
    """Square lattice: 4 contacting neighbors per cell (~78.5% packing)."""
    d = 2 * radius + edge_thickness
    return _grid_centers(num_cells, bounds, radius, d, d, hex_offset=False)


def generate_polygon_vertices(
    center: Tuple[float, float],
    radius: float,
    num_vertices: int = 40,
    start_angle: float = 0.0
) -> List[Tuple[float, float]]:
    """Generate vertices for a circular polygon."""
    vertices = []
    for i in range(num_vertices):
        angle = start_angle + 2 * math.pi * i / num_vertices
        x = center[0] + radius * math.cos(angle)
        y = center[1] + radius * math.sin(angle)
        vertices.append((x, y))
    return vertices


class PolyhoopSimulation:
    """Generates training data from polyhoop polygon simulations."""

    def __init__(self, cfg: DictConfig, device: str = None):
        """Initialize the polyhoop simulation.

        Args:
            cfg: Full Hydra config containing data config with simulation parameters
            device: Device to use for output tensors ('cuda', 'cpu', or None for auto-detect)
        """
        self.cfg = cfg
        self.data = cfg.data
        self.ph = cfg.data.params  # polyhoop-specific params
        self.gen = cfg.data.generator

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Seed RNG
        torch.manual_seed(self.data.seed)
        np.random.seed(self.data.seed)
        polyhoop.seed(self.data.seed)

        # Thread configuration. libgomp otherwise grabs every logical CPU,
        # which is *slower* than 32-48 threads: measured s/frame on a 128-core
        # emeraldrapids node was 0.283 @32, 0.281 @48, 0.310 @128. Rasterization
        # peaks around 16 threads and regresses past 32 (memory-bandwidth
        # bound). Use sched_getaffinity, not cpu_count — LSF cgroup-enforces cpu.
        n_cpus = len(os.sched_getaffinity(0))
        self.omp_threads = int(self.ph.get("omp_threads", min(48, n_cpus)))
        self.raster_threads = int(self.ph.get("raster_threads", min(16, n_cpus)))
        polyhoop.set_num_threads(self.omp_threads)

        self.dt = self.data.dt
        self.num_substeps = self.ph.num_substeps
        self.simulation_dt = self.dt / self.num_substeps

        # Configure polyhoop parameters
        params = polyhoop.get_params()
        params.dt = self.simulation_dt
        # Global gam / kl are no longer used by the force loop (each polygon
        # carries its own values), but set them to a scalar anyway so other
        # diagnostic code can read a sensible value. If the config specifies
        # a dist, skip — per-cell values get populated below.
        gam_spec = self.ph.get("gam", 1e3)
        params.gam = 1e3 if is_dist_param(gam_spec) else float(gam_spec)
        params.ka = self.ph.ka
        params.kr = self.ph.kr
        kh_spec_global = self.ph.get("kh", 1e6)
        params.kh = 1e6 if is_dist_param(kh_spec_global) else float(kh_spec_global)
        kl_spec_global = self.ph.get("kl", 1e4)
        params.kl = 1e4 if is_dist_param(kl_spec_global) else float(kl_spec_global)
        params.kb = self.ph.get("kb", 0)
        params.cv = self.ph.cv
        params.fp = self.ph.get("fp", 0)
        params.Dr = self.ph.get("Dr", 0.5)
        params.h = self.ph.get("edge_thickness", 0.01)
        params.alpha_mu = self.ph.get("alpha_mu", 0)
        params.alpha_CV = self.ph.get("alpha_CV", 0)
        params.oscillatory_growth = self.ph.get("oscillatory_growth", False)
        params.position_noise_std = self.ph.get("position_noise_std", 0)
        # No division - keep Amax infinite (default)
        polyhoop.set_params(params)

        self.oscillatory_growth = params.oscillatory_growth

        # Create ensemble
        self.ensemble = polyhoop.Ensemble()

        # Initialize radii (supports both fixed values and dist specs)
        radius_spec = self.ph.radius
        radii_np = resolve_dist_param(radius_spec, self.data.n_objects)
        self.radii = torch.tensor(radii_np, device=self.device, dtype=torch.float32)

        num_vertices = self.ph.get("num_vertices", 40)
        radii_np = self.radii.cpu().numpy()

        # Pre-sample area oscillation phase/amp/period so each polygon can be
        # instantiated with its phase-adjusted target area. C++ integrates
        # alpha_eff = amp·2π/P·cos(2π(t-phase)/P) from t=0, which yields
        # A0(t) = A0(0) + amp·[sin(2π(t-phase)/P) + sin(2π·phase/P)].
        # With a uniform A0(0)=π·r², every cell returns to π·r² at t=k·P,
        # making the random phase visually irrelevant. Subtracting
        # amp·sin(2π·phase/P) from A0(0) turns A0(t) into a pure sinusoid.
        # We bake this by shrinking/growing each starting polygon so the C++
        # constructor's A0 = area() already equals the target — avoids a
        # spring-force impulse at t=0 that would otherwise deform cells and
        # spike neighbor interactions.
        area_fluct_period_np = resolve_dist_param(
            self.ph.get("area_fluct_period", 0), self.data.n_objects
        )
        area_fluct_amp_np = resolve_dist_param(
            self.ph.get("area_fluct_amp", 0), self.data.n_objects
        )
        random_phases_cfg = self.ph.get("random_phases", True)
        if random_phases_cfg and self.ph.get("oscillatory_growth", False):
            area_phase_np = np.where(
                area_fluct_period_np > 0,
                np.random.uniform(0, 1, size=self.data.n_objects) * area_fluct_period_np,
                0.0,
            )
        else:
            area_phase_np = np.zeros(self.data.n_objects)

        if self.ph.get("oscillatory_growth", False):
            safe_period = np.where(area_fluct_period_np > 0, area_fluct_period_np, 1.0)
            area_A0_offset_np = np.where(
                area_fluct_period_np > 0,
                area_fluct_amp_np * np.sin(2 * np.pi * area_phase_np / safe_period),
                0.0,
            )
        else:
            area_A0_offset_np = np.zeros(self.data.n_objects)

        # Shrink or grow each starting polygon so its area equals the target
        # A0(0) = π·r² − amp·sin(2π·phase/P). Radius scales as sqrt(A/πr²).
        base_areas = math.pi * radii_np**2
        target_A0 = np.maximum(base_areas - area_A0_offset_np, 1e-6)
        radii_adjusted_np = radii_np * np.sqrt(target_A0 / base_areas)

        # Pack using whichever is larger (original vs phase-adjusted) so cells
        # never overlap regardless of the sampled phase.
        packing_radii = np.maximum(radii_np, radii_adjusted_np)

        # Placement strategy: random rejection sampling (default) or grid.
        # Grid modes use a uniform spacing (max packing radius + edge thickness)
        # so every cell starts just inside the contact zone; density_fraction
        # is ignored for grids since spacing comes from radius alone.
        placement = self.ph.get("placement", "random")
        if placement == "random":
            density_fraction = self.ph.get("density_fraction", 1.0)
            if density_fraction <= 0:
                raise ValueError("density_fraction must be positive")
            # Scale placement bounds: smaller density_fraction -> larger area
            scale = 1.0 / math.sqrt(density_fraction)
            placement_size = self.data.domain_size * scale
            offset = (placement_size - self.data.domain_size) / 2
            centers = generate_random_centers(
                num_cells=self.data.n_objects,
                bounds=(
                    -offset,
                    self.data.domain_size + offset,
                    -offset,
                    self.data.domain_size + offset,
                ),
                radii=packing_radii,
                seed=self.data.seed,
            )
        elif placement in ("hex_grid", "square_grid"):
            grid_radius = float(np.max(packing_radii))
            edge_h = float(self.ph.get("edge_thickness", 0.01))
            grid_fn = (
                generate_hex_grid_centers
                if placement == "hex_grid"
                else generate_square_grid_centers
            )
            centers = grid_fn(
                num_cells=self.data.n_objects,
                bounds=(0.0, self.data.domain_size,
                        0.0, self.data.domain_size),
                radius=grid_radius,
                edge_thickness=edge_h,
            )
            # Optionally scramble which cell ID lands at each lattice site so
            # per-cell params (kh, gam, kl, ...) are randomly distributed
            # across the grid rather than assigned in row-major order. The
            # lattice positions themselves are unchanged.
            if self.ph.get("shuffle_placement", False):
                np.random.shuffle(centers)
        else:
            raise ValueError(
                f"Unknown placement: {placement!r}. "
                "Use 'random', 'hex_grid', or 'square_grid'."
            )

        # Optional axis-aligned wall box. Centered in the domain. Side
        # length comes from `wall_box_size`: positive value sets it explicitly,
        # 0 (default) auto-fits to the placed cell extent + one radius pad
        # so cells start just inside the walls. Walls reuse polyhoop's own
        # repulsion (kr) and damping (cc); wall_kh defaults to 0 (no adhesion).
        # We always write wall params (zeroing them when disabled) because
        # polyhoop's Parameters struct is a process-global — leaving stale
        # values from a previous sim would silently activate the walls.
        wall_params = polyhoop.get_params()
        if self.ph.get("wall_box", False):
            xs = np.array([c[0] for c in centers])
            ys = np.array([c[1] for c in centers])
            max_radius = float(np.max(packing_radii))
            edge_h = float(self.ph.get("edge_thickness", 0.01))
            user_size = float(self.ph.get("wall_box_size", 0))
            if user_size > 0:
                L = user_size
            else:
                # Per-side padding around the placed cell extent. Defaults to
                # one max_radius so cells start fully inside the wall; set
                # `wall_pad` smaller to squeeze the box closer to the lattice.
                pad = float(self.ph.get("wall_pad", max_radius))
                cell_extent = max(xs.max() - xs.min(), ys.max() - ys.min())
                L = cell_extent + 2 * pad + 2 * edge_h
            cx = self.data.domain_size / 2
            cy = self.data.domain_size / 2
            wall_params.wall_xmin = cx - L / 2
            wall_params.wall_xmax = cx + L / 2
            wall_params.wall_ymin = cy - L / 2
            wall_params.wall_ymax = cy + L / 2
            wall_params.wall_kh = float(self.ph.get("wall_kh", 0))
            self.wall_box = (wall_params.wall_xmin, wall_params.wall_xmax,
                             wall_params.wall_ymin, wall_params.wall_ymax)
        else:
            wall_params.wall_xmin = 0
            wall_params.wall_xmax = 0
            wall_params.wall_ymin = 0
            wall_params.wall_ymax = 0
            wall_params.wall_kh = 0
            self.wall_box = None
        polyhoop.set_params(wall_params)

        # Add polygons to ensemble using the phase-adjusted radii so the C++
        # constructor sets A0 = area() = target A0(0) directly.
        for i, center in enumerate(centers):
            vertices = generate_polygon_vertices(center, radii_adjusted_np[i], num_vertices)
            self.ensemble.add_polygon(vertices, phase=False)

        self.n_polygons = self.ensemble.num_polygons()
        if self.n_polygons != self.data.n_objects:
            print(f"Warning: Only placed {self.n_polygons} of {self.data.n_objects} polygons")
            # Trim pre-sampled per-cell arrays to match actual number of polygons
            self.radii = self.radii[:self.n_polygons]
            area_fluct_period_np = area_fluct_period_np[:self.n_polygons]
            area_fluct_amp_np = area_fluct_amp_np[:self.n_polygons]
            area_phase_np = area_phase_np[:self.n_polygons]

        # Initialize growth rates (supports both fixed values and dist specs)
        growth_rate_spec = self.ph.get("growth_rate", 0)
        growth_rates_np = resolve_dist_param(growth_rate_spec, self.n_polygons)
        self.growth_rates = torch.tensor(growth_rates_np, device=self.device, dtype=torch.float32)
        for i in range(self.n_polygons):
            poly = self.ensemble.get_polygon(i)
            poly.alpha = float(self.growth_rates[i])

        # Area fluctuation period/amp were pre-sampled above so polygons could
        # be built with phase-adjusted radii.
        self.area_fluct_periods = torch.tensor(area_fluct_period_np, device=self.device, dtype=torch.float32)
        self.area_fluct_amps = torch.tensor(area_fluct_amp_np, device=self.device, dtype=torch.float32)

        # Initialize per-cell line tension
        gam_spec = self.ph.get("gamma", 1e3)
        gam_np = resolve_dist_param(gam_spec, self.n_polygons)
        self.gams = torch.tensor(gam_np, device=self.device, dtype=torch.float32)

        gam_fluct_period_spec = self.ph.get("gam_fluct_period", 0)
        gam_fluct_period_np = resolve_dist_param(gam_fluct_period_spec, self.n_polygons)
        self.gam_fluct_periods = torch.tensor(gam_fluct_period_np, device=self.device, dtype=torch.float32)

        gam_fluct_amp_spec = self.ph.get("gam_fluct_amp", 0)
        gam_fluct_amp_np = resolve_dist_param(gam_fluct_amp_spec, self.n_polygons)
        self.gam_fluct_amps = torch.tensor(gam_fluct_amp_np, device=self.device, dtype=torch.float32)

        # Per-cell exponential gamma ramp. Config value is the log10 ratio
        # over the full recording period: gamma_rate = log10(target/initial),
        # i.e. gamma_rate=4 means gam grows by 10^4 during the trajectory.
        # Convert to a per-time natural-log rate for the C++ force loop:
        #   gam_eff(t) = gam * exp(rate_internal * (t - t_ref))
        # with rate_internal = gamma_rate * ln(10) / (trajectory_length * dt).
        # Activated only when recording starts (gam_rate stays 0 during init).
        gamma_rate_spec = self.ph.get("gamma_rate", 0)
        gamma_rate_np = resolve_dist_param(gamma_rate_spec, self.n_polygons)
        total_recording_time = self.gen.trajectory_length * self.dt
        if total_recording_time > 0:
            rate_internal = gamma_rate_np * math.log(10) / total_recording_time
        else:
            rate_internal = np.zeros_like(gamma_rate_np)
        self.gam_rates = torch.tensor(rate_internal, device=self.device, dtype=torch.float32)

        # Initialize per-cell elastic tension (reads the same 'kl' key as the
        # global scalar; scalar -> all cells same, dist -> per-cell variation)
        kl_spec = self.ph.get("kl", 1e4)
        kl_np = resolve_dist_param(kl_spec, self.n_polygons)
        self.kls = torch.tensor(kl_np, device=self.device, dtype=torch.float32)

        kl_fluct_period_spec = self.ph.get("kl_fluct_period", 0)
        kl_fluct_period_np = resolve_dist_param(kl_fluct_period_spec, self.n_polygons)
        self.kl_fluct_periods = torch.tensor(kl_fluct_period_np, device=self.device, dtype=torch.float32)

        kl_fluct_amp_spec = self.ph.get("kl_fluct_amp", 0)
        kl_fluct_amp_np = resolve_dist_param(kl_fluct_amp_spec, self.n_polygons)
        self.kl_fluct_amps = torch.tensor(kl_fluct_amp_np, device=self.device, dtype=torch.float32)

        # Per-cell exponential kl ramp (same semantics as gamma_rate: log10
        # ratio over the full recording period). Activated at recording start.
        kl_rate_spec = self.ph.get("kl_rate", 0)
        kl_rate_np = resolve_dist_param(kl_rate_spec, self.n_polygons)
        if total_recording_time > 0:
            kl_rate_internal = kl_rate_np * math.log(10) / total_recording_time
        else:
            kl_rate_internal = np.zeros_like(kl_rate_np)
        self.kl_rates = torch.tensor(kl_rate_internal, device=self.device, dtype=torch.float32)

        # Initialize per-cell adhesion strength. Reads the same 'kh' key as
        # the global scalar; scalar -> all cells same, dist -> per-cell
        # variation. Pairs in interaction() couple via sqrt(kh_a * kh_b),
        # which produces Steinberg-style sorting whenever values differ.
        kh_spec = self.ph.get("kh", 1e6)
        kh_np = resolve_dist_param(kh_spec, self.n_polygons)
        self.khs = torch.tensor(kh_np, device=self.device, dtype=torch.float32)

        kh_fluct_period_spec = self.ph.get("kh_fluct_period", 0)
        kh_fluct_period_np = resolve_dist_param(kh_fluct_period_spec, self.n_polygons)
        self.kh_fluct_periods = torch.tensor(kh_fluct_period_np, device=self.device, dtype=torch.float32)

        kh_fluct_amp_spec = self.ph.get("kh_fluct_amp", 0)
        kh_fluct_amp_np = resolve_dist_param(kh_fluct_amp_spec, self.n_polygons)
        self.kh_fluct_amps = torch.tensor(kh_fluct_amp_np, device=self.device, dtype=torch.float32)

        kh_rate_spec = self.ph.get("kh_rate", 0)
        kh_rate_np = resolve_dist_param(kh_rate_spec, self.n_polygons)
        if total_recording_time > 0:
            kh_rate_internal = kh_rate_np * math.log(10) / total_recording_time
        else:
            kh_rate_internal = np.zeros_like(kh_rate_np)
        self.kh_rates = torch.tensor(kh_rate_internal, device=self.device, dtype=torch.float32)

        # initialize elongation values
        elongation_spec = self.ph.get("elongation", 0.5)
        elongation_np = resolve_dist_param(elongation_spec, self.n_polygons)
        self.elongations = torch.tensor(elongation_np, device=self.device, dtype=torch.float32)

        # Randomize per-cell oscillation phases so cells don't start in sync.
        # Phase sampled uniformly in [0, period); zero when period = 0.
        # Area phases were pre-sampled above (needed for phase-adjusted radii);
        # gam/kl phases are sampled here since they apply instantaneously and
        # don't affect initial polygon geometry.
        random_phases = self.ph.get("random_phases", True)
        if random_phases:
            gam_phase_np = np.where(
                gam_fluct_period_np > 0,
                np.random.uniform(0, 1, size=self.n_polygons) * gam_fluct_period_np,
                0,
            )
            kl_phase_np = np.where(
                kl_fluct_period_np > 0,
                np.random.uniform(0, 1, size=self.n_polygons) * kl_fluct_period_np,
                0,
            )
            kh_phase_np = np.where(
                kh_fluct_period_np > 0,
                np.random.uniform(0, 1, size=self.n_polygons) * kh_fluct_period_np,
                0,
            )
        else:
            gam_phase_np = np.zeros(self.n_polygons)
            kl_phase_np = np.zeros(self.n_polygons)
            kh_phase_np = np.zeros(self.n_polygons)
        self.area_fluct_phases = torch.tensor(area_phase_np, device=self.device, dtype=torch.float32)
        self.gam_fluct_phases = torch.tensor(gam_phase_np, device=self.device, dtype=torch.float32)
        self.kl_fluct_phases = torch.tensor(kl_phase_np, device=self.device, dtype=torch.float32)
        self.kh_fluct_phases = torch.tensor(kh_phase_np, device=self.device, dtype=torch.float32)

        # Set per-polygon fluctuation, tension, and elongation values
        for i in range(self.n_polygons):
            poly = self.ensemble.get_polygon(i)
            poly.area_fluct_period = float(self.area_fluct_periods[i])
            poly.area_fluct_amp = float(self.area_fluct_amps[i])
            poly.area_fluct_phase = float(self.area_fluct_phases[i])
            poly.gam = float(self.gams[i])
            poly.gam_fluct_period = float(self.gam_fluct_periods[i])
            poly.gam_fluct_amp = float(self.gam_fluct_amps[i])
            poly.gam_fluct_phase = float(self.gam_fluct_phases[i])
            poly.kl = float(self.kls[i])
            poly.kl_fluct_period = float(self.kl_fluct_periods[i])
            poly.kl_fluct_amp = float(self.kl_fluct_amps[i])
            poly.kl_fluct_phase = float(self.kl_fluct_phases[i])
            poly.kh = float(self.khs[i])
            poly.kh_fluct_period = float(self.kh_fluct_periods[i])
            poly.kh_fluct_amp = float(self.kh_fluct_amps[i])
            poly.kh_fluct_phase = float(self.kh_fluct_phases[i])
            poly.elongation = float(self.elongations[i])
        # Compute maximum possible cell radius for image generation.
        # Include init_steps since cells grow during warm-up too.
        max_initial_radius = self.radii.max().item()
        init_steps = self.gen.get("init_steps", 0)
        total_time = self.dt * (self.gen.trajectory_length + init_steps)
        initial_area = math.pi * max_initial_radius**2

        if self.oscillatory_growth:
            # Oscillatory mode: area swings by ±amp around initial
            max_fluct_amp = self.area_fluct_amps.max().item()
            max_area = initial_area + max_fluct_amp
        else:
            # Linear growth mode: area increases linearly
            alpha_max = self.growth_rates.max().item()
            max_area = initial_area + alpha_max * total_time

        self.max_possible_radius = math.sqrt(max_area / math.pi)

        # Add safety margin for deformations
        self.max_possible_radius *= 1.5

        # Initialize centroid positions and velocities
        self.positions = self._get_centroids()
        self.velocities = torch.zeros(self.n_polygons, 2, device=self.device)

    def get_simulation_metadata(self) -> dict:
        """Return simulation-specific metadata for caching.

        This metadata should be stored with the cached data and loaded
        during training instead of being recalculated.

        Returns:
            Dict with:
                - rs: Initial radii tensor (n_objects,)
                - growth_rates: Growth rates tensor (n_objects,)
                - area_fluct_periods: Area fluctuation periods tensor (n_objects,)
                - area_fluct_amps: Area fluctuation amplitudes tensor (n_objects,)
                - r_max: Maximum possible radius (scalar)
        """
        return {
            "rs": self.radii.cpu(),
            "growth_rates": self.growth_rates.cpu(),
            "area_fluct_periods": self.area_fluct_periods.cpu(),
            "area_fluct_amps": self.area_fluct_amps.cpu(),
            "area_fluct_phases": self.area_fluct_phases.cpu(),
            "gams": self.gams.cpu(),
            "gam_fluct_periods": self.gam_fluct_periods.cpu(),
            "gam_fluct_amps": self.gam_fluct_amps.cpu(),
            "gam_fluct_phases": self.gam_fluct_phases.cpu(),
            "gam_rates": self.gam_rates.cpu(),
            "kls": self.kls.cpu(),
            "kl_fluct_periods": self.kl_fluct_periods.cpu(),
            "kl_fluct_amps": self.kl_fluct_amps.cpu(),
            "kl_fluct_phases": self.kl_fluct_phases.cpu(),
            "kl_rates": self.kl_rates.cpu(),
            "khs": self.khs.cpu(),
            "kh_fluct_periods": self.kh_fluct_periods.cpu(),
            "kh_fluct_amps": self.kh_fluct_amps.cpu(),
            "kh_fluct_phases": self.kh_fluct_phases.cpu(),
            "kh_rates": self.kh_rates.cpu(),
            "elongations": self.elongations.cpu(),
            "r_max": torch.tensor(self.max_possible_radius),
        }

    def _get_centroids(self) -> torch.Tensor:
        """Extract centroid positions from all polygons."""
        centroids = torch.zeros(self.n_polygons, 2, device=self.device)
        for i in range(self.n_polygons):
            poly = self.ensemble.get_polygon(i)
            c = poly.centroid()
            centroids[i, 0] = c.x
            centroids[i, 1] = c.y
        return centroids

    def _get_effective_radii(self) -> torch.Tensor:
        """Get effective radius from polygon areas: r = sqrt(A/pi)."""
        radii = torch.zeros(self.n_polygons, device=self.device)
        for i in range(self.n_polygons):
            poly = self.ensemble.get_polygon(i)
            area = poly.A if poly.A > 0 else poly.area()
            radii[i] = math.sqrt(abs(area) / math.pi)
        return radii

    def _compute_interaction_graph(self) -> torch.sparse.Tensor:
        """Ground-truth contact graph from the C++ force-loop criterion.

        Two polygons are "in contact" iff any vertex of one lies within
        drmax = h + sh + ss of an edge of the other (same check that gates
        adhesion/repulsion in polyhoop_bindings.cpp::interaction). Returns a
        symmetric sparse COO adjacency matrix.
        """
        pairs = self.ensemble.get_contact_pairs()
        if len(pairs) == 0:
            return torch.sparse_coo_tensor(
                torch.zeros((2, 0), dtype=torch.long, device=self.device),
                torch.zeros(0, dtype=torch.bool, device=self.device),
                (self.n_polygons, self.n_polygons),
                dtype=torch.bool, device=self.device,
            )

        pairs_t = torch.tensor(pairs, dtype=torch.long, device=self.device).t()  # (2, M)
        indices = torch.cat([pairs_t, pairs_t.flip(0)], dim=1)  # symmetrise
        values = torch.ones(indices.shape[1], dtype=torch.bool, device=self.device)
        return torch.sparse_coo_tensor(
            indices, values, (self.n_polygons, self.n_polygons),
            dtype=torch.bool, device=self.device,
        )

    def step(self):
        """Performs one simulation step.

        Returns:
            new_positions: Updated centroid positions
            new_velocities: Updated centroid velocities
            forces: Approximate forces (from acceleration)
            interaction_graph: Sparse adjacency matrix of interactions
        """
        old_positions = self._get_centroids()

        # Run polyhoop step
        self.ensemble.run(self.num_substeps)

        # Get new state
        new_positions = self._get_centroids()
        new_velocities = (new_positions - old_positions) / self.dt

        # Approximate forces from velocity change (F = m*a, assume m=1)
        forces = (new_velocities - self.velocities) / self.dt

        # Compute interaction graph
        interaction_graph = self._compute_interaction_graph()

        # Update stored state
        self.positions = new_positions
        self.velocities = new_velocities

        return new_positions, new_velocities, forces, interaction_graph

    def relax(self):
        """Performs relaxation steps."""
        relaxation_steps = self.ph.get("relaxation_steps", 0)
        if relaxation_steps > 0:
            self.ensemble.run(relaxation_steps)
            self.positions = self._get_centroids()
            self.velocities = torch.zeros(self.n_polygons, 2, device=self.device)

    def _generate_image(self) -> torch.Tensor:
        """Generates channeled images for each polygon.

        Returns:
            channeled_image: (n_polygons, image_size, image_size) tensor
        """
        image_size = self.data.image_size
        r_max = self.max_possible_radius

        # Create grid centered at origin
        grid = torch.linspace(-r_max , r_max , image_size, device=self.device)
        grid_xx, grid_yy = torch.meshgrid(grid, grid, indexing="ij")

        channeled_image = torch.zeros(self.n_polygons, image_size, image_size, device=self.device)

        for i in range(self.n_polygons):
            poly = self.ensemble.get_polygon(i)
            centroid = poly.centroid()
            vertices = poly.get_vertex_positions()

            # Convert vertices to tensor, centered at origin
            verts = torch.tensor(vertices, device=self.device, dtype=torch.float32)
            verts[:, 0] -= centroid.x
            verts[:, 1] -= centroid.y

            # Rasterize polygon using ray casting
            channeled_image[i] = self._rasterize_polygon(verts, grid_xx, grid_yy)

        return channeled_image

    def _rasterize_polygon(self, vertices: torch.Tensor, grid_xx: torch.Tensor, grid_yy: torch.Tensor) -> torch.Tensor:
        """Rasterize a polygon onto a grid using skimage polygon2mask.

        Args:
            vertices: (n_vertices, 2) polygon vertices
            grid_xx, grid_yy: meshgrid coordinates

        Returns:
            mask: (image_size, image_size) binary mask
        """
        image_size = grid_xx.shape[0]
        verts_np = vertices.cpu().numpy()

        scale = (image_size - 1) / (2 * self.max_possible_radius)
        center = (image_size - 1) / 2

        # row corresponds to x, col corresponds to y (matching meshgrid "ij" indexing)
        verts_np = np.stack((verts_np[:, 0] * scale + center,
                             verts_np[:, 1] * scale + center), axis=-1)

        mask_np = polygon2mask((image_size, image_size), verts_np)
        mask = torch.tensor(mask_np, dtype=torch.float32, device=self.device)
        return mask

    def _snapshot_polygons(self):
        """Extract polygon vertices and centroids as numpy arrays.

        This must be called before the next step() mutates the ensemble.
        The returned snapshot can be passed to _generate_image_from_snapshot
        from any thread without touching the ensemble.

        Returns:
            List of (centroid_x, centroid_y, vertices_np) tuples
        """
        snapshot = []
        for i in range(self.n_polygons):
            poly = self.ensemble.get_polygon(i)
            centroid = poly.centroid()
            vertices = poly.get_vertex_positions()
            verts_np = np.array(vertices, dtype=np.float32)
            snapshot.append((centroid.x, centroid.y, verts_np))
        return snapshot

    def _generate_image_parallel(self, snapshot=None, executor=None, out=None) -> torch.Tensor:
        """Generate channeled images from a polygon snapshot (thread-safe).

        If snapshot is None, takes a snapshot from the current ensemble state.

        Args:
            snapshot: Optional pre-extracted snapshot from _snapshot_polygons()
            executor: Optional ThreadPoolExecutor to split polygons over
            out: Optional (n_polygons, image_size, image_size) tensor to fill

        Returns:
            channeled_image: (n_polygons, image_size, image_size) tensor
        """
        if snapshot is None:
            snapshot = self._snapshot_polygons()
        return _generate_image_from_snapshot(
            snapshot, self.data.image_size, self.max_possible_radius, self.device,
            executor=executor, n_threads=self.raster_threads, out=out,
        )

    def _generate_combined_image(self, positions, channeled_image, r_max=None):
        """Generates a combined 2D image representation of the polygons."""
        if r_max is None:
            r_max = self.max_possible_radius
        return generate_combined_image(positions, channeled_image, self.data.domain_size, r_max, self.device)

    def generate_combined_images_batch(self, all_positions, all_channeled_images, r_max=None):
        """Generates combined images for a batch of positions and channeled images."""
        n_samples = all_positions.shape[0]

        first_combined = self._generate_combined_image(
            all_positions[0], all_channeled_images[0], r_max=r_max
        )
        combined_size = first_combined.shape[0]

        combined_images = torch.zeros((n_samples, combined_size, combined_size), device=self.device)
        combined_images[0] = first_combined

        for i in range(1, n_samples):
            combined_images[i] = self._generate_combined_image(
                all_positions[i], all_channeled_images[i], r_max=r_max
            )

        return combined_images

    def generate(self):
        """Generates a dataset of polygon interactions.

        Returns:
            all_times: (n_samples, 1)
            all_current_states: (n_samples, n_polygons, 4) - [pos_x, pos_y, vel_x, vel_y]
            all_next_states: (n_samples, n_polygons, 4) - all_current_states one step on
            all_forces: (n_samples, n_polygons * 2)
            all_channeled_images: (n_samples, n_polygons, H, W) - same frame as all_current_states
            all_interaction_graphs: List of sparse tensors - same frame as all_current_states
        """
        all_current_states = []
        all_next_states = []
        all_forces = []
        all_interaction_graphs = []
        all_times = []

        # Preallocate the image buffer and fill it in place. Accumulating a list
        # of per-frame tensors and torch.stack-ing at the end doubles peak RAM
        # (for a 2000-frame, 600-cell, 128px trajectory that is ~78 GB live and
        # ~156 GB at the stack).
        image_size = self.data.image_size
        all_channeled_images = torch.empty(
            (self.gen.trajectory_length, self.n_polygons, image_size, image_size),
            device=self.device, dtype=torch.float32,
        )

        # Initialization period: advance the sim for N full steps before
        # recording, letting the system settle. Each init step runs
        # num_substeps substeps (matching the data-collection cadence).
        init_steps = self.gen.get("init_steps", 0)
        if init_steps > 0:
            # Optional compression toward domain center during init only.
            # Applies a uniform force per cell (centroid -> center direction)
            # that gradually packs cells together; disabled for recording.
            # The force is turned off for a trailing settle period so cells
            # decelerate via cv damping before data capture. Terminal velocity
            # under the force is v_term = force/cv; after time t the velocity
            # decays as v_term * exp(-cv*t). Solve for t until v < v_thresh.
            init_compress_force = self.gen.get("init_compress_force", 0)
            cv = self.ph.get("cv", 0)
            # v_thresh picked to match the natural (non-compression) motion
            # floor in polyhoop — cells already jiggle at this scale from
            # neighbor repulsion, so further settling is not measurable.
            v_thresh = 0.1
            if init_compress_force > 0 and cv > 0:
                terminal_v = init_compress_force / cv
                if terminal_v > v_thresh:
                    t_settle = math.log(terminal_v / v_thresh) / cv
                    settle_steps = math.ceil(t_settle / self.dt)
                else:
                    settle_steps = 0
            else:
                settle_steps = 0
            compress_steps = max(0, init_steps - settle_steps)

            params = polyhoop.get_params()
            prev_cf = params.compress_force
            prev_cx = params.compress_cx
            prev_cy = params.compress_cy
            force_on = init_compress_force > 0 and compress_steps > 0
            if force_on:
                params.compress_force = float(init_compress_force)
                params.compress_cx = self.data.domain_size / 2
                params.compress_cy = self.data.domain_size / 2
                polyhoop.set_params(params)

            try:
                for i in tqdm.trange(init_steps, desc="init"):
                    if force_on and i == compress_steps:
                        # Disable compression; enter settle phase
                        params.compress_force = prev_cf
                        params.compress_cx = prev_cx
                        params.compress_cy = prev_cy
                        polyhoop.set_params(params)
                        force_on = False
                    self.step()
            finally:
                if force_on:
                    params.compress_force = prev_cf
                    params.compress_cx = prev_cx
                    params.compress_cy = prev_cy
                    polyhoop.set_params(params)

        # Activate exponential gamma and kl ramps for the recording phase.
        # During init, both gam_rate and kl_rate were 0. Now set per-cell rates
        # plus the shared simulation_start_time so the ramp clocks effectively
        # start at 0 when data capture begins:
        #   gam_eff(t) = gam * exp(gam_rate * (t - simulation_start_time))
        #   kl_eff (t) = kl  * exp(kl_rate  * (t - simulation_start_time))
        if (torch.any(self.gam_rates != 0)
                or torch.any(self.kl_rates != 0)
                or torch.any(self.kh_rates != 0)):
            params = polyhoop.get_params()
            params.simulation_start_time = self.ensemble.t
            polyhoop.set_params(params)
            for i in range(self.n_polygons):
                poly = self.ensemble.get_polygon(i)
                g_rate = float(self.gam_rates[i])
                k_rate = float(self.kl_rates[i])
                h_rate = float(self.kh_rates[i])
                if g_rate != 0:
                    poly.gam_rate = g_rate
                if k_rate != 0:
                    poly.kl_rate = k_rate
                if h_rate != 0:
                    poly.kh_rate = h_rate

        # Two pools, both needed. The outer 1-thread pool overlaps rasterisation
        # of frame t with the physics step for frame t+1 (OpenMP-parallel, so it
        # drops the GIL). The inner pool splits the frame's polygons across
        # threads — which only pays off because rasterize_batch batches the
        # crossing test enough for numpy to release the GIL; skimage's
        # polygon2mask does not, despite the comment that used to be here.
        executor = ThreadPoolExecutor(max_workers=1)
        raster_executor = (
            ThreadPoolExecutor(max_workers=self.raster_threads)
            if self.raster_threads > 1 else None
        )
        pending_future = None

        # Frame t's state, raster and contact graph are all read from the
        # ensemble as step t-1 left it, so they describe one instant, and
        # next_states[t] is the state after step t. The graph step() returns is
        # therefore frame t+1's; frame 0's is read off the post-init state.
        interaction_graph = self._compute_interaction_graph()

        for t in tqdm.trange(self.gen.trajectory_length):
            current_states = torch.cat((self.positions, self.velocities), dim=1)

            # Snapshot polygon state (fast) then submit rasterisation to thread
            snapshot = self._snapshot_polygons()
            if pending_future is not None:
                # Wait for the previous frame (it writes into its own slot)
                pending_future.result()
            pending_future = executor.submit(
                self._generate_image_parallel, snapshot, raster_executor,
                all_channeled_images[t],
            )
            all_interaction_graphs.append(interaction_graph)

            # Perform simulation step
            new_positions, new_velocities, forces, interaction_graph = self.step()

            next_states = torch.cat((new_positions, new_velocities), dim=1)

            all_current_states.append(current_states)
            all_next_states.append(next_states)
            all_forces.append(forces.flatten())
            all_times.append(torch.tensor([t * self.dt], device=self.device))

        # Collect the last pending image
        if pending_future is not None:
            pending_future.result()
        executor.shutdown(wait=False)
        if raster_executor is not None:
            raster_executor.shutdown(wait=False)

        return (
            torch.stack(all_times),
            torch.stack(all_current_states),
            torch.stack(all_next_states),
            torch.stack(all_forces),
            all_channeled_images,
            torch.stack(all_interaction_graphs),
        )


#%%
if __name__ == "__main__":
    # --- Test and Visualization ---
    # get the default config
    from omegaconf import OmegaConf
    from hydra import compose, initialize
    #%%

    with initialize(config_path="configs"):
        # config should come from data/polyhoop.yaml
        test_cfg = compose(config_name="data/polyhoop")

    data = test_cfg.data
    n_samples_test = data.generator.trajectory_length

    # 1. Instantiate the simulation
    print(f"Creating PolyhoopSimulation with {data.n_objects} polygons...")
    sim = PolyhoopSimulation(test_cfg, device="cpu")

    print(f"Actually placed {sim.n_polygons} polygons")

    # 2. Generate the data
    print(f"Generating {n_samples_test} samples...")
    times, states, next_states, forces, channeled_images, interaction_graphs = sim.generate()

    positions = states[:, :, :2]

    # Generate combined images
    print("Generating combined images...")
    images = sim.generate_combined_images_batch(
        positions.view(n_samples_test, sim.n_polygons, 2), channeled_images
    )

    print("Data generation complete.")
    print(f"Images shape: {images.shape}")
    print(f"Channeled images shape: {channeled_images.shape}")
    print(f"States shape: {states.shape}")
    print(f"Interaction graphs shape: {len(interaction_graphs)}")
    print(f"Interaction graphs is sparse: {interaction_graphs[0].is_sparse}")

    first_graph_dense = interaction_graphs[0].to_dense()
    print(f"Average connections at first timestep: {first_graph_dense.sum():.2f}")

    # 3. Display the output
    print("Plotting results...")
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle("Polyhoop Generated Training Data")

    # Plot trajectories in 2D
    for i in range(min(sim.n_polygons, 50)):  # Limit for visibility
        ax1.plot(
            positions[:, i, 0].numpy(),
            positions[:, i, 1].numpy(),
            "o-",
            markersize=2,
        )
    ax1.set_title("Polygon Centroid Trajectories (2D)")
    ax1.set_xlabel("X Position")
    ax1.set_ylabel("Y Position")
    ax1.grid(True)
    ax1.axis("equal")

    # Histogram of velocities
    velocities = states[:, :, 2:4]
    speed = torch.norm(velocities, dim=2)
    import seaborn as sns
    sns.histplot(speed.flatten().numpy(), bins=50, ax=ax2, color='blue')
    ax2.set_title("Histogram of Polygon Speeds")

    # Sample combined image
    sample_image_idx = n_samples_test - 1
    im3 = ax3.imshow(
        images[sample_image_idx].numpy(),
        origin="lower",
        extent=[0, data.domain_size, 0, data.domain_size],
        cmap="viridis",
    )
    ax3.set_title(f"Combined Image at Time Step {sample_image_idx}")
    plt.colorbar(im3, ax=ax3, label="Intensity")

    # Interaction graph
    interaction_graph_dense = interaction_graphs[sample_image_idx].to_dense().numpy()
    im4 = ax4.imshow(interaction_graph_dense, cmap="binary", interpolation="nearest")
    ax4.set_title(f"Interaction Graph at Time Step {sample_image_idx}")
    ax4.set_xlabel("Polygon Index (j)")
    ax4.set_ylabel("Polygon Index (i)")
    ax4.set_xticks(range(0, sim.n_polygons, max(1, sim.n_polygons // 10)))
    ax4.set_yticks(range(0, sim.n_polygons, max(1, sim.n_polygons // 10)))
    plt.colorbar(im4, ax=ax4, label="Interaction (1=yes, 0=no)")

    num_interactions = interaction_graph_dense.sum()
    max_possible = sim.n_polygons * (sim.n_polygons - 1)
    ax4.text(
        0.02,
        0.98,
        f"Interactions: {int(num_interactions)}/{max_possible}\nSparsity: {100*(1-num_interactions/max_possible):.1f}%",
        transform=ax4.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig("polyhoop_test_output.png", dpi=150)
    print("Saved test output to polyhoop_test_output.png")

    # Make animation
    import matplotlib.animation as animation

    fig2, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(
        images[0].numpy(),
        origin="lower",
        extent=[0, data.domain_size, 0, data.domain_size],
        cmap="viridis",
    )
    ax.set_title("Polyhoop Simulation Over Time")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")

    def update(frame):
        im.set_array(images[frame].numpy())
        return [im]

    fps = 30
    skip_frames = 2
    ani = animation.FuncAnimation(
        fig2,
        update,
        frames=np.arange(0, n_samples_test, skip_frames),
        blit=True,
        interval=1000 / fps,
    )
    with tqdm.tqdm(total=n_samples_test // skip_frames) as pbar:
        callback = lambda *_: pbar.update()
        ani.save(
            "polyhoop_simulation.mp4",
            writer="ffmpeg",
            fps=fps,
            progress_callback=callback,
        )
    print("Saved animation to polyhoop_simulation.mp4")

    #%%
    # --- Oscillating Cell Test ---
    print("\n" + "="*60)
    print("Running oscillating cell test (500 steps)...")
    print("="*60)

    # Load base config and merge oscillating_cell experiment overrides
    osc_cfg = OmegaConf.load("configs/data/polyhoop.yaml")
    osc_exp = OmegaConf.load("configs/experiment/oscillating_cell.yaml")
    osc_cfg = OmegaConf.merge(osc_cfg, osc_exp.get("data", {}))
    osc_cfg = OmegaConf.create({"data": osc_cfg})

    # Override trajectory length to 500
    osc_cfg.data.generator.trajectory_length = 500

    print(f"Config: oscillatory_growth={osc_cfg.data.params.oscillatory_growth}")
    print(f"        radius={osc_cfg.data.params.radius}")
    print(f"        density_fraction={osc_cfg.data.params.density_fraction}")
    print(f"        area_fluct_period={osc_cfg.data.params.area_fluct_period}")
    print(f"        area_fluct_amp={osc_cfg.data.params.area_fluct_amp}")

    osc_sim = PolyhoopSimulation(osc_cfg, device="cpu")
    print(f"Created simulation with {osc_sim.n_polygons} polygons")
    print(f"Oscillatory growth: {osc_sim.oscillatory_growth}")

    # Run and time the simulation
    import time
    start = time.time()
    osc_times, osc_states, _, _, osc_images, _ = osc_sim.generate()
    elapsed = time.time() - start
    print(f"Generated {len(osc_times)} frames in {elapsed:.2f}s")

    # Generate combined images
    osc_positions = osc_states[:, :, :2]
    osc_combined = osc_sim.generate_combined_images_batch(osc_positions, osc_images)

    from matplotlib import animation
    # Save animation
    fig3, ax3 = plt.subplots(figsize=(6, 6))
    im3 = ax3.imshow(
        osc_combined[0].numpy(),
        origin="lower",
        extent=[0, osc_cfg.data.domain_size, 0, osc_cfg.data.domain_size],
        cmap="viridis",
    )
    ax3.set_title("Oscillating Cells")

    def update_osc(frame):
        im3.set_array(osc_combined[frame].numpy())
        return [im3]

    osc_ani = animation.FuncAnimation(
        fig3, update_osc, frames=len(osc_combined), blit=True, interval=1000/30
    )
    osc_ani.save("oscillating_cells.mp4", writer="ffmpeg", fps=30)
    print("Saved oscillating cell animation to oscillating_cells.mp4")
