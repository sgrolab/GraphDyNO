#!/usr/bin/env python3
"""
Example script demonstrating the PolyHoop Python bindings.

Usage:
    python run_polyhoop.py [input_file.off] [num_frames] [steps_per_frame]

The bindings support OpenMP parallelization when compiled with -fopenmp.
"""

import sys
import os
import math
from typing import List, Tuple, Optional

# Try to import numpy for advanced generation (optional)
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

# Try to import the compiled module
try:
    import polyhoop
except ImportError as e:
    print(f"Error: Could not import polyhoop module: {e}")
    print("\nTo build the module with OpenMP support, run:")
    print("  pip install pybind11")
    print("  c++ -O3 -shared -std=c++17 -fPIC -fopenmp $(python3 -m pybind11 --includes) "
          "polyhoop_bindings.cpp -o polyhoop$(python3-config --extension-suffix)")
    print("\nWithout OpenMP (single-threaded):")
    print("  c++ -O3 -shared -std=c++17 -fPIC $(python3 -m pybind11 --includes) "
          "polyhoop_bindings.cpp -o polyhoop$(python3-config --extension-suffix)")
    sys.exit(1)


# =============================================================================
# Polygon Generation Functions (consistent with generate_ensemble.py)
# =============================================================================

def generate_polygon_vertices(
    center: Tuple[float, float],
    radius: float,
    num_vertices: int = 40,
    start_angle: float = 0.0
) -> List[Tuple[float, float]]:
    """
    Generate vertices for a circular polygon.

    Args:
        center: (x, y) center coordinates
        radius: radius of the polygon
        num_vertices: number of vertices (default: 40)
        start_angle: starting angle in radians (default: 0)

    Returns:
        List of (x, y) tuples for polygon vertices
    """
    vertices = []
    for i in range(num_vertices):
        angle = start_angle + 2 * math.pi * i / num_vertices
        x = center[0] + radius * math.cos(angle)
        y = center[1] + radius * math.sin(angle)
        vertices.append((x, y))
    return vertices


def generate_grid_centers(
    rows: int,
    cols: int,
    spacing: float,
    origin: Tuple[float, float] = (0.0, 0.0)
) -> List[Tuple[float, float]]:
    """Generate cell centers in a regular grid pattern."""
    centers = []
    for i in range(rows):
        for j in range(cols):
            x = origin[0] + j * spacing
            y = origin[1] + i * spacing
            centers.append((x, y))
    return centers


def generate_hex_centers(
    rows: int,
    cols: int,
    spacing: float,
    origin: Tuple[float, float] = (0.0, 0.0)
) -> List[Tuple[float, float]]:
    """Generate cell centers in hexagonal packing pattern."""
    centers = []
    for i in range(rows):
        for j in range(cols):
            x = origin[0] + j * spacing + (i % 2) * spacing / 2
            y = origin[1] + i * spacing * math.sqrt(3) / 2
            centers.append((x, y))
    return centers


def generate_ring_centers(
    num_rings: int,
    cell_radius: float,
    center: Tuple[float, float] = (0.0, 0.0)
) -> List[Tuple[float, float]]:
    """Generate cell centers in concentric rings."""
    ring_radii = [(i + 1) * 2.5 * cell_radius for i in range(num_rings)]
    centers = []

    for ring_r in ring_radii:
        num_cells = max(6, int(2 * math.pi * ring_r / (2.2 * cell_radius)))
        for i in range(num_cells):
            angle = 2 * math.pi * i / num_cells
            x = center[0] + ring_r * math.cos(angle)
            y = center[1] + ring_r * math.sin(angle)
            centers.append((x, y))

    return centers


def generate_random_centers(
    num_cells: int,
    bounds: Tuple[float, float, float, float],
    radius: float,
    min_separation: Optional[float] = None,
    seed: Optional[int] = None
) -> List[Tuple[float, float]]:
    """
    Generate randomly positioned cell centers with minimum separation.

    Args:
        num_cells: number of cells to place
        bounds: (xmin, xmax, ymin, ymax) bounding box
        radius: cell radius (used for boundary padding)
        min_separation: minimum distance between centers (default: 2*radius)
        seed: random seed for reproducibility

    Returns:
        List of (x, y) center coordinates
    """
    if seed is not None:
        if HAS_NUMPY:
            np.random.seed(seed)
        else:
            import random
            random.seed(seed)

    min_sep = min_separation if min_separation else 2 * radius
    xmin, xmax, ymin, ymax = bounds
    centers = []

    for _ in range(num_cells):
        placed = False
        for _ in range(1000):  # max attempts
            if HAS_NUMPY:
                x = np.random.uniform(xmin + radius, xmax - radius)
                y = np.random.uniform(ymin + radius, ymax - radius)
            else:
                import random
                x = random.uniform(xmin + radius, xmax - radius)
                y = random.uniform(ymin + radius, ymax - radius)

            # Check minimum separation from existing centers
            valid = all(
                math.sqrt((x - cx)**2 + (y - cy)**2) >= min_sep
                for cx, cy in centers
            )
            if valid:
                centers.append((x, y))
                placed = True
                break

        if not placed:
            print(f"Warning: Could only place {len(centers)} of {num_cells} cells")
            break

    return centers


def print_openmp_info():
    """Print OpenMP configuration info."""
    print("OpenMP Configuration:")
    print(f"  Enabled: {polyhoop.openmp_enabled}")
    print(f"  Max threads: {polyhoop.get_max_threads()}")
    print(f"  Available processors: {polyhoop.get_num_procs()}")
    print()


def print_simulation_info(ensemble, frame=None):
    """Print information about the current simulation state."""
    prefix = f"Frame {frame}: " if frame is not None else ""
    print(f"{prefix}t={ensemble.t:.6f}, "
          f"{ensemble.num_polygons()} polygons, "
          f"{ensemble.total_vertices()} vertices")


def run_from_file(input_file, num_frames=100, steps_per_frame=100, output_prefix="frame"):
    """Run simulation from an OFF file."""
    print(f"Loading ensemble from: {input_file}")

    # Seed RNG for reproducibility
    polyhoop.seed(42)

    # Get and optionally modify parameters
    params = polyhoop.get_params()
    print(f"\nSimulation parameters:")
    print(f"  dt = {params.dt}")
    print(f"  h = {params.h}")
    print(f"  gam = {params.gam}")
    print(f"  ka = {params.ka}")
    print(f"  kr = {params.kr}")
    print(f"  fp = {params.fp} (polarity force)")
    print(f"  Dr = {params.Dr} (rotational diffusion)")
    print()

    # Run the simulation using the convenience function
    ensemble = polyhoop.run_simulation(
        input_file,
        num_frames=num_frames,
        steps_per_frame=steps_per_frame,
        output_prefix=output_prefix,
        verbose=True
    )

    return ensemble


def run_programmatic_example(num_frames=50, steps_per_frame=100, n_cells=25, pattern="hex"):
    """Run a simulation by creating polygons programmatically.

    Args:
        num_frames: Number of output frames
        steps_per_frame: Simulation steps between frames
        n_cells: Number of cells to create
        pattern: Cell arrangement pattern - "hex", "grid", "rings", or "random"
    """
    print(f"Running programmatic simulation example (pattern={pattern}, n_cells={n_cells})...")

    # Seed RNG for reproducibility
    polyhoop.seed(123)

    # Configure parameters for this example
    params = polyhoop.get_params()
    params.fp = 0  # Disable polarity force for this example
    params.dt = 1e-4
    params.gam = 1e3
    params.ka = 1e5
    params.kr = 1e7
    params.cv = 10
    params.alpha_mu = 1
    polyhoop.set_params(params)

    # Create an empty ensemble
    ensemble = polyhoop.Ensemble()

    # Cell parameters
    radius = 1.0
    spacing = 2.2
    num_vertices = 40

    # Generate cell centers based on pattern
    if pattern == "hex":
        # Hexagonal packing
        cols = int(math.ceil(math.sqrt(n_cells)))
        rows = int(math.ceil(n_cells / cols))
        centers = generate_hex_centers(rows=rows, cols=cols, spacing=spacing)
        centers = centers[:n_cells]

    elif pattern == "grid":
        # Regular grid
        cols = int(math.ceil(math.sqrt(n_cells)))
        rows = int(math.ceil(n_cells / cols))
        centers = generate_grid_centers(rows=rows, cols=cols, spacing=spacing)
        centers = centers[:n_cells]

    elif pattern == "rings":
        # Concentric rings - estimate num_rings needed for n_cells
        num_rings = max(1, int(math.ceil(math.sqrt(n_cells / 6))))
        centers = generate_ring_centers(num_rings=num_rings, cell_radius=radius)
        centers = centers[:n_cells]

    elif pattern == "random":
        # Random placement - estimate bounds from n_cells
        side = math.sqrt(n_cells) * spacing
        centers = generate_random_centers(
            num_cells=n_cells,
            bounds=(0, side, 0, side),
            radius=radius,
            seed=42
        )

    else:
        raise ValueError(f"Unknown pattern: {pattern}. Use 'hex', 'grid', 'rings', or 'random'")

    # Add polygons to ensemble
    for center in centers:
        vertices = generate_polygon_vertices(center, radius, num_vertices)
        ensemble.add_polygon(vertices, phase=False)

    print(f"Created {ensemble.num_polygons()} polygons")
    print_simulation_info(ensemble, frame=0)

    # Run simulation manually step by step
    for frame in range(1, num_frames + 1):
        ensemble.run(steps_per_frame)
        print_simulation_info(ensemble, frame=frame)
        # Optionally save VTK output
        ensemble.output_vtk(f"programmatic_{frame:06d}.vtp")

    print("\nFinal state:")
    print_simulation_info(ensemble, frame=num_frames)

    # Print some polygon details
    print("\nPolygon details:")
    for i in range(min(3, ensemble.num_polygons())):
        poly = ensemble.get_polygon(i)
        print(f"  Polygon {i}: area={poly.A:.6f}, "
              f"perimeter={poly.perimeter():.6f}, "
              f"vertices={poly.num_vertices()}, "
              f"polarity=({poly.polarity.x:.3f}, {poly.polarity.y:.3f})")

    return ensemble


def inspect_polygon(ensemble, idx):
    """Inspect a single polygon in detail."""
    poly = ensemble.get_polygon(idx)
    print(f"\nPolygon {idx} details:")
    print(f"  Phase: {poly.phase}")
    print(f"  Area (actual): {poly.A:.6f}")
    print(f"  Area (target): {poly.A0:.6f}")
    print(f"  Area (max/division): {poly.Amax}")
    print(f"  Growth rate: {poly.alpha}")
    print(f"  Perimeter: {poly.perimeter():.6f}")
    print(f"  Rest perimeter: {poly.perimeter0():.6f}")
    print(f"  Centroid: {poly.centroid().x:.6f}, {poly.centroid().y:.6f}")
    print(f"  Polarity: ({poly.polarity.x:.6f}, {poly.polarity.y:.6f})")
    print(f"  Num vertices: {poly.num_vertices()}")

    # Print first few vertex positions
    positions = poly.get_vertex_positions()
    print(f"  Vertex positions (first 5):")
    for i, (x, y) in enumerate(positions[:5]):
        print(f"    {i}: ({x:.6f}, {y:.6f})")
    if len(positions) > 5:
        print(f"    ... and {len(positions) - 5} more")


def main():
    """Main entry point."""
    print_openmp_info()

    # Optionally set number of threads (uncomment to use)
    # polyhoop.set_num_threads(4)

    if len(sys.argv) > 1:
        # Run from OFF file
        input_file = sys.argv[1]
        num_frames = int(sys.argv[2]) if len(sys.argv) > 2 else 100
        steps_per_frame = int(sys.argv[3]) if len(sys.argv) > 3 else 100
        output_prefix = sys.argv[4] if len(sys.argv) > 4 else "frame"

        if not os.path.exists(input_file):
            print(f"Error: Input file '{input_file}' not found")
            sys.exit(1)

        ensemble = run_from_file(input_file, num_frames, steps_per_frame, output_prefix)
    else:
        # Run programmatic example
        print("No input file specified. Running programmatic example.")
        print("Usage: python run_polyhoop.py [input.off] [num_frames] [steps_per_frame] [output_prefix]")
        print()
        ensemble = run_programmatic_example()

    # Inspect first polygon
    if ensemble.num_polygons() > 0:
        inspect_polygon(ensemble, 0)


if __name__ == "__main__":
    ensemble = run_programmatic_example(100, 1000, n_cells = 200, pattern="grid")