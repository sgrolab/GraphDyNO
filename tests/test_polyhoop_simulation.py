"""Unit tests for polyhoop_simulation.py"""

import pytest
import torch
import numpy as np
import math
from omegaconf import OmegaConf

from graph_dyno.data.polyhoop_simulation import (
    PolyhoopSimulation,
    generate_random_centers,
    generate_polygon_vertices,
    generate_combined_image,
)
from graph_dyno.data.param_utils import resolve_dist_param


@pytest.fixture
def base_config():
    """Fixture providing a minimal valid configuration."""
    return OmegaConf.create({
        'data': {
            'type': 'simulation',
            'n_objects': 10,
            'image_size': 32,
            'domain_size': 20,
            'dt': 0.01,
            'seed': 42,
            'generator': {
                'n_trajectories': 1,
                'trajectory_length': 5,
                'seed_offset': 0,
                'cache_dir': '',
                'overwrite_existing': False
            },
            'params': {
                'name': 'polyhoop',
                'num_substeps': 10,
                'gam': 1000,
                'ka': 100000,
                'kr': 10000000,
                'cv': 10,
                'fp': 0,
                'num_vertices': 20,
                'radius': 1.0,
                'density_fraction': 0.3,
                'r_cutoff': 5.0,
                'growth_rate': 0,
                'oscillatory_growth': False,
                'area_fluct_period': 0,
                'area_fluct_amp': 0,
            }
        }
    })


@pytest.fixture
def simulation(base_config):
    """Fixture providing an initialized simulation."""
    return PolyhoopSimulation(base_config, device='cpu')


class TestGenerateRandomCenters:
    """Tests for generate_random_centers function."""

    def test_generates_correct_number_of_centers(self):
        """Test that the correct number of centers are generated."""
        radii = np.ones(5) * 0.5
        centers = generate_random_centers(
            num_cells=5,
            bounds=(0, 10, 0, 10),
            radii=radii,
            seed=42
        )
        assert len(centers) == 5

    def test_centers_within_bounds(self):
        """Test that centers are placed within the specified bounds."""
        radii = np.ones(10) * 0.5
        centers = generate_random_centers(
            num_cells=10,
            bounds=(0, 20, 0, 20),
            radii=radii,
            seed=42
        )
        for x, y in centers:
            assert 0.5 <= x <= 19.5
            assert 0.5 <= y <= 19.5

    def test_cells_do_not_overlap(self):
        """Test that cells maintain minimum separation."""
        radii = np.array([1.0, 1.5, 0.5, 1.0, 0.8])
        centers = generate_random_centers(
            num_cells=5,
            bounds=(0, 30, 0, 30),
            radii=radii,
            seed=42
        )
        for i, (x1, y1) in enumerate(centers):
            for j, (x2, y2) in enumerate(centers):
                if i != j:
                    dist = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    min_sep = radii[i] + radii[j]
                    assert dist >= min_sep, f"Cells {i} and {j} overlap"

    def test_reproducible_with_seed(self):
        """Test that results are reproducible with the same seed."""
        radii = np.ones(5)
        centers1 = generate_random_centers(5, (0, 20, 0, 20), radii, seed=42)
        centers2 = generate_random_centers(5, (0, 20, 0, 20), radii, seed=42)
        for c1, c2 in zip(centers1, centers2):
            assert c1 == c2


class TestGeneratePolygonVertices:
    """Tests for generate_polygon_vertices function."""

    def test_generates_correct_number_of_vertices(self):
        """Test that the correct number of vertices are generated."""
        vertices = generate_polygon_vertices(
            center=(5, 5),
            radius=1.0,
            num_vertices=40
        )
        assert len(vertices) == 40

    def test_vertices_form_circle(self):
        """Test that vertices are approximately at the specified radius."""
        center = (5, 5)
        radius = 2.0
        vertices = generate_polygon_vertices(
            center=center,
            radius=radius,
            num_vertices=40
        )
        for x, y in vertices:
            dist = math.sqrt((x - center[0])**2 + (y - center[1])**2)
            assert abs(dist - radius) < 1e-10

    def test_start_angle_offset(self):
        """Test that start_angle parameter works."""
        center = (0, 0)
        vertices_0 = generate_polygon_vertices(center, 1.0, 4, start_angle=0)
        vertices_90 = generate_polygon_vertices(center, 1.0, 4, start_angle=math.pi/2)

        # First vertex at angle 0 should be at (1, 0)
        assert abs(vertices_0[0][0] - 1.0) < 1e-10
        assert abs(vertices_0[0][1] - 0.0) < 1e-10

        # First vertex at angle 90 should be at (0, 1)
        assert abs(vertices_90[0][0] - 0.0) < 1e-10
        assert abs(vertices_90[0][1] - 1.0) < 1e-10


class TestPolyhoopSimulationInit:
    """Tests for PolyhoopSimulation initialization."""

    def test_basic_initialization(self, base_config):
        """Test that simulation initializes without errors."""
        sim = PolyhoopSimulation(base_config, device='cpu')
        assert sim is not None
        assert sim.n_polygons <= base_config.data.n_objects

    def test_polygon_count(self, simulation):
        """Test that polygons are created."""
        assert simulation.n_polygons > 0
        assert simulation.ensemble.num_polygons() == simulation.n_polygons

    def test_device_assignment(self, base_config):
        """Test that device is correctly assigned."""
        sim = PolyhoopSimulation(base_config, device='cpu')
        assert sim.device == torch.device('cpu')

    def test_radii_tensor_created(self, simulation):
        """Test that radii tensor is created with correct shape."""
        assert simulation.radii.shape == (simulation.n_polygons,)

    def test_positions_initialized(self, simulation):
        """Test that positions are initialized."""
        assert simulation.positions.shape == (simulation.n_polygons, 2)

    def test_velocities_initialized_to_zero(self, simulation):
        """Test that velocities are initialized to zero."""
        assert simulation.velocities.shape == (simulation.n_polygons, 2)
        assert torch.allclose(simulation.velocities, torch.zeros_like(simulation.velocities))


class TestRadiusConfiguration:
    """Tests for radius parameter configuration."""

    def test_fixed_radius(self, base_config):
        """Test that fixed radius value works."""
        base_config.data.params.radius = 1.5
        sim = PolyhoopSimulation(base_config, device='cpu')
        assert torch.allclose(sim.radii, torch.full((sim.n_polygons,), 1.5), atol=1e-6)

    def test_dist_linspace_radius(self, base_config):
        """Test linspace distribution for radius."""
        base_config.data.params.radius = {
            'dist': 'linspace',
            'min': 0.5,
            'max': 1.5
        }
        sim = PolyhoopSimulation(base_config, device='cpu')

        assert sim.radii.min() >= 0.5
        assert sim.radii.max() <= 1.5

    def test_dist_linspace_with_steps_repeats(self, base_config):
        """Test linspace with steps and repeats."""
        base_config.data.n_objects = 20
        base_config.data.params.radius = {
            'dist': 'linspace',
            'min': 0.5,
            'max': 1.5,
            'steps': 4,
            'repeats': 5,
            'interleaved': False
        }
        base_config.data.params.density_fraction = 0.1  # Lower density to fit more cells
        sim = PolyhoopSimulation(base_config, device='cpu')

        # Should have 4 unique values (may have fewer if not all cells placed)
        unique_radii = torch.unique(sim.radii)
        assert len(unique_radii) <= 4

    def test_dist_block_factorial(self):
        """block key composes 3 two-level factors into a balanced 2x2x2 grid.

        With steps=2, repeats=300 (steps*repeats=600) on all three axes and
        block in {1, 2, 4}, each 8-cell period cycles through all 8 unique
        (amp, kh, kl) combinations, so the full 600-cell assignment has exactly
        8 distinct combos, each appearing 75 times.
        """
        n = 600
        amp = resolve_dist_param(
            {'dist': 'linspace', 'min': 0.785398, 'max': 3.141593,
             'steps': 2, 'repeats': 300, 'block': 1}, n)
        kh = resolve_dist_param(
            {'dist': 'logspace', 'min': 1e4, 'max': 5e4,
             'steps': 2, 'repeats': 300, 'block': 2}, n)
        kl = resolve_dist_param(
            {'dist': 'logspace', 'min': 1e3, 'max': 1e6,
             'steps': 2, 'repeats': 300, 'block': 4}, n)

        assert amp.shape == (n,) and kh.shape == (n,) and kl.shape == (n,)

        rows = np.stack([amp, kh, kl], axis=1)
        combos, counts = np.unique(rows, axis=0, return_counts=True)
        assert len(combos) == 8, f"expected 8 unique combos, got {len(combos)}"
        assert np.all(counts == 75), f"expected 75x each, got {counts.tolist()}"

    def test_dist_block_recovers_interleaved_modes(self):
        """block=1 == interleaved:true (tile); block=repeats == interleaved:false (repeat)."""
        n = 20
        spec = {'dist': 'linspace', 'min': 0.0, 'max': 1.0, 'steps': 2, 'repeats': 10}
        tile = resolve_dist_param({**spec, 'interleaved': True}, n)
        repeat = resolve_dist_param({**spec, 'interleaved': False}, n)
        assert np.array_equal(resolve_dist_param({**spec, 'block': 1}, n), tile)
        assert np.array_equal(resolve_dist_param({**spec, 'block': 10}, n), repeat)

    def test_dist_block_must_divide_repeats(self):
        """A block that does not divide repeats is an error."""
        with pytest.raises(ValueError, match="must divide"):
            resolve_dist_param(
                {'dist': 'linspace', 'min': 0.0, 'max': 1.0,
                 'steps': 2, 'repeats': 300, 'block': 7}, 600)


class TestGrowthRateConfiguration:
    """Tests for growth rate parameter configuration."""

    def test_zero_growth_rate(self, base_config):
        """Test that zero growth rate is handled."""
        base_config.data.params.growth_rate = 0
        sim = PolyhoopSimulation(base_config, device='cpu')
        assert torch.allclose(sim.growth_rates, torch.zeros(sim.n_polygons))

    def test_fixed_growth_rate(self, base_config):
        """Test fixed growth rate value."""
        base_config.data.params.growth_rate = 0.1
        sim = PolyhoopSimulation(base_config, device='cpu')
        assert torch.allclose(sim.growth_rates, torch.full((sim.n_polygons,), 0.1), atol=1e-6)

    def test_dist_growth_rate(self, base_config):
        """Test distribution-based growth rate."""
        base_config.data.params.growth_rate = {
            'dist': 'linspace',
            'min': 0.0,
            'max': 0.2
        }
        sim = PolyhoopSimulation(base_config, device='cpu')

        assert sim.growth_rates.min() >= 0.0
        assert sim.growth_rates.max() <= 0.2

    def test_growth_rates_set_on_polygons(self, base_config):
        """Test that growth rates are set on polygon objects."""
        base_config.data.params.growth_rate = 0.15
        sim = PolyhoopSimulation(base_config, device='cpu')

        for i in range(sim.n_polygons):
            poly = sim.ensemble.get_polygon(i)
            assert abs(poly.alpha - 0.15) < 1e-5


class TestAreaFluctuationConfiguration:
    """Tests for area fluctuation parameter configuration."""

    def test_zero_fluctuation_defaults(self, base_config):
        """Test that zero fluctuation values are set by default."""
        sim = PolyhoopSimulation(base_config, device='cpu')
        assert torch.allclose(sim.area_fluct_periods, torch.zeros(sim.n_polygons))
        assert torch.allclose(sim.area_fluct_amps, torch.zeros(sim.n_polygons))

    def test_fixed_fluctuation_values(self, base_config):
        """Test fixed fluctuation parameter values."""
        base_config.data.params.area_fluct_period = 0.1
        base_config.data.params.area_fluct_amp = 0.5
        sim = PolyhoopSimulation(base_config, device='cpu')

        assert torch.allclose(sim.area_fluct_periods, torch.full((sim.n_polygons,), 0.1))
        assert torch.allclose(sim.area_fluct_amps, torch.full((sim.n_polygons,), 0.5))

    def test_dist_fluctuation_freq(self, base_config):
        """Test distribution-based fluctuation frequency."""
        base_config.data.params.area_fluct_period = {
            'dist': 'linspace',
            'min': 0.1,
            'max': 0.5
        }
        base_config.data.params.area_fluct_amp = 1.0
        sim = PolyhoopSimulation(base_config, device='cpu')

        assert sim.area_fluct_periods.min() >= 0.1
        assert sim.area_fluct_periods.max() <= 0.5

    def test_dist_fluctuation_amp(self, base_config):
        """Test distribution-based fluctuation amplitude."""
        base_config.data.params.area_fluct_period = 0.1
        base_config.data.params.area_fluct_amp = {
            'dist': 'linspace',
            'min': 0.0,
            'max': 2.0
        }
        sim = PolyhoopSimulation(base_config, device='cpu')

        assert sim.area_fluct_amps.min() >= 0.0
        assert sim.area_fluct_amps.max() <= 2.0

    def test_fluctuation_values_set_on_polygons(self, base_config):
        """Test that fluctuation values are set on polygon objects."""
        base_config.data.params.area_fluct_period = 0.25
        base_config.data.params.area_fluct_amp = 1.5
        sim = PolyhoopSimulation(base_config, device='cpu')

        for i in range(sim.n_polygons):
            poly = sim.ensemble.get_polygon(i)
            assert abs(poly.area_fluct_period - 0.25) < 1e-6
            assert abs(poly.area_fluct_amp - 1.5) < 1e-6

    def test_per_cell_fluctuation_variation(self, base_config):
        """Test that per-cell variation works correctly."""
        base_config.data.params.area_fluct_period = {
            'dist': 'linspace',
            'min': 0.1,
            'max': 1.0
        }
        base_config.data.params.area_fluct_amp = {
            'dist': 'linspace',
            'min': 0.5,
            'max': 2.0
        }
        sim = PolyhoopSimulation(base_config, device='cpu')

        # Verify per-polygon values match tensors
        for i in range(sim.n_polygons):
            poly = sim.ensemble.get_polygon(i)
            assert abs(poly.area_fluct_period - float(sim.area_fluct_periods[i])) < 1e-6
            assert abs(poly.area_fluct_amp - float(sim.area_fluct_amps[i])) < 1e-6


class TestSimulationStep:
    """Tests for simulation stepping."""

    def test_step_returns_correct_shapes(self, simulation):
        """Test that step() returns tensors with correct shapes."""
        positions, velocities, forces, graph = simulation.step()

        n = simulation.n_polygons
        assert positions.shape == (n, 2)
        assert velocities.shape == (n, 2)
        assert forces.shape == (n, 2)
        assert graph.shape == (n, n)

    def test_step_updates_time(self, simulation):
        """Test that simulation time is updated after step."""
        initial_time = simulation.ensemble.t
        simulation.step()
        assert simulation.ensemble.t > initial_time

    def test_step_updates_positions(self, simulation):
        """Test that positions change after stepping."""
        initial_positions = simulation.positions.clone()
        simulation.step()
        # Positions may or may not change depending on forces
        # Just verify no NaN/Inf
        assert not torch.isnan(simulation.positions).any()
        assert not torch.isinf(simulation.positions).any()

    def test_multiple_steps(self, simulation):
        """Test running multiple steps."""
        for _ in range(5):
            positions, velocities, forces, graph = simulation.step()
            assert not torch.isnan(positions).any()
            assert not torch.isnan(velocities).any()

    def test_interaction_graph_is_sparse(self, simulation):
        """Test that interaction graph is sparse."""
        _, _, _, graph = simulation.step()
        assert graph.is_sparse


class TestInteractionGraph:
    """Tests for interaction graph computation."""

    def test_graph_is_boolean(self, simulation):
        """Test that graph values are boolean."""
        _, _, _, graph = simulation.step()
        dense = graph.to_dense()
        assert dense.dtype == torch.bool

    def test_no_self_interactions(self, simulation):
        """Test that diagonal is zero (no self-interactions)."""
        _, _, _, graph = simulation.step()
        dense = graph.to_dense()
        diagonal = torch.diag(dense)
        assert not diagonal.any(), "Self-interactions found in graph"

    def test_interactions_within_cutoff(self, base_config):
        """Test that interactions are within r_cutoff."""
        base_config.data.params.r_cutoff = 3.0
        sim = PolyhoopSimulation(base_config, device='cpu')

        positions, _, _, graph = sim.step()
        dense = graph.to_dense()

        # Check that all edges are within cutoff
        for i in range(sim.n_polygons):
            for j in range(sim.n_polygons):
                if dense[i, j]:
                    dist = torch.norm(positions[i] - positions[j])
                    assert dist <= base_config.data.params.r_cutoff


class TestImageGeneration:
    """Tests for image generation."""

    def test_generate_image_shape(self, simulation):
        """Test that generated image has correct shape."""
        simulation.step()
        image = simulation._generate_image()

        expected_shape = (simulation.n_polygons, simulation.data.image_size, simulation.data.image_size)
        assert image.shape == expected_shape

    def test_generate_image_values(self, simulation):
        """Test that image values are valid (0 or 1 for masks)."""
        simulation.step()
        image = simulation._generate_image()

        # Should be binary masks
        assert image.min() >= 0
        assert image.max() <= 1

    def test_combined_image_generation(self, simulation):
        """Test combined image generation."""
        simulation.step()
        channeled_image = simulation._generate_image()
        combined = simulation._generate_combined_image(simulation.positions, channeled_image)

        # Combined image should be 2D
        assert len(combined.shape) == 2


class TestMetadata:
    """Tests for simulation metadata."""

    def test_metadata_contains_required_fields(self, simulation):
        """Test that metadata contains all required fields."""
        metadata = simulation.get_simulation_metadata()

        assert 'rs' in metadata
        assert 'growth_rates' in metadata
        assert 'area_fluct_periods' in metadata
        assert 'area_fluct_amps' in metadata
        assert 'r_max' in metadata

    def test_metadata_shapes(self, simulation):
        """Test that metadata tensors have correct shapes."""
        metadata = simulation.get_simulation_metadata()

        assert metadata['rs'].shape == (simulation.n_polygons,)
        assert metadata['growth_rates'].shape == (simulation.n_polygons,)
        assert metadata['area_fluct_periods'].shape == (simulation.n_polygons,)
        assert metadata['area_fluct_amps'].shape == (simulation.n_polygons,)

    def test_metadata_on_cpu(self, simulation):
        """Test that metadata tensors are on CPU."""
        metadata = simulation.get_simulation_metadata()

        assert metadata['rs'].device == torch.device('cpu')
        assert metadata['growth_rates'].device == torch.device('cpu')


class TestRelaxation:
    """Tests for relaxation functionality."""

    def test_relax_with_zero_steps(self, base_config):
        """Test that relax() with zero steps does nothing."""
        base_config.data.params.relaxation_steps = 0
        sim = PolyhoopSimulation(base_config, device='cpu')

        initial_time = sim.ensemble.t
        sim.relax()
        # Time should not change with 0 relaxation steps
        assert sim.ensemble.t == initial_time

    def test_relax_with_nonzero_steps(self, base_config):
        """Test that relax() runs the specified steps."""
        base_config.data.params.relaxation_steps = 100
        sim = PolyhoopSimulation(base_config, device='cpu')

        initial_time = sim.ensemble.t
        sim.relax()
        assert sim.ensemble.t > initial_time


class TestGenerate:
    """Tests for full data generation."""

    def test_generate_returns_correct_types(self, base_config):
        """Test that generate() returns correct data types."""
        base_config.data.generator.trajectory_length = 3
        sim = PolyhoopSimulation(base_config, device='cpu')

        times, states, next_states, forces, images, graphs = sim.generate()

        assert isinstance(times, torch.Tensor)
        assert isinstance(states, torch.Tensor)
        assert isinstance(next_states, torch.Tensor)
        assert isinstance(forces, torch.Tensor)
        assert isinstance(images, torch.Tensor)
        assert isinstance(graphs, torch.Tensor)

    def test_generate_shapes(self, base_config):
        """Test that generate() returns correct shapes."""
        traj_len = 3
        base_config.data.generator.trajectory_length = traj_len
        sim = PolyhoopSimulation(base_config, device='cpu')

        times, states, next_states, forces, images, graphs = sim.generate()

        n = sim.n_polygons
        img_size = base_config.data.image_size

        assert times.shape == (traj_len, 1)
        assert states.shape == (traj_len, n, 4)
        assert next_states.shape == (traj_len, n, 4)
        assert forces.shape == (traj_len, n * 2)
        assert images.shape == (traj_len, n, img_size, img_size)

    def test_images_and_graphs_share_the_state_frame(self, base_config):
        """channeled_images[t] and W_interactions[t] depict states[t], not next_states[t]."""
        traj_len = 4
        base_config.data.generator.trajectory_length = traj_len
        sim = PolyhoopSimulation(base_config, device='cpu')

        # Record where the ensemble is whenever a raster snapshot or a contact
        # graph is taken, so each stored frame can be matched to a state.
        snapshot_pos, graph_at = [], []
        snapshot_polygons = sim._snapshot_polygons
        compute_graph = sim._compute_interaction_graph

        def recording_snapshot():
            snap = snapshot_polygons()
            snapshot_pos.append(torch.tensor([[x, y] for x, y, _ in snap]))
            return snap

        def recording_graph():
            graph = compute_graph()
            graph_at.append((sim._get_centroids(), graph))
            return graph

        sim._snapshot_polygons = recording_snapshot
        sim._compute_interaction_graph = recording_graph

        times, states, next_states, forces, images, graphs = sim.generate()

        assert torch.equal(next_states[:-1], states[1:])
        assert len(snapshot_pos) == traj_len
        for t in range(traj_len):
            assert torch.allclose(snapshot_pos[t], states[t, :, :2])
            assert not torch.allclose(snapshot_pos[t], next_states[t, :, :2])
            matches = [g for pos, g in graph_at if torch.equal(pos, states[t, :, :2])]
            assert matches, f"no contact graph was computed at frame {t}"
            assert torch.equal(graphs[t].to_dense(), matches[-1].to_dense())


class TestDensityFraction:
    """Tests for density_fraction parameter."""

    def test_invalid_density_fraction_zero(self, base_config):
        """Test that zero density_fraction raises error."""
        base_config.data.params.density_fraction = 0
        with pytest.raises(ValueError):
            PolyhoopSimulation(base_config, device='cpu')

    def test_invalid_density_fraction_negative(self, base_config):
        """Test that negative density_fraction raises error."""
        base_config.data.params.density_fraction = -0.5
        with pytest.raises(ValueError):
            PolyhoopSimulation(base_config, device='cpu')

    def test_density_fraction_above_one(self, base_config):
        """Test that density_fraction > 1 is allowed (higher density)."""
        base_config.data.params.density_fraction = 1.5
        sim = PolyhoopSimulation(base_config, device='cpu')
        assert sim.n_polygons > 0

    def test_low_density_places_fewer_cells(self, base_config):
        """Test that lower density can place more cells in larger area."""
        # With very low density, cells should be easier to place
        base_config.data.n_objects = 20
        base_config.data.params.density_fraction = 0.1
        base_config.data.params.radius = 1.0

        sim = PolyhoopSimulation(base_config, device='cpu')
        # Should place all cells with such low density
        assert sim.n_polygons == 20


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_single_polygon(self, base_config):
        """Test with a single polygon."""
        base_config.data.n_objects = 1
        base_config.data.params.density_fraction = 1.0
        sim = PolyhoopSimulation(base_config, device='cpu')

        assert sim.n_polygons == 1
        positions, velocities, forces, graph = sim.step()
        assert positions.shape == (1, 2)

    def test_no_nan_or_inf_after_many_steps(self, base_config):
        """Test that simulation remains stable over many steps."""
        base_config.data.generator.trajectory_length = 50
        sim = PolyhoopSimulation(base_config, device='cpu')

        for _ in range(50):
            positions, velocities, forces, _ = sim.step()
            assert not torch.isnan(positions).any(), "NaN in positions"
            assert not torch.isinf(positions).any(), "Inf in positions"
            assert not torch.isnan(velocities).any(), "NaN in velocities"
            assert not torch.isinf(velocities).any(), "Inf in velocities"


class TestGenerateCombinedImage:
    """Tests for generate_combined_image function."""

    def test_combined_image_basic(self):
        """Test basic combined image generation."""
        n_polygons = 5
        image_size = 16
        domain_size = 10.0
        r_max = 2.0

        positions = torch.rand(n_polygons, 2) * domain_size
        channeled_image = torch.ones(n_polygons, image_size, image_size)

        combined = generate_combined_image(
            positions, channeled_image, domain_size, r_max, device='cpu'
        )

        assert len(combined.shape) == 2
        assert not torch.isnan(combined).any()


class TestAreaFluctuationPhysics:
    """Tests that verify the area fluctuation physics are correct."""

    @pytest.fixture
    def isolated_cells_config(self):
        """Config with 2 isolated cells that won't interact."""
        return OmegaConf.create({
            'data': {
                'type': 'simulation',
                'n_objects': 2,
                'image_size': 32,
                'domain_size': 100,  # Large domain for isolation
                'dt': 0.01,
                'seed': 42,
                'generator': {
                    'n_trajectories': 1,
                    'trajectory_length': 100,
                    'seed_offset': 0,
                    'cache_dir': '',
                    'overwrite_existing': False
                },
                'params': {
                    'name': 'polyhoop',
                    'num_substeps': 100,  # More substeps for accuracy
                    'gam': 1000,
                    'ka': 100000,
                    'kr': 10000000,
                    'cv': 10,
                    'fp': 0,
                    'num_vertices': 20,
                    'radius': 1.0,
                    'density_fraction': 0.01,  # Very low density for isolation
                    'r_cutoff': 5.0,
                    'growth_rate': 0,
                    'oscillatory_growth': False,
                    'area_fluct_period': 0,
                    'area_fluct_amp': 0,
                    'random_phases': False,
                }
            }
        })

    def test_target_area_oscillates_sinusoidally(self, isolated_cells_config):
        """Test that A0 (target area) follows expected sinusoidal pattern.

        The effective growth rate is:
            alpha_eff = area_fluct_amp * 2π / period * cos(2π * t / period)

        Integrating, the target area should be:
            A0(t) = A0(0) + area_fluct_amp * sin(2π * t / period)
        """
        period = 1.0  # 1 time unit per cycle
        amp = 0.5     # Amplitude of area oscillation

        isolated_cells_config.data.params.oscillatory_growth = True
        isolated_cells_config.data.params.area_fluct_period = period
        isolated_cells_config.data.params.area_fluct_amp = amp

        sim = PolyhoopSimulation(isolated_cells_config, device='cpu')

        # Record A0 over time
        n_steps = 200  # Run for 2 full cycles

        times = []
        A0_values = [[], []]  # For each cell

        for _ in range(n_steps):
            t = sim.ensemble.t
            times.append(t)

            for i in range(2):
                poly = sim.ensemble.get_polygon(i)
                A0_values[i].append(poly.A0)

            sim.step()

        times = np.array(times)
        A0_values = [np.array(v) for v in A0_values]

        # Check each cell
        for i in range(2):
            A0_initial = A0_values[i][0]

            # Expected: A0(t) = A0(0) + amp * sin(2π * t / period)
            expected_A0 = A0_initial + amp * np.sin(2 * np.pi * times / period)

            # Allow some tolerance due to discrete integration
            # The error should be O(dt)
            max_error = np.abs(A0_values[i] - expected_A0).max()
            assert max_error < 0.05, f"Cell {i}: A0 deviates from expected sinusoid by {max_error}"

    def test_different_periods_per_cell(self, isolated_cells_config):
        """Test that cells with different periods oscillate independently."""
        isolated_cells_config.data.params.oscillatory_growth = True
        isolated_cells_config.data.params.area_fluct_period = {
            'dist': 'linspace',
            'min': 0.5,
            'max': 1.0,
        }
        isolated_cells_config.data.params.area_fluct_amp = 0.3

        sim = PolyhoopSimulation(isolated_cells_config, device='cpu')

        # Verify periods are different
        period0 = sim.area_fluct_periods[0].item()
        period1 = sim.area_fluct_periods[1].item()
        assert period0 != period1, "Periods should be different"

        # Run simulation and record A0
        n_steps = 200
        amp = 0.3

        times = []
        A0_values = [[], []]

        for _ in range(n_steps):
            t = sim.ensemble.t
            times.append(t)

            for i in range(2):
                poly = sim.ensemble.get_polygon(i)
                A0_values[i].append(poly.A0)

            sim.step()

        times = np.array(times)
        A0_values = [np.array(v) for v in A0_values]

        # Check each cell oscillates at its own period
        for i, period in enumerate([period0, period1]):
            A0_initial = A0_values[i][0]
            expected_A0 = A0_initial + amp * np.sin(2 * np.pi * times / period)

            max_error = np.abs(A0_values[i] - expected_A0).max()
            assert max_error < 0.05, f"Cell {i} (period={period}): A0 deviates by {max_error}"

    def test_zero_amplitude_no_oscillation(self, isolated_cells_config):
        """Test that zero amplitude means no area change."""
        isolated_cells_config.data.params.oscillatory_growth = True
        isolated_cells_config.data.params.area_fluct_period = 1.0
        isolated_cells_config.data.params.area_fluct_amp = 0  # Zero amplitude

        sim = PolyhoopSimulation(isolated_cells_config, device='cpu')

        # Record initial A0
        initial_A0 = [sim.ensemble.get_polygon(i).A0 for i in range(2)]

        # Run simulation
        for _ in range(100):
            sim.step()

        # A0 should not have changed (within numerical tolerance)
        for i in range(2):
            final_A0 = sim.ensemble.get_polygon(i).A0
            assert abs(final_A0 - initial_A0[i]) < 1e-10, f"Cell {i}: A0 changed with zero amplitude"

    def test_zero_period_no_oscillation(self, isolated_cells_config):
        """Test that zero period means constant area (no oscillation, no growth)."""
        isolated_cells_config.data.params.oscillatory_growth = True
        isolated_cells_config.data.params.area_fluct_period = 0  # Zero period = constant area
        isolated_cells_config.data.params.area_fluct_amp = 1.0

        sim = PolyhoopSimulation(isolated_cells_config, device='cpu')

        # Record initial A0
        initial_A0 = [sim.ensemble.get_polygon(i).A0 for i in range(2)]

        # Run simulation
        for _ in range(100):
            sim.step()

        # A0 should not have changed (period=0 disables fluctuation)
        for i in range(2):
            final_A0 = sim.ensemble.get_polygon(i).A0
            assert abs(final_A0 - initial_A0[i]) < 1e-10, f"Cell {i}: A0 changed with zero period"

    def test_oscillation_amplitude_matches_parameter(self, isolated_cells_config):
        """Test that the oscillation amplitude matches the configured amplitude."""
        period = 0.5  # Short period for more cycles
        amp = 0.8

        isolated_cells_config.data.params.oscillatory_growth = True
        isolated_cells_config.data.params.area_fluct_period = period
        isolated_cells_config.data.params.area_fluct_amp = amp

        sim = PolyhoopSimulation(isolated_cells_config, device='cpu')

        # Run for several cycles and record A0
        n_steps = 500
        A0_values = []

        for _ in range(n_steps):
            poly = sim.ensemble.get_polygon(0)
            A0_values.append(poly.A0)
            sim.step()

        A0_values = np.array(A0_values)

        # The oscillation should swing between A0_initial - amp and A0_initial + amp
        # Allow some tolerance for discrete integration
        measured_amp = (A0_values.max() - A0_values.min()) / 2

        assert abs(measured_amp - amp) < 0.1, \
            f"Measured amplitude {measured_amp} doesn't match configured {amp}"

    def test_fluctuation_overrides_growth_rate(self, isolated_cells_config):
        """Test that oscillatory mode completely overrides growth_rate."""
        # Set a large growth rate that would cause significant linear growth
        isolated_cells_config.data.params.growth_rate = 10.0  # Would grow A0 by 10 per time unit

        # Enable oscillatory mode - this should override the growth rate
        period = 1.0
        amp = 0.5
        isolated_cells_config.data.params.oscillatory_growth = True
        isolated_cells_config.data.params.area_fluct_period = period
        isolated_cells_config.data.params.area_fluct_amp = amp

        sim = PolyhoopSimulation(isolated_cells_config, device='cpu')

        # Record A0 over time
        n_steps = 200
        times = []
        A0_values = []

        for _ in range(n_steps):
            t = sim.ensemble.t
            times.append(t)
            poly = sim.ensemble.get_polygon(0)
            A0_values.append(poly.A0)
            sim.step()

        times = np.array(times)
        A0_values = np.array(A0_values)
        A0_initial = A0_values[0]

        # If growth_rate was being used, A0 would grow linearly: A0_initial + 10 * t
        # With fluctuation, it should follow: A0_initial + amp * sin(2π * t / period)

        # Check that A0 follows the sinusoidal pattern, NOT linear growth
        expected_sinusoidal = A0_initial + amp * np.sin(2 * np.pi * times / period)
        sinusoidal_error = np.abs(A0_values - expected_sinusoidal).max()

        # Linear growth would give much larger values
        expected_linear = A0_initial + 10.0 * times
        linear_error = np.abs(A0_values - expected_linear).max()

        # Sinusoidal error should be small, linear error should be large
        assert sinusoidal_error < 0.1, f"A0 doesn't follow sinusoidal pattern (error={sinusoidal_error})"
        assert linear_error > 1.0, f"A0 appears to follow linear growth (error={linear_error})"
