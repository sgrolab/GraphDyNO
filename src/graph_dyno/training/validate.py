#%%
"""Validation functions for trained models."""

import torch
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Tuple
from omegaconf import DictConfig


def _validate_sitting_cell(
    model,
    idx: int,
    cfg: DictConfig,
    device: str = "cuda",
    n_steps: int = 200,
    initial_position: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """
    Validate that a cell with a given latent embedding stays in place ("sits") over time.

    Internal function used by validate_all_sitting_cells.

    Args:
        model: Trained GraphModel in eval mode
        idx: Index of the particle to validate
        cfg: Hydra DictConfig containing data.dt, data.domain_size, etc.
        device: Device to run on ('cuda' or 'cpu')
        n_steps: Number of simulation steps to run
        initial_position: Starting position (2,). Defaults to random within domain.

    Returns:
        dict with trajectory data and displacement metrics
    """
    model.eval()

    # Extract parameters from config
    dt = cfg.data.dt
    domain_size = cfg.data.domain_size

    # Set up initial conditions
    if initial_position is None:
        initial_position = torch.rand(2, device=device) * domain_size
    else:
        initial_position = initial_position.to(device)

    # Get embeddings (unified model always uses parameter_embedding and shape_embedding)
    b = model.parameter_embedding[idx].unsqueeze(0).detach().to(device)  # (1, embedding_dim)
    # Initialize shape embedding from time 0
    t_idx_zero = torch.tensor([0], device=device)
    L = model.shape_embedding.access(t_idx_zero)[0, idx].unsqueeze(0).detach()  # (1, shape_embedding_dim)

    if b.dim() == 1:
        b = b.unsqueeze(0)  # (1, embedding_dim)

    n_particles = 1

    # Initialize state: [x, y, vx, vy]
    x_init = initial_position.unsqueeze(0)  # (1, 2)
    v_init = torch.zeros_like(x_init)  # (1, 2)
    states = torch.cat([x_init, v_init], dim=-1)  # (1, 4)

    # Storage for trajectory
    positions = [states[:, :2].cpu().numpy()]
    velocities = [states[:, 2:].cpu().numpy()]
    times = [0.0]

    with torch.no_grad():
        for step in range(n_steps):
            # Build interaction matrix (empty for single isolated cell)
            indices = torch.zeros((2, 0), dtype=torch.long, device=device)
            values = torch.zeros(0, dtype=torch.bool, device=device)
            W_interactions = torch.sparse_coo_tensor(
                indices, values, (n_particles, n_particles),
                dtype=torch.bool, device=device
            )

            # Generate next state (unified interface for both modes)
            new_states, image, L_next = model.generate(
                states, W_interactions, b, L
            )
            L = L_next  # Update shape embedding for next step

            states = new_states
            positions.append(states[:, :2].cpu().numpy())
            velocities.append(states[:, 2:].cpu().numpy())
            times.append((step + 1) * dt)

    # Convert to arrays
    positions = np.array(positions).squeeze()  # (n_steps+1, 2)
    velocities = np.array(velocities).squeeze()  # (n_steps+1, 2)
    times = np.array(times)  # (n_steps+1,)

    # Expected position is initial position (cell should sit still)
    initial_pos = positions[0]

    # Compute displacements from expected (initial) position
    displacements = np.linalg.norm(positions - initial_pos, axis=-1)  # (n_steps+1,)

    # RMSE: root mean square error of position vs expected position
    rmse = np.sqrt(np.mean(displacements ** 2))

    return {
        'positions': positions,
        'velocities': velocities,
        'displacements': displacements,
        'times': times,
        'rmse': rmse,
        'final_displacement': displacements[-1],
        'max_displacement': displacements.max(),
    }


def validate_all_sitting_cells(
    model,
    cfg: DictConfig,
    device: str = "cuda",
    n_steps: int = 200,
) -> Tuple[float, pd.DataFrame]:
    """
    Validate all unique combinations of radius and growth rate.

    Use this function during training to validate the model. For post-training
    analysis on saved runs, use `validate_all_sitting_cells_from_run()`.

    Args:
        model: GraphModel (will be set to eval mode).
               If model has 'rs' and 'growth_rates' metadata set, will group by
               unique combinations. Otherwise validates all particles.
        cfg: Hydra DictConfig containing data.dt, data.domain_size, etc.
        device: Device to run on ('cuda' or 'cpu')
        n_steps: Number of simulation steps

    Returns:
        Tuple of:
            - overall_rmse: Overall RMSE across all particles (single summary statistic)
            - summary_df: DataFrame with columns [radius, growth_rate, growth_rate_percentage,
                          rmse, final_displacement, max_displacement, times, displacements]
    """
    # Get number of nodes from the model (unified model always uses parameter_embedding)
    num_nodes = model.parameter_embedding.shape[0]

    # Try to get metadata from model
    if hasattr(model, 'get_metadata'):
        rs = model.get_metadata('rs')
        growth_rates = model.get_metadata('growth_rates')
    else:
        rs = getattr(model, 'rs', None)
        growth_rates = getattr(model, 'growth_rates', None)

    # Build dataframe of all particles
    if rs is not None and growth_rates is not None:
        r_values = rs.cpu().numpy()
        growth_rates_np = growth_rates.cpu().numpy()

        df = pd.DataFrame({
            'particle_index': np.arange(len(r_values)),
            'radius': r_values,
            'growth_rate': growth_rates_np,
        })
        df['growth_rate_percentage'] = (df['growth_rate'] / df['radius']).round(4)

        # Get unique combinations and pick one representative particle for each
        unique_combinations = df.groupby(['radius', 'growth_rate']).first().reset_index()
    else:
        # No radius/growth_rate info - validate all particles
        unique_combinations = pd.DataFrame({
            'particle_index': np.arange(num_nodes),
            'radius': np.nan,
            'growth_rate': np.nan,
            'growth_rate_percentage': np.nan,
        })

    results = []
    all_displacements = []

    for _, row in unique_combinations.iterrows():
        idx = int(row['particle_index'])
        result = _validate_sitting_cell(
            model=model,
            idx=idx,
            cfg=cfg,
            device=device,
            n_steps=n_steps,
        )
        results.append({
            'particle_index': idx,
            'radius': row['radius'],
            'growth_rate': row['growth_rate'],
            'growth_rate_percentage': row['growth_rate_percentage'],
            'rmse': result['rmse'],
            'final_displacement': result['final_displacement'],
            'max_displacement': result['max_displacement'],
            'times': result['times'],
            'displacements': result['displacements'],
        })
        all_displacements.extend(result['displacements'])

    # Build summary dataframe
    summary_df = pd.DataFrame(results)

    # Overall RMSE across all particles and timesteps
    overall_rmse = np.sqrt(np.mean(np.array(all_displacements) ** 2))

    return overall_rmse, summary_df


def validate_all_sitting_cells_from_run(
    run_dict: Dict[str, Any],
    n_steps: int = 200,
) -> Tuple[float, pd.DataFrame]:
    """
    Validate all unique combinations of radius and growth rate from a saved run.

    Convenience wrapper around `validate_all_sitting_cells()` for post-training
    analysis using output from `plot.load_run()`.

    Args:
        run_dict: Output from plot.load_run() containing model, cfg, etc.
        n_steps: Number of simulation steps

    Returns:
        Tuple of:
            - overall_rmse: Overall RMSE across all particles
            - summary_df: DataFrame with validation results per particle
    """
    return validate_all_sitting_cells(
        model=run_dict['model'],
        cfg=run_dict['cfg'],
        device=run_dict['device'],
        n_steps=n_steps,
    )


def plot_sitting_cell_validation(
    summary_df: pd.DataFrame,
    save_path: Optional[str] = None,
    **kwargs
) -> None:
    """
    Plot validation results for sitting cells.

    Args:
        summary_df: DataFrame from validate_all_sitting_cells() with columns
                    [radius, growth_rate, growth_rate_percentage, rmse,
                     final_displacement, max_displacement, times, displacements]
        save_path: Optional path to save the figure. If None, just displays.
    """
    import matplotlib.pyplot as plt

    default_fig_kwargs = {'figsize': (10, 3)}
    default_fig_kwargs.update(kwargs)
    fig, axes = plt.subplots(1, 3, **default_fig_kwargs)

    # Plot displacements over time, colored by growth rate percentage
    ax = axes[0]
    growth_rates_pct = summary_df['growth_rate_percentage'].values
    cmap = plt.cm.coolwarm
    norm = plt.Normalize(vmin=growth_rates_pct.min(), vmax=growth_rates_pct.max())
    for _, row in summary_df.iterrows():
        color = cmap(norm(row['growth_rate_percentage']))
        ax.plot(row['times'], row['displacements'], color=color, alpha=0.7)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    plt.colorbar(sm, ax=ax, label='Growth rate %')
    ax.set_xlabel('Time')
    ax.set_ylabel('Displacement from initial position')
    ax.set_title('Cell displacement over time')

    # Plot RMSE vs growth rate percentage
    ax = axes[1]
    scatter = ax.scatter(
        summary_df['growth_rate_percentage'],
        summary_df['rmse'],
        c=summary_df['radius'],
        cmap='viridis',
        s=50,
        edgecolors='k',
        linewidths=0.5
    )
    plt.colorbar(scatter, ax=ax, label='Radius')
    ax.set_xlabel('Growth rate percentage')
    ax.set_ylabel('RMSE')
    ax.set_title('RMSE vs growth rate')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)

    # Plot RMSE vs radius
    ax = axes[2]
    scatter = ax.scatter(
        summary_df['radius'],
        summary_df['rmse'],
        c=summary_df['growth_rate_percentage'],
        cmap='coolwarm',
        s=50,
        edgecolors='k',
        linewidths=0.5
    )
    plt.colorbar(scatter, ax=ax, label='Growth rate %')
    ax.set_xlabel('Radius')
    ax.set_ylabel('RMSE')
    ax.set_title('RMSE vs radius')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved to: {save_path}")


if __name__ == "__main__":
    from graph_dyno.plotting.plot import load_run

    # Load most recent run
    run = load_run('awesome-sloth-330')

    # Validate all unique combinations of radius and growth rate
    overall_rmse, summary_df = validate_all_sitting_cells_from_run(run, n_steps=200)

    print(f"\nValidation Results:")
    print(f"  Number of unique (radius, growth_rate) combinations: {len(summary_df)}")
    print(f"  Overall RMSE: {overall_rmse:.6f}")
    print(f"\nPer-particle summary:")
    print(summary_df[['radius', 'growth_rate_percentage', 'rmse', 'final_displacement']].to_string())

    # Plot results
    plot_sitting_cell_validation(
        summary_df,
        save_path=run['save_path'] / 'sitting_cell_validation.png'
    )
# %%
