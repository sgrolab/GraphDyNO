#%%
import matplotlib.animation as animation
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from graph_dyno.models.model_factory import create_model
from graph_dyno.models.graph_model_dino import GraphModelDINO
from tqdm import tqdm
import matplotlib
import seaborn as sns
from pathlib import Path
from omegaconf import OmegaConf
from scipy.ndimage import shift as ndimage_shift
import mlflow


def generate_combined_image_dino(positions, channeled_image, domain_size, r_max, device=None):
    """Assemble a combined image by placing each shape's centroid at its position.

    Unlike the simulation-specific generate_combined_image, this finds each
    shape's center-of-mass in the channeled image and aligns *that* to the
    target position with subpixel interpolation, avoiding double-discretization
    artifacts.

    Args:
        positions: (n, 2) polygon/sphere centroid positions
        channeled_image: (n, image_size, image_size) per-object images
        domain_size: physical domain size
        r_max: max radius used when rasterizing the channeled images
        device: output device
    Returns:
        combined_image: (combined_size, combined_size) assembled image
    """
    if device is None:
        device = channeled_image.device

    n = positions.shape[0]
    w = channeled_image.shape[1]
    dx = 2 * r_max / (w - 1)
    combined_size = int(domain_size / dx)
    combined_image = torch.zeros((combined_size, combined_size), device=device)
    grid_center = (w - 1) / 2.0
    half_w = w // 2

    for i in range(n):
        img_np = channeled_image[i].cpu().numpy()

        # Find shape centroid (center of mass of nonzero pixels)
        rows, cols = np.where(img_np > 0.5)
        if len(rows) > 0:
            cx = rows.mean()
            cy = cols.mean()
        else:
            cx, cy = grid_center, grid_center

        # Offset of centroid from image grid center
        offset_x = cx - grid_center
        offset_y = cy - grid_center

        # Where the image center should go so that the *centroid* lands at pos
        exact_x = positions[i, 0].item() / dx - offset_x
        exact_y = positions[i, 1].item() / dx - offset_y

        center_x = int(exact_x)
        center_y = int(exact_y)
        sub_x = exact_x - center_x
        sub_y = exact_y - center_y

        x_start = max(center_x - half_w, 0)
        x_end = min(center_x + half_w, combined_size)
        y_start = max(center_y - half_w, 0)
        y_end = min(center_y + half_w, combined_size)

        ch_x_start = max(half_w - center_x, 0)
        ch_x_end = w - max(center_x + half_w - combined_size, 0)
        ch_y_start = max(half_w - center_y, 0)
        ch_y_end = w - max(center_y + half_w - combined_size, 0)

        if x_start < x_end and y_start < y_end:
            shifted = ndimage_shift(img_np, (sub_x, sub_y), order=1)
            shifted = torch.tensor(shifted, device=device, dtype=channeled_image.dtype)
            combined_image[x_start:x_end, y_start:y_end] += shifted[
                ch_x_start:ch_x_end, ch_y_start:ch_y_end
            ]

    return combined_image


def generate_combined_image_colored(
    positions, channeled_image, domain_size, r_max,
    cmap="Greens_r", overlap_color=(1.0, 1.0, 1.0), norm_multiplier=1, norm=None,
    values=None, device=None,
):
    """Assemble a combined RGBA image, coloring each mask by a per-cell value.

    Args:
        positions: (n, 2) polygon/sphere centroid positions
        channeled_image: (n, image_size, image_size) per-object images
        domain_size: physical domain size
        r_max: max radius used when rasterizing the channeled images
        cmap: matplotlib colormap name or colormap object
        overlap_color: RGB tuple used where multiple masks overlap
        norm: matplotlib Normalize object for colormap normalization (default None)
        values: optional (n,) per-cell scalar values to color by; defaults to
            per-cell mask area (number of foreground pixels).
    Returns:
        rgba: (combined_size, combined_size, 4) numpy RGBA image
    """
    n = positions.shape[0]
    w = channeled_image.shape[1]
    dx = 2 * r_max / (w - 1)
    combined_size = int(domain_size / dx)
    grid_center = (w - 1) / 2.0
    half_w = w // 2

    masks = []
    areas = np.empty(n)
    for i in range(n):
        img_np = channeled_image[i].cpu().numpy()
        mask = img_np > 0.5
        masks.append((img_np, mask))
        areas[i] = mask.sum()

    if values is None:
        color_values = areas
    else:
        if isinstance(values, torch.Tensor):
            color_values = values.detach().cpu().numpy()
        else:
            color_values = np.asarray(values)
        if color_values.shape[0] != n:
            raise ValueError(
                f"values has length {color_values.shape[0]}, expected {n}"
            )

    if isinstance(cmap, str):
        cmap_obj = plt.colormaps[cmap]
    else:
        cmap_obj = cmap
    if norm is None:
        norm = plt.Normalize(
            vmin=color_values.min(), vmax=color_values.max() * norm_multiplier
        )

    # Track per-pixel: RGBA color and how many masks contribute
    rgba = np.zeros((combined_size, combined_size, 4))
    count = np.zeros((combined_size, combined_size), dtype=int)
    # Store the last-written color per pixel so we can detect overlap
    color_buffer = np.zeros((combined_size, combined_size, 3))

    overlap_rgb = np.array(overlap_color[:3], dtype=float)

    for i in range(n):
        img_np, mask_full = masks[i]
        color = np.array(cmap_obj(norm(color_values[i]))[:3])

        rows, cols = np.where(mask_full)
        if len(rows) > 0:
            cx, cy = rows.mean(), cols.mean()
        else:
            cx, cy = grid_center, grid_center

        offset_x = cx - grid_center
        offset_y = cy - grid_center
        exact_x = positions[i, 0].item() / dx - offset_x
        exact_y = positions[i, 1].item() / dx - offset_y
        center_x = int(exact_x)
        center_y = int(exact_y)
        sub_x = exact_x - center_x
        sub_y = exact_y - center_y

        x_start = max(center_x - half_w, 0)
        x_end = min(center_x + half_w, combined_size)
        y_start = max(center_y - half_w, 0)
        y_end = min(center_y + half_w, combined_size)

        ch_x_start = max(half_w - center_x, 0)
        ch_x_end = w - max(center_x + half_w - combined_size, 0)
        ch_y_start = max(half_w - center_y, 0)
        ch_y_end = w - max(center_y + half_w - combined_size, 0)

        if x_start >= x_end or y_start >= y_end:
            continue

        shifted = ndimage_shift(img_np, (sub_x, sub_y), order=1)
        patch = shifted[ch_x_start:ch_x_end, ch_y_start:ch_y_end]
        patch_mask = patch > 0.5

        region_count = count[x_start:x_end, y_start:y_end]

        # Where this mask lands on already-occupied pixels -> overlap
        overlap = patch_mask & (region_count > 0)
        # Where this mask lands on empty pixels -> new
        new = patch_mask & (region_count == 0)

        rgba[x_start:x_end, y_start:y_end][new, :3] = color
        rgba[x_start:x_end, y_start:y_end][new, 3] = 1.0
        color_buffer[x_start:x_end, y_start:y_end][new] = color

        rgba[x_start:x_end, y_start:y_end][overlap, :3] = overlap_rgb
        rgba[x_start:x_end, y_start:y_end][overlap, 3] = 1.0

        count[x_start:x_end, y_start:y_end] += patch_mask.astype(int)

    return rgba, norm, cmap_obj


def _preregister_metadata(model, state_dict):
    """Register metadata buffers from a state_dict so load_state_dict won't fail."""
    prefix = "_meta_"
    for key, value in state_dict.items():
        if key.startswith(prefix):
            meta_name = key[len(prefix):]
            model.set_metadata(meta_name, value)


def load_run(run_id=None, experiments_dir="./experiments", use_last=False):
    """
    Load a run from either MLflow or Hydra.

    Args:
        run_id: Optional MLflow run ID or Hydra run folder name (e.g., "2024-01-15/10-30-45").
                If None, loads the most recent run.
        experiments_dir: Base directory for experiments (default: "./experiments")
        use_last: If True, load the last (final) checkpoint instead of the best.

    Returns:
        dict with: model, cfg, save_path, device
        Model metadata (rs, growth_rates, r_max) is loaded automatically via state_dict.
    """
    experiments_dir = Path(experiments_dir)

    if run_id is None:
        # Find the most recent Hydra run
        save_path = _find_most_recent_run(experiments_dir)
    elif "/" in str(run_id):
        # Assume it's a Hydra path like "2024-01-15/10-30-45"
        save_path = experiments_dir / run_id
    else:
        # Try to find by MLflow run ID
        save_path = _find_run_by_mlflow_name(run_id)

    if not save_path.exists():
        raise ValueError(f"Run path not found: {save_path}")

    # Load Hydra config
    hydra_config_path = save_path / ".hydra" / "config.yaml"
    if not hydra_config_path.exists():
        raise ValueError(f"Hydra config not found at: {hydra_config_path}")

    cfg = OmegaConf.load(hydra_config_path)

    # Set up device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model (type is determined by config)
    model = create_model(cfg, device=device)

    # Load model weights: prefer best, unless use_last is set
    best_model_path = save_path / "graph_decoder_model_best.pth"
    final_model_path = save_path / "graph_decoder_model.pth"

    if use_last:
        load_order = [(final_model_path, "last"), (best_model_path, "best")]
    else:
        load_order = [(best_model_path, "best"), (final_model_path, "last")]

    for path, label in load_order:
        if path.exists():
            state_dict = torch.load(path, map_location=device)
            _preregister_metadata(model, state_dict)
            # Checkpoints from before the x_features normalisation buffers were
            # added: the model's defaults (ones/zeros) are the identity they
            # were trained with.
            for key in ("feature_scale", "feature_offset"):
                if key not in state_dict and hasattr(model, key):
                    state_dict[key] = getattr(model, key)
            model.load_state_dict(state_dict)
            print(f"Loaded {label} model from: {path}")
            break
    else:
        print("Warning: No saved model weights found, using initialized model")

    model.eval()

    # Metadata (rs, growth_rates, r_max) is loaded automatically via state_dict

    return {
        "model": model,
        "cfg": cfg,
        "save_path": save_path,
        "device": device,
    }


def _find_most_recent_run(experiments_dir):
    """Find the most recent Hydra run directory."""
    experiments_dir = Path(experiments_dir)

    if not experiments_dir.exists():
        raise ValueError(f"Experiments directory not found: {experiments_dir}")

    # Find all date directories, then find most recent time directory within most recent date
    date_dirs = sorted([d for d in experiments_dir.iterdir() if d.is_dir() and d.name != "mlruns" and d.name != "multirun" and "dino" not in d.name], reverse=True)

    if not date_dirs:
        raise ValueError(f"No experiment runs found in: {experiments_dir}")

    for date_dir in date_dirs:
        time_dirs = sorted([d for d in date_dir.iterdir() if d.is_dir()], reverse=True)
        if time_dirs:
            return time_dirs[0]

    raise ValueError(f"No experiment runs found in: {experiments_dir}")


def _find_run_by_mlflow_name(run_name):
    """Find Hydra run directory by MLflow run name."""
    runs = mlflow.search_runs(filter_string=f"attributes.run_name = '{run_name}'", search_all_experiments=True)
    if not runs.empty:
        try:
            hydra_dir = runs.iloc[0]["tags.hydra_output_dir"]
        except KeyError:
            # check the params
            hydra_dir = runs.iloc[0]["params.hydra_output_dir"]

        return Path(hydra_dir)
    else:
        raise ValueError(f"MLflow run name not found: {run_name}")

def rescale_dino_to_channeled(decoded_images, crop_sides, image_size):
    """Rescale DINO decoded 224x224 SDF images to channeled image (binary mask) space.

    The DINO decoder outputs SDF images in [-1, 1] where positive = inside.
    This function converts them to binary masks and rescales to the original
    channeled image coordinate system.

    Args:
        decoded_images: (N, 224, 224) SDF images from decode_to_image()
        crop_sides: (N,) crop side lengths in channeled image pixels
        image_size: Original channeled image size (e.g. 64 or 128)

    Returns:
        (N, image_size, image_size) binary mask images in channeled image coordinate system
    """
    N = decoded_images.shape[0]
    device = decoded_images.device
    # Convert SDF to binary mask: positive SDF = inside object = 1
    masks = (decoded_images > 0).float()
    result = torch.zeros(N, image_size, image_size, device=device)

    # Grid center matching polyhoop's linspace(-r_max, r_max, image_size)
    grid_center = (image_size - 1) / 2.0

    for i in range(N):
        cs = int(crop_sides[i].clamp(1, image_size).item())
        # Resize 224x224 -> crop_side x crop_side
        img = masks[i].unsqueeze(0).unsqueeze(0)  # (1, 1, 224, 224)
        resized = F.interpolate(img, size=(cs, cs), mode='bilinear', align_corners=False)
        resized = (resized.squeeze(0).squeeze(0) > 0.5).float()  # re-threshold after interpolation

        # Find shape center of mass in the resized crop, then align it to the
        # grid center. This corrects for the encoding's bbox-centered crop which
        # may not be centered on the canvas center.
        ys, xs = torch.where(resized > 0.5)
        if len(ys) > 0:
            cy_crop = ys.float().mean().item()
            cx_crop = xs.float().mean().item()
        else:
            cy_crop = (cs - 1) / 2.0
            cx_crop = (cs - 1) / 2.0

        y_start = round(grid_center - cy_crop)
        x_start = round(grid_center - cx_crop)

        # Clip to canvas bounds
        src_y0 = max(-y_start, 0)
        src_x0 = max(-x_start, 0)
        src_y1 = min(cs, image_size - y_start)
        src_x1 = min(cs, image_size - x_start)
        dst_y0 = max(y_start, 0)
        dst_x0 = max(x_start, 0)
        dst_y1 = dst_y0 + (src_y1 - src_y0)
        dst_x1 = dst_x0 + (src_x1 - src_x0)

        if dst_y0 < dst_y1 and dst_x0 < dst_x1:
            result[i, dst_y0:dst_y1, dst_x0:dst_x1] = resized[src_y0:src_y1, src_x0:src_x1]

    return result


def dino_forward_and_decode(model, batch, cfg):
    """Run DINO model forward pass and decode to channeled images.

    Args:
        model: GraphModelDINO instance
        batch: Dictionary containing states, W_interactions, dino_embeddings
        cfg: OmegaConf config

    Returns:
        new_states: Updated states from model
        channeled_images: (B, N, image_size, image_size) rescaled to channeled image space
    """
    image_size = cfg.data.image_size
    B = batch["L"].shape[0]

    new_states, L_out, next_scale = model.forward_from_batch(batch)
    decoded_images = model.decode_to_image(L_out)  # (B, N, 224, 224)

    channeled_list = []
    for b in range(B):
        channeled_list.append(
            rescale_dino_to_channeled(decoded_images[b], next_scale[b], image_size)
        )
    channeled_images = torch.stack(channeled_list)  # (B, N, image_size, image_size)

    return new_states, channeled_images


def plot_generated_images(batch, model, cfg: OmegaConf, n_samples=4):
    """
    Plot generated images from the model alongside ground truth images.
    First half shows combined images, second half shows individual channeled images.

    Args:
        batch: Dictionary containing batch data (states, W_interactions, timestep_idx, images)
        model: GraphModel
        simulation_type: Type of simulation ('sphere_interaction' or 'polyhoop')
        n_samples: Number of samples to display (default 4, split between combined and channeled)

    Returns:
        fig, ax: Matplotlib figure and axes
    """
    # Get metadata from model
    r_max = model.get_metadata('r_max')
    domain_size = model.domain_size
    if r_max is not None:
        r_max = r_max.item()

    states = batch["states"]
    W_interactions = batch["W_interactions"]
    timestep_idx = batch["timestep_idx"]

    # Limit n_samples to batch size
    batch_size = states.shape[0]
    n_samples = min(n_samples, batch_size)

    # Split samples: first half combined, second half channeled
    n_combined = n_samples // 2
    n_channeled = n_samples - n_combined

    # Get model predictions
    with torch.no_grad():
        if isinstance(model, GraphModelDINO):
            new_states, decoded_images = dino_forward_and_decode(model, batch, cfg)
        else:
            new_states, decoded_images, _ = model(states, W_interactions, timestep_idx)

    # Select appropriate image generation function
    simulation_type = cfg.data.params.name
    if simulation_type == 'sphere_interaction':
        from sphere_interaction import generate_combined_image as gen_combined
    elif simulation_type == 'polyhoop':
        from graph_dyno.data.polyhoop_simulation import generate_combined_image as gen_combined
    elif simulation_type == 'real':
        # Real cells are rasterized object-centred, like the DINO decode path,
        # so the centroid-aligning combiner is the right ground-truth renderer.
        # It also guards off-canvas writes, so padded slots render nothing.
        gen_combined = generate_combined_image_dino
    else:
        raise ValueError(f"Unknown simulation type: {simulation_type}")

    # For DINO models, use centroid-based placement for generated images
    gen_combined_fn = generate_combined_image_dino if isinstance(model, GraphModelDINO) else gen_combined

    # Create figure with 3 rows: ground truth, generated, and difference
    fig, axes = plt.subplots(3, n_samples, figsize=(3 * n_samples, 9), dpi=150)
    if n_samples == 1:
        axes = axes.reshape(3, 1)

    # First half: combined images
    for i in range(n_combined):
        x_sample = new_states[i, :, :2].cpu()
        channeled_image_sample = decoded_images[i].cpu()
        generated_combined = gen_combined_fn(
            positions=x_sample,
            channeled_image=channeled_image_sample,
            domain_size=domain_size,
            r_max=r_max,
        )

        if "channeled_images" in batch:
            gt_channeled = batch["channeled_images"][i].cpu()
            gt_positions = states[i, :, :2].cpu()
            gt_combined = gen_combined(
                positions=gt_positions,
                channeled_image=gt_channeled,
                domain_size=domain_size,
                r_max=r_max,
            )
            axes[0, i].imshow(gt_combined.detach(), cmap='gray')
            axes[0, i].set_title(f'GT Combined {i+1}')

            # Difference: red=GT only, blue=gen only, green=both
            gt_np = gt_combined.detach().cpu().numpy()
            gen_np = generated_combined.detach().cpu().numpy()
            diff = gt_np - gen_np
            diff_rgb = np.zeros((*gt_np.shape, 3))
            diff_rgb[diff > 0.5, 0] = 1
            diff_rgb[diff < -0.5, 2] = 1
            diff_rgb[(gt_np > 0.5) & (gen_np > 0.5), 1] = 0.5
            axes[2, i].imshow(diff_rgb)
            axes[2, i].set_title(f'Diff {i+1}')
        else:
            axes[0, i].axis('off')
            axes[0, i].set_title('No GT available')
            axes[2, i].axis('off')

        axes[1, i].imshow(generated_combined.detach(), cmap='gray')
        axes[1, i].set_title(f'Gen Combined {i+1}')

        for row in range(3):
            axes[row, i].set_xticks([])
            axes[row, i].set_yticks([])

    # Second half: individual channeled images
    num_nodes = decoded_images.shape[1]
    for j in range(n_channeled):
        i = n_combined + j  # Column index in plot
        sample_idx = j % batch_size  # Which batch sample to use
        node_idx = (j * 17) % num_nodes  # Pick different nodes (stride by prime for variety)

        # Get generated channeled image for this node
        gen_channeled = decoded_images[sample_idx, node_idx].cpu().squeeze()

        if "channeled_images" in batch:
            gt_channeled = batch["channeled_images"][sample_idx, node_idx].cpu().squeeze()
            axes[0, i].imshow(gt_channeled.detach(), cmap='gray')
            axes[0, i].set_title(f'GT Node {node_idx}')

            # Difference: red=GT only, blue=gen only, green=both
            gt_np = gt_channeled.detach().cpu().numpy()
            gen_np = gen_channeled.detach().cpu().numpy()
            diff = gt_np - gen_np
            diff_rgb = np.zeros((*gt_np.shape, 3))
            diff_rgb[diff > 0.5, 0] = 1
            diff_rgb[diff < -0.5, 2] = 1
            diff_rgb[(gt_np > 0.5) & (gen_np > 0.5), 1] = 0.5
            axes[2, i].imshow(diff_rgb)
            axes[2, i].set_title(f'Diff Node {node_idx}')
        else:
            axes[0, i].axis('off')
            axes[0, i].set_title('No GT available')
            axes[2, i].axis('off')

        axes[1, i].imshow(gen_channeled.detach(), cmap='gray')
        axes[1, i].set_title(f'Gen Node {node_idx}')

        for row in range(3):
            axes[row, i].set_xticks([])
            axes[row, i].set_yticks([])

    axes[0, 0].set_ylabel('Ground Truth')
    axes[1, 0].set_ylabel('Generated')
    axes[2, 0].set_ylabel('Diff (red=GT, blue=gen)')

    plt.tight_layout()
    return fig, axes


def plot_training(batch, model, losses, epoch, cfg: OmegaConf):
    # Get metadata from model
    rs = model.get_metadata('rs')
    r_max = model.get_metadata('r_max')
    domain_size = model.domain_size
    if r_max is not None:
        r_max = r_max.item()

    states = batch["states"]  # (batch_size, num_nodes, spatial_dim)
    W_interactions = batch["W_interactions"]

    if isinstance(model, GraphModelDINO):
        new_states, decoded_images = dino_forward_and_decode(model, batch, cfg)
    else:
        timestep_idx = batch["timestep_idx"]
        new_states, decoded_images, _ = model(states, W_interactions, timestep_idx)

    x_sample = new_states[0, :, :2].cpu()
    channeled_image_sample = decoded_images[0].cpu()

    simulation_type = cfg.data.params.name

    if isinstance(model, GraphModelDINO):
        sample_combined_image = generate_combined_image_dino(
            positions=x_sample,
            channeled_image=channeled_image_sample,
            domain_size=domain_size,
            r_max=r_max,
        )
    elif simulation_type == 'sphere_interaction':
        from sphere_interaction import generate_combined_image as generate_combined_image_sphere_interaction
        sample_combined_image = generate_combined_image_sphere_interaction(
            positions=x_sample,
            channeled_image=channeled_image_sample,
            domain_size=domain_size,
            r_max=r_max,
        )
    elif simulation_type == 'polyhoop':
        from graph_dyno.data.polyhoop_simulation import generate_combined_image as generate_combined_image_polyhoop
        sample_combined_image = generate_combined_image_polyhoop(
            positions=x_sample,
            channeled_image=channeled_image_sample,
            domain_size=domain_size,
            r_max=r_max,
        )

    fig, ax = plt.subplots(2,2, figsize=(7.5, 7.5), dpi=300)
    ax[0,0].imshow(sample_combined_image.detach(), cmap='gray')

    # plot loss curve
    ax[0,1].semilogy(losses[:epoch+1,0].cpu(), label='Train Loss')
    ax[0,1].semilogy(losses[:epoch+1,1].cpu(), label='Val Loss')
    ax[0,1].legend()

    # Helper to plot embedding with optional coloring by rs
    def plot_embedding_scatter(ax, embedding_2d, rs, label='Sphere Radius', colorbar=True):
        if rs is not None:
            r_values = rs[:model.num_nodes].cpu().numpy()
            norm = plt.Normalize(vmin=r_values.min(), vmax=r_values.max())
            cmap = matplotlib.colormaps['viridis']
            ax.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=r_values, s=5, edgecolors='k', linewidths=0.2, norm=norm, cmap=cmap)
            if colorbar:
                cax = ax.inset_axes([1.05, 0, 0.05, 1])
                plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), label=label, cax=cax)
        else:
            ax.scatter(embedding_2d[:, 0], embedding_2d[:, 1], s=5, edgecolors='k', linewidths=0.2)

    if model.embedding_dim == 1:
        ax[1,0].hist(model.parameter_embedding.data.cpu().squeeze(), bins=30)
    elif model.embedding_dim == 2:
        embedding_data = model.parameter_embedding.data.cpu().numpy()
        plot_embedding_scatter(ax[1,0], embedding_data, rs, colorbar=False)
    elif model.embedding_dim > 2:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        embedding_2d = pca.fit_transform(model.parameter_embedding.data.cpu().numpy())
        plot_embedding_scatter(ax[1,0], embedding_2d, rs, colorbar=False)

    # plot shape embedding / bottleneck PCA
    if isinstance(model, GraphModelDINO):
        # For DINO: PCA of bottleneck vectors from first batch element
        bottleneck_0 = batch["L"][0].cpu().numpy()  # (num_nodes, bottleneck_dim)
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        bottleneck_2d = pca.fit_transform(bottleneck_0)
        plot_embedding_scatter(ax[1,1], bottleneck_2d, rs, label='Bottleneck PCA', colorbar=True)
        ax[1,1].set_title('Bottleneck PCA')
    else:
        # For base GraphModel: shape embedding at t=0
        t_idx_zero = torch.tensor([0], device=model.device)
        shape_at_t0 = model.shape_embedding.access(t_idx_zero)[0].cpu()  # (num_nodes, shape_embedding_dim)
        if model.shape_embedding_dim == 1:
            ax[1,1].hist(shape_at_t0.squeeze().numpy(), bins=30)
        elif model.shape_embedding_dim == 2:
            shape_data = shape_at_t0.numpy()
            plot_embedding_scatter(ax[1,1], shape_data, rs, colorbar=True)
        elif model.shape_embedding_dim > 2:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            embedding_2d = pca.fit_transform(shape_at_t0.numpy())
            plot_embedding_scatter(ax[1,1], embedding_2d, rs, colorbar=True)

    return fig, ax

    

def plot_embedding(model):
    """Plot the parameter embedding, colored by rs if available in model metadata."""
    embedding = model.parameter_embedding.detach().cpu()
    rs = model.get_metadata('rs') if hasattr(model, 'get_metadata') else None
    r_values = rs[:model.num_nodes].cpu().numpy() if rs is not None else None

    fig, ax = plt.subplots(figsize=(4,4), dpi=300)
    if model.embedding_dim == 1:
        ax.hist(embedding.squeeze(), bins=30, color='blue', alpha=0.7)
        ax.set_xlabel('Embedding Dimension 1')
        ax.set_ylabel('Count')
    elif model.embedding_dim == 2:
        if r_values is not None:
            norm = plt.Normalize(vmin=r_values.min(), vmax=r_values.max())
            cmap = matplotlib.colormaps['viridis']
            sc = ax.scatter(embedding[:, 0], embedding[:, 1], c=r_values, s=10, edgecolors='k', linewidths=0.2, norm=norm, cmap=cmap)
            plt.colorbar(sc, label='Sphere Radius', ax=ax)
        else:
            ax.scatter(embedding[:, 0], embedding[:, 1], s=10, edgecolors='k', linewidths=0.2)
        ax.set_xlabel('Embedding Dimension 1')
        ax.set_ylabel('Embedding Dimension 2')
    elif model.embedding_dim > 2:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        embedding_2d = pca.fit_transform(model.parameter_embedding.data.cpu().numpy())
        if r_values is not None:
            norm = plt.Normalize(vmin=r_values.min(), vmax=r_values.max())
            cmap = matplotlib.colormaps['viridis']
            ax.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=r_values, s=5, edgecolors='k', linewidths=0.2, norm=norm, cmap=cmap)
            plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), label='Sphere Radius', ax=ax)
        else:
            ax.scatter(embedding_2d[:, 0], embedding_2d[:, 1], s=5, edgecolors='k', linewidths=0.2)

    return fig, ax

def make_predicted_video(model, cfg, num_frames=-1):
    # load the dataset
    from graph_dyno.data.dataloader import create_split_dataloaders_from_folder

    r_max = model.get_metadata('r_max')
    if r_max is not None:
        r_max = r_max.item()

    folder_path=cfg.dataloader.folder_path
    temp_cfg = cfg.copy()
    temp_cfg.dataloader.batch_size = 1
    train, val, test = create_split_dataloaders_from_folder(
        folder_path=folder_path,
        cfg=temp_cfg,
        batch_size=1,
        num_workers=0,
        shuffle=False,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    domain_size = model.domain_size
    # now make predictions on all the images in the val set and make a video
    combined_images = []
    # set the model to eval mode
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(train, desc="Generating Predictions")):
            if num_frames > 0 and i >= num_frames:
                break
            if batch is None:
                continue
            batch = {k: v.to(device) for k, v in batch.items()}

            if isinstance(model, GraphModelDINO):
                new_states, decoded_images = dino_forward_and_decode(model, batch, cfg)
            else:
                new_states, decoded_images, _ = model(
                    batch["states"], batch["W_interactions"], batch["timestep_idx"]
                )

            x_sample = new_states[0, :, :2].cpu()
            channeled_image_sample = decoded_images[0].cpu()

            if isinstance(model, GraphModelDINO):
                sample_combined_image = generate_combined_image_dino(
                    positions=x_sample,
                    channeled_image=channeled_image_sample,
                    domain_size=domain_size,
                    r_max=r_max,
                )
            elif cfg.data.params.name == 'sphere_interaction':
                from sphere_interaction import generate_combined_image as generate_combined_image_sphere_interaction
                sample_combined_image = generate_combined_image_sphere_interaction(
                    positions=x_sample,
                    channeled_image=channeled_image_sample,
                    domain_size=domain_size,
                    r_max=r_max,
                )
            elif cfg.data.params.name == 'polyhoop':
                from graph_dyno.data.polyhoop_simulation import generate_combined_image as generate_combined_image_polyhoop
                sample_combined_image = generate_combined_image_polyhoop(
                    positions=x_sample,
                    channeled_image=channeled_image_sample,
                    domain_size=domain_size,
                    r_max=r_max,
                )

            combined_images.append(sample_combined_image)

    fig, ax = plt.subplots(figsize=(5,5), dpi=300)
    im = ax.imshow(combined_images[0].detach(), cmap='gray')

    def update(frame):
        im.set_array(frame)
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=[img.detach().numpy() for img in combined_images], blit=True, repeat=False)
    with tqdm(total=len(combined_images), desc="Saving Video") as pbar:
        progress = lambda *args: pbar.update()
        ani.save(f"prediction_{cfg.data.params.name}.mp4", writer='ffmpeg', fps=10, progress_callback=progress)
    plt.close(fig)


def make_rollout_video(model, cfg, num_frames=100):
    """Autoregressive rollout: start from training data at t=0, then generate forward."""
    from graph_dyno.data.dataloader import create_split_dataloaders_from_folder

    r_max = model.get_metadata('r_max')
    if r_max is not None:
        r_max = r_max.item()

    folder_path = cfg.dataloader.folder_path
    temp_cfg = cfg.copy()
    temp_cfg.dataloader.batch_size = 1
    train, _, _ = create_split_dataloaders_from_folder(
        folder_path=folder_path,
        cfg=temp_cfg,
        batch_size=1,
        num_workers=0,
        shuffle=False,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    domain_size = model.domain_size
    r_cutoff = cfg.data.params.r_cutoff
    image_size = cfg.data.image_size

    # Get initial conditions from first non-None batch
    batch = None
    for batch in train:
        if batch is not None:
            break
    batch = {k: v.to(device) for k, v in batch.items()}

    # Current state: (num_nodes, feature_dim)
    x = batch["states"][0]  # unbatched
    num_nodes = x.shape[0]

    # Initial bottleneck and scale (L is precomputed at dataloader setup)
    L = batch["L"][0]  # (num_nodes, bottleneck_dim)

    current_scale = batch["crop_sides"][0].float()  # (num_nodes,)
    b = model.parameter_embedding.data  # (num_nodes, embedding_dim)

    # Initialize velocities for step 0 (zero velocity — no previous state)
    if model.use_velocities:
        prev_L = L.clone()
        prev_scale = current_scale.clone()

    combined_images = []
    model.eval()
    with torch.no_grad():
        for step in tqdm(range(num_frames), desc="Rolling out"):
            # Compute interaction graph from current positions
            positions = x[:, :2]
            diff = positions.unsqueeze(0) - positions.unsqueeze(1)
            dist = torch.norm(diff, dim=2)
            mask = (dist <= r_cutoff) & (dist > 0)
            indices = mask.nonzero(as_tuple=False).t()
            values = torch.ones(indices.shape[1], dtype=torch.bool, device=device)
            W = torch.sparse_coo_tensor(
                indices, values, (num_nodes, num_nodes),
                dtype=torch.bool, device=device,
            )

            # Compute velocities if enabled
            L_velocity = None
            scale_velocity = None
            if model.use_velocities:
                L_velocity = L - prev_L
                scale_velocity = current_scale - prev_scale

            # Single-step generate (unbatched)
            new_x, L_out, next_scale = model.generate(x, W, b, L, current_scale, L_velocity, scale_velocity)

            # Decode to images
            decoded_sdf = model.decode_to_image(L_out)  # (num_nodes, 224, 224)
            channeled = rescale_dino_to_channeled(decoded_sdf, next_scale, image_size)

            combined = generate_combined_image_dino(
                positions=new_x[:, :2].cpu(),
                channeled_image=channeled.cpu(),
                domain_size=domain_size,
                r_max=r_max,
            )
            combined_images.append(combined)

            # Save current as prev before advancing
            if model.use_velocities:
                prev_L = L
                prev_scale = current_scale

            # Advance state
            x = new_x
            L = L_out
            current_scale = next_scale

    # Save video
    fig, ax = plt.subplots(figsize=(5, 5), dpi=300)
    im = ax.imshow(combined_images[0].detach(), cmap='gray')
    ax.axis('off')

    def update(frame):
        im.set_array(frame)
        return [im]

    ani = animation.FuncAnimation(
        fig, update,
        frames=[img.detach().numpy() for img in combined_images],
        blit=True, repeat=False,
    )
    out_path = f"rollout_{cfg.data.params.name}.mp4"
    with tqdm(total=len(combined_images), desc="Saving Rollout Video") as pbar:
        progress = lambda *args: pbar.update()
        ani.save(out_path, writer='ffmpeg', fps=10, progress_callback=progress)
    plt.close(fig)
    print(f"Saved rollout video to {out_path}")

import os
def plot_embedding_colored_by_changing_params(model, use_subplots=False, save_dir=None, **plot_kwargs):
    """Plot the parameter embedding colored by changing parameters like oscillation period and amplitude."""
    if save_dir is not None and not os.path.exists(save_dir):
        raise ValueError(f"Save directory {save_dir} does not exist.")
    parameter_embedding = model.parameter_embedding.detach().cpu()
    if model.embedding_dim > 2:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        parameter_embedding = pca.fit_transform(parameter_embedding.numpy())
    changing_params = {key: value for key, value in model.metadata.items() if isinstance(value, torch.Tensor) and value.numel() > 0 and len(torch.unique(value)) > 1}
    if use_subplots:
        n_subplots = len(changing_params)
        # try to make it square otherwise make it horizontal aspect
        n_cols = int(np.ceil(np.sqrt(n_subplots)))
        n_rows = int(np.ceil(n_subplots / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols,4*n_rows), dpi=300, sharex=True, sharey=True)
        axes = np.atleast_1d(axes)
        
    for i, (key, value) in enumerate(changing_params.items()):
        # if there is variation in this parameter across nodes, plot it
        # check that there are at least 2 unique values
        if not use_subplots:
            fig, ax = plt.subplots(figsize=(4,4), dpi=300)
        else:
            ax = axes.flatten()[i]  
        seaborn_kwargs = {
            'legend': False,
            'palette': 'viridis',
        }
        seaborn_kwargs.update(plot_kwargs)
        sns.scatterplot(x=parameter_embedding[:,0], y=parameter_embedding[:,1], hue=value.cpu(), ax=ax, **seaborn_kwargs)
        ax.set_xlabel('Embedding Dimension 1')
        ax.set_ylabel('Embedding Dimension 2')
        ax.set_title(f'{key}')
        if save_dir is not None and not use_subplots:
            fig.savefig(os.path.join(save_dir, f'embedding_{key}.png'))
    if use_subplots and save_dir is not None:
        fig.savefig(os.path.join(save_dir, f'embedding_changing_params.png'))
    return fig


#%%
if __name__ == "__main__":
    run = load_run()  # or load_run("2024-01-15/10-30-45") for a specific run
    # print the run keys
    print("Run keys:", run.keys())
    # print the date and time of the run
    print("Run path:", run["save_path"])
    model = run["model"]
    cfg = run["cfg"]
    save_path = run["save_path"]
    device = run["device"]

    # Get metadata from model
    rs = model.get_metadata('rs')
    growth_rates = model.get_metadata('growth_rates')
    r_max = model.get_metadata('r_max')
    domain_size = model.domain_size
    dt = cfg.data.dt
    r_cutoff = cfg.data.params.r_cutoff
    oscillation_period = model.get_metadata('area_fluct_periods').cpu()
    oscillation_amplitude = model.get_metadata('area_fluct_amps').cpu()


    #%%
    make_rollout_video(model, cfg, num_frames=500)
    #%%
    make_predicted_video(model, cfg, num_frames=500)

    #%%
    #plot the embedding colored by period
    parameter_embedding = model.parameter_embedding.detach().cpu()
    sns.scatterplot(x=parameter_embedding[:,0], y=parameter_embedding[:,1], hue=oscillation_amplitude, legend=False, palette='viridis')
    plt.xlabel('Embedding Dimension 1')
    plt.ylabel('Embedding Dimension 2')

    #%%
    fig = plot_embedding_colored_by_changing_params(model, use_subplots=True, save_dir=save_path)

    #%%
    # scatter the unique pairs of period and amplitude
    unique_periods = torch.unique(oscillation_period)
    unique_amps = torch.unique(oscillation_amplitude)
    unique_pairs = set((p.item(), a.item()) for p in unique_periods for a in unique_amps)
    pair_to_indices = {pair: [] for pair in unique_pairs}
    for i in range(len(oscillation_period)):
        pair = (oscillation_period[i].item(), oscillation_amplitude[i].item())
        pair_to_indices[pair].append(i)

    sns.scatterplot(x=parameter_embedding[:,0], y=parameter_embedding[:,1], hue=[(oscillation_period[i].item(), oscillation_amplitude[i].item()) for i in range(len(oscillation_period))], legend=False)

    #%%
    omega = 2 * np.pi / oscillation_period
    rate = omega * oscillation_amplitude
    hue_var = rate
    sns.scatterplot(x=parameter_embedding[:,0], y=parameter_embedding[:,1], hue=hue_var, legend=False, palette='viridis')

    #%%
    # for each group of amplitueds plot their embedding colored by period
    fig, axs = plt.subplots(2,int(len(unique_amps)/2), figsize=(10,10), dpi=300, sharex=True, sharey=True)
    for i, amp in enumerate(unique_amps):
        indices = (oscillation_amplitude == amp).nonzero(as_tuple=False).squeeze()
        sns.scatterplot(x=parameter_embedding[indices,0], y=parameter_embedding[indices,1], hue=oscillation_period[indices].cpu(), legend=False, palette='viridis', ax=axs[i//(len(unique_amps)//2), i%(len(unique_amps)//2)])
        axs[i//(len(unique_amps)//2), i%(len(unique_amps)//2)].set_title(f'Amplitude {amp.item()}')
        axs[i//(len(unique_amps)//2), i%(len(unique_amps)//2)].set_xlabel('Embedding Dimension 1')
        axs[i//(len(unique_amps)//2), i%(len(unique_amps)//2)].set_ylabel('Embedding Dimension 2')
    plt.tight_layout()



    # %%
    import pandas as pd
    r_values = rs.cpu().numpy()
    growth_rates_np = growth_rates.cpu().numpy()
    df = pd.DataFrame({
        'radius': r_values,
        'growth_rate': growth_rates_np,
    })

    df['growth_rate_percentage'] = df['growth_rate'] / df['radius']
    df['growth_rate_percentage'] = df['growth_rate_percentage'].round(4)
    print(df)

    df.query('radius == 0.5').index.values

    #%%

    parameter_embedding = model.parameter_embedding.detach().cpu()
    sns.scatterplot(x=parameter_embedding[:,0], y=parameter_embedding[:,1], hue=df['growth_rate_percentage'], palette='viridis', legend=False)

    #%%
    # Get shape embeddings for all timepoints
    t_indices = torch.arange(model.num_timepoints, device=model.device)
    shape_embedding = torch.stack([model.shape_embedding.access(t_indices[i:i+1])[0] for i in range(model.num_timepoints)]).detach().cpu()

    xlim = (shape_embedding[:,:,0].min(), shape_embedding[:,:,0].max())
    ylim = (shape_embedding[:,:,1].min(), shape_embedding[:,:,1].max())

    fig, ax = plt.subplots(figsize=(6,6), dpi=300)

    def update(frame):
        t = frame
        ax.clear()
        sns.scatterplot(x=shape_embedding[t,:,0], y=shape_embedding[t,:,1], hue=df['radius'] + df['growth_rate']*dt*t, palette='viridis', legend=False, ax=ax)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        return ax

    ani = animation.FuncAnimation(fig, update, frames=shape_embedding.shape[0], blit=False, repeat=False)
    with tqdm(total=shape_embedding.shape[0], desc="Saving Shape Embedding Video") as pbar:
        progress = lambda *args: pbar.update()
        ani.save(f"{save_path}/shape_embedding.mp4", writer='ffmpeg', fps=30, progress_callback=progress)


    #%%
    # make a df of the latent space trajectories over time
    df_traj = pd.DataFrame(columns=['particle_index', 'time', 's_emb_0', 's_emb_1', 'rt', 'r0', 'growth_rate', 'growth_rate_percentage'])
    # Get shape embeddings for all timepoints
    t_indices = torch.arange(model.num_timepoints, device=model.device)
    shape_embedding_traj = torch.stack([model.shape_embedding.access(t_indices[i:i+1])[0] for i in range(model.num_timepoints)]).detach().cpu()

    for i in range(shape_embedding_traj.shape[1]):
        dt = 0.01
        t0 = 0
        tf = (shape_embedding_traj.shape[0]-1)*dt
        t = np.arange(t0, tf+dt, dt)
        r0 = df['radius'].iloc[i]
        growth_rate = df['growth_rate'].iloc[i]
        growth_rate_percentage = df['growth_rate_percentage'].iloc[i]
        rt = r0 + growth_rate * t

        traj_dict = {
            'particle_index': i,
            'time': t,
            's_emb_0': shape_embedding_traj[:,i,0].numpy(),
            's_emb_1': shape_embedding_traj[:,i,1].numpy(),
            'rt': rt,
            'r0': r0,
            'growth_rate': growth_rate,
            'growth_rate_percentage': growth_rate_percentage
        }
        df_particle = pd.DataFrame(traj_dict)
        df_traj = pd.concat([df_traj, df_particle], ignore_index=True)
    print(df_traj)

    #%%
    sns.scatterplot(df_traj.query('time==0'), x='s_emb_0', y='s_emb_1', hue='growth_rate_percentage', s=20, palette='viridis')


    #%%
    plt.plot(shape_embedding_traj[:,:,0], shape_embedding_traj[:,:,1], alpha=0.3)

    #%%

    # sample a latent variable first
    latent_space = model.latent_space.detach()
    # indices = df.query('growth_rate_percentage == -.05').index.values
    indices = df.index.values
    trajectories = []
    for i in indices:
        # choose corder
        # generate a time trajectory in the time embedding space
        dt = 0.01
        t0 = 0
        tf = 10
        t = torch.arange(t0, tf+dt, dt).unsqueeze(-1).to(device)  # (T, 1)
        b = latent_space[i].unsqueeze(0).expand(t.shape[0], -1)  # (T, embedding_dim)
        if model.use_time_and_embedding_concat:
            time_trajectory = model.time_embedding(torch.cat([t, b], dim=1)).detach()
        else:
            time_trajectory = model.time_embedding(t).detach()  # (T, embedding_dim)
                    
        traj_dict = {
            'particle_index': i, 
            'time': t.squeeze().cpu().numpy(),
            'b0_0': b[:,0].cpu().numpy(),
            'b0_1': b[:,1].cpu().numpy(),
            'bt_0': (b[:,0] + time_trajectory[:,0]).cpu().numpy(),
            'bt_1': (b[:,1] + time_trajectory[:,1]).cpu().numpy()
            }
        trajectories.append(traj_dict)

    df_traj = pd.DataFrame(trajectories)
    df_traj = df_traj.explode(['time', 'b0_0', 'b0_1', 'bt_0', 'bt_1']).reset_index(drop=True)
    # turn the columns into float
    df_traj = df_traj.astype({'time': 'float', 'b0_0': 'float', 'b0_1': 'float', 'bt_0': 'float', 'bt_1': 'float'})
    print(df_traj)

    #%% plot one time trajectory

    import seaborn as sns
    latent_space = model.latent_space.detach()
    print(latent_space.shape)
    sns.set_context("talk")
    # map the initial radius and growth rate to particle index
    df_traj['r0'] = df_traj['particle_index'].map(lambda i: df['radius'].iloc[i])
    df_traj['growth_rate'] = df_traj['particle_index'].map(lambda i: df['growth_rate'].iloc[i])
    df_traj['growth_rate_percentage'] = df_traj['particle_index'].map(lambda i: df['growth_rate_percentage'].iloc[i])
    df_traj['rt'] = df_traj['r0'] + df_traj['growth_rate'] * df_traj['time']

    #%%
    fig, ax = plt.subplots(1,2, figsize=(13, 7), sharey=True)
    # set separation
    fig.subplots_adjust(wspace=0.1)
    # scatter plot of latent space at t=0
    sns.scatterplot(df_traj.query('time == 0'), x='b0_0', y='b0_1', hue='r0', ax=ax[0], s=30, palette='flare_r')
    sns.scatterplot(df_traj.query('time == 0'), x='b0_0', y='b0_1', hue='growth_rate_percentage', ax=ax[1], s=30, palette='tab10')
    handle, label = ax[0].get_legend_handles_labels()
    ax[0].legend(handle, label, title=r"$r_0$")
    ax[0].set_xlabel(r"$b_0$")
    ax[0].set_ylabel(r"$b_1$")
    handle, label = ax[1].get_legend_handles_labels()
    ax[1].legend(handle, label, title=r"$\alpha/r_0$")
    ax[1].set_xlabel(r"$b_0$")
    ax[1].set_ylabel(r"$b_1$")

    fig.savefig(save_path / 'latent_space_initial.png', dpi=300)

    #%%
    fig, ax = plt.subplots(figsize=(6,6))
    sns.scatterplot(df_traj.query('time == 0'), x='bt_0', y='bt_1', hue='growth_rate_percentage', s=20, palette='viridis', ax=ax)
    handle, label = ax.get_legend_handles_labels()
    ax.legend(handle, label, title=r"$\alpha/r_0$")
    ax.set_xlabel(r"$b_0(t=0)$")
    ax.set_ylabel(r"$b_1(t=0)$")
    fig.savefig(save_path / 'latent_space_initial_bt.png', dpi=300)



    #%%
    fig, ax = plt.subplots(figsize=(6,6))
    for particle_index in df_traj['particle_index'].unique():
        df_particle = df_traj.query(f'particle_index == {particle_index}')
        ax.plot(df_particle['bt_0'], df_particle['bt_1'], label=f'Particle {particle_index}')
    
    fig.savefig(save_path / 'latent_space_trajectories_all.png', dpi=300)

    #%%
    t0 = 0
    tf = 2
    fig, ax = plt.subplots(2, 2, figsize=(6,6), sharex=True, sharey=True)
    sns.scatterplot(df_traj.query('@tf >= time >= @t0 and growth_rate_percentage==-0.05'), x='bt_0', y='bt_1', hue='rt', s=20, palette='viridis', ax=ax[0,0], legend=False)
    sns.scatterplot(df_traj.query('@tf >= time >= @t0 and growth_rate_percentage==-0.025'), x='bt_0', y='bt_1', hue='rt', s=20, palette='viridis', ax=ax[0,1], legend=False)
    sns.scatterplot(df_traj.query('@tf >= time >= @t0 and growth_rate_percentage==0.025'), x='bt_0', y='bt_1', hue='rt', s=20, palette='viridis', ax=ax[1,0], legend=False)
    sns.scatterplot(df_traj.query('@tf >= time >= @t0 and growth_rate_percentage==0.05'), x='bt_0', y='bt_1', hue='rt', s=20, palette='viridis', ax=ax[1,1], legend=False)

    #%%
    # make animation of the latent space trajectories over time
    fig, ax = plt.subplots(figsize=(6,6), dpi=300)
    times = np.sort(df_traj['time'].unique())
    vmin = df_traj['rt'].min()
    vmax = df_traj['rt'].max()
    xlim = (df_traj['bt_0'].min(), df_traj['bt_0'].max())
    ylim = (df_traj['bt_1'].min(), df_traj['bt_1'].max())
    padding_percentage = 0.03
    padding = (padding_percentage * (xlim[1] - xlim[0]), padding_percentage * (ylim[1] - ylim[0]))
    xlim = (xlim[0] - padding[0], xlim[1] + padding[0])
    ylim = (ylim[0] - padding[1], ylim[1] + padding[1])
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    df_time = df_traj.query(f'time == {times[0]}')  
    sc = ax.scatter(df_time['bt_0'], df_time['bt_1'], s=20, c=df_time['rt'], cmap='viridis', vmin=vmin, vmax=vmax)
    ax.set_xlabel(r"$b_0(t)$")
    ax.set_ylabel(r"$b_1(t)$")
    cbar = plt.colorbar(sc, ax=ax, label='r(t)')


    def update(frame):
        df_time = df_traj.query(f'time == {times[frame]}')
        # update the scatter plot states
        sc.set_offsets(np.c_[df_time['bt_0'], df_time['bt_1']])
        # update the colors
        sc.set_array(df_time['rt'])

        return ax.collections

    fps = 60
    ani = animation.FuncAnimation(fig, update, frames=len(df_traj['time'].unique()), blit=True, interval=200)
    with tqdm(total=len(df_traj['time'].unique()), desc="Saving animation") as pbar:
        callback = lambda *args: pbar.update()
        ani.save(save_path / 'latent_space_trajectories.mp4', writer='ffmpeg', progress_callback=callback, fps=fps, dpi=300)

    #%% do some rollout visualization

    # sample a latent variable first
    radii = rs.cpu().numpy()

    b_growth =df_traj.query('r0 == @radii[-10] and growth_rate_percentage == .05').sample(n=1, random_state=10)[['b0_0', 'b0_1']].values
    b_growth = torch.tensor(b_growth, dtype=torch.float32).to(device).squeeze()

    # # just get the initial positions from one of the validation batches
    # batch = next(iter(val_dataloader))
    # batch = {k: v.to(device) for k, v in batch.items()}
    # initial_states = batch["states"]
    # x_init = initial_states[0, :, :2]  # (n_spheres)

    print(b_growth)
    n_particles = 1
    # initialize states in a grid
    grid_size = int(n_particles**0.5)
    spacing = 2
    start_offset = 15
    x_init = torch.tensor([[start_offset+i * spacing, start_offset+j * spacing] for i in range(grid_size) for j in range(grid_size)], dtype=torch.float32).to(device)
    v_init = torch.zeros_like(x_init).to(device)
    

    n_steps = 200
    dt = 0.01
    save_step = 4
    states = torch.cat([x_init, v_init], dim=-1)  # (n_spheres, 4)
    trajectory = [states.cpu().numpy()]
    from pathlib import Path
    save_dir = Path('./rollout_images')
    save_dir.mkdir(exist_ok=True)
    images_to_write = []

    model.eval()
    with torch.no_grad():
        for step in tqdm(range(n_steps)):
            r_vec = states.unsqueeze(0) - states.unsqueeze(1)

            # Compute all pairwise distances
            # r: (n_spheres, n_spheres)
            r = torch.norm(r_vec, dim=2)
            # Create mask for interactions (within cutoff and not self)
            interaction_mask = torch.ones(n_particles, n_particles, dtype=torch.bool, device=states.device)

            # Zero out the diagonal (no self-interaction)
            interaction_mask[torch.arange(n_particles), torch.arange(n_particles)] = (
                False
            )

            interaction_mask = interaction_mask & (r < r_cutoff)
            indices = interaction_mask.nonzero(as_tuple=False).t()  # Shape: (2, nnz)
            values = torch.ones(indices.shape[1], dtype=torch.bool, device=states.device)
            W_interactions = torch.sparse_coo_tensor(
                indices, values, (n_particles, n_particles), dtype=torch.bool, device=states.device
            )

            t = torch.tensor([[step * dt]], device=device)

            new_x, image, bprime = model.generate(states, W_interactions, b_growth.expand(n_particles, -1), t=t)

            states = new_x
            trajectory.append(states.cpu().numpy())

            # show the images using generate_combined_image
            if step % save_step == 0:
                channeled_image = image.cpu().squeeze()
                if channeled_image.dim() == 2:
                    channeled_image = channeled_image.unsqueeze(0)
                combined_image = generate_combined_image(
                    positions=states[:,:2].cpu(),
                    channeled_image=channeled_image,
                    domain_size=domain_size,
                    r_max=r_max,
                )
                images_to_write.append(combined_image.numpy())

    states = torch.tensor(trajectory)  # (n_steps, n_spheres, 4)
    plt.scatter(states[:,0,0], states[:,0,1], c=torch.arange(n_steps+1))

    velocity = torch.norm(states[:, :, 2:], dim=-1)  # (n_steps, n_spheres)
    plt.figure()
    plt.plot(velocity[:,0])
    #%%

    import numpy as np
    from matplotlib import animation
    fig, ax = plt.subplots(figsize=(6,6), dpi=150)
    images_to_write = np.array(images_to_write)
    im = ax.imshow(images_to_write[0], cmap='viridis', vmin=0, vmax = images_to_write.max())
    ax.axis('off')

    def update_frame(frame):
        im.set_array(images_to_write[frame])
        return [im]

    ani = animation.FuncAnimation(fig, update_frame, frames=len(images_to_write), blit=True, interval=100)
    with tqdm(total=len(images_to_write), desc="Saving rollout animation") as pbar:
        callback = lambda *args: pbar.update()
        ani.save( save_path / 'rollout_simulation.mp4', writer='ffmpeg', progress_callback=callback, fps=30 )

    #%%

    if model.embedding_dim == 1:
        # create a line of latent vectors
        n_points = 20
        z = torch.linspace(0.8, 1.1, n_points).unsqueeze(-1).to(device)
        # z = torch.asarray([1.0]).to(device)

        with torch.no_grad():
            decoded_images = model.decoder(z).squeeze().cpu()

        print(decoded_images.shape)

        fig, axes = plt.subplots(1, n_points, figsize=(15, 3))
        for i in range(n_points):
            if n_points == 1:
                ax = axes
                im = decoded_images
            else:
                ax = axes[i]
                im = decoded_images[i]
            ax.imshow(im, cmap='gray')
            ax.axis('off')
        plt.tight_layout()
        plt.show()
    elif model.embedding_dim == 2:
        # create a grid of latent vectors
        # color circle by size
        n_points_per_axis = 10
        z2 = torch.linspace(0.5, 1.5, n_points_per_axis)
        z1 = torch.linspace(-1, 1.5, n_points_per_axis)
        z_grid = torch.stack(torch.meshgrid(z1, z2), dim=-1).reshape(-1, 2).to(device)

        with torch.no_grad():
            decoded_images = model.decoder(z_grid).squeeze().cpu()

        print(decoded_images.shape)

        fig, axes = plt.subplots(n_points_per_axis, n_points_per_axis, figsize=(15, 15))
        for i in range(n_points_per_axis):
            for j in range(n_points_per_axis):
                ax = axes[i, j]
                im = decoded_images[i * n_points_per_axis + j]
                ax.imshow(im, cmap='gray')
                ax.axis('off')
        plt.tight_layout()
        plt.show()

    #%% try to visualize the interaction term now

    # create a grid of latent vectors

    x0 = torch.tensor([0, 0], dtype=torch.float32).to(device)
    x1 = torch.tensor([1, 0], dtype=torch.float32).to(device)
    r0s = df_traj.r0.unique()
    r0s = np.sort(r0s)


    # plot the interaction as a function of distance
    distances = list(torch.linspace(0, 5, 100).to(device))
    import numpy as np
    interactions  = []
    from itertools import product
    r0_temp = r0s[0]
    bt = df_traj.query(f'r0 == {r0_temp} and growth_rate_percentage == 0.05 and time==0').sample(n=1, random_state=10)[['bt_0', 'bt_1']].values[0]
    b1s = [df_traj.query(f'r0 == {r0} and growth_rate_percentage == 0.05 and time==0').sample(n=1, random_state=10)[['bt_0', 'bt_1']].values[0] for r0 in r0s]
    for b1 in b1s:
        current_list = []
        for d in distances:
            x_i = x0.unsqueeze(0)
            x_j = (x0 + torch.tensor([d, 0], dtype=torch.float32).to(device)).unsqueeze(0)
            v0 = torch.tensor([[0.0, 0.0]], dtype=torch.float32).to(device)
            z_i = torch.cat([x_i, v0], dim=-1)
            z_j = torch.cat([x_j, v0], dim=-1)
            b0 = torch.tensor([bt], dtype=torch.float32).to(device)
            b1_temp = torch.tensor([b1], dtype=torch.float32).to(device)
            d_ij = torch.norm(x_i - x_j, dim=-1, keepdim=True)  # (1, 1) 

            input_interaction = torch.cat([z_i, z_j, b0, b1_temp, d_ij], dim=-1)
            input_interaction = input_interaction.squeeze()
            with torch.no_grad():
                interaction_ij = model.mlp_interaction(input_interaction)
            # interactions.append(interaction_ij.cpu())
            current_list.append(interaction_ij.cpu())
        interactions.append(current_list)


    # interactions = torch.stack(interactions).squeeze()
    distances_plot = torch.tensor(distances).cpu()

    with torch.no_grad():
        # print(interactions.shape)
        # scatter plot of distance vs interaction
        fig, ax = plt.subplots(figsize=(6,6))
        # set the color order
        ax.set_prop_cycle(plt.cycler('color', plt.cm.viridis(np.linspace(0, 1, len(b1s)))))

        for j, b1 in enumerate(b1s):
            interactions_b1 = torch.stack(interactions[j]).squeeze()
            if interactions_b1.dim() == 1:
                ax.plot(distances_plot, interactions_b1.cpu(), label=f"{r0s[j]}")
            elif interactions_b1.dim() == 2:
                plt.scatter(distances_plot, interactions_b1[:,0].cpu(), s=5, label=f"channel 0")
                plt.scatter(distances_plot, interactions_b1[:,1].cpu(), s=5, label=f"channel 1")
                plt.legend()
        ax.set_xlabel("Distance")
        ax.set_ylabel("Interaction")
        ax.set_title("Interaction vs Distance for $r_0=$" + str(r0_temp))
        cax = ax.inset_axes([1.05, 0, 0.05, 1])
        scalar = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=r0s.min(), vmax=r0s.max()))
        fig.colorbar(scalar, cax=cax, label='$r_0$')
