"""Build the seg-ablation sidecars for a dataset, with a pre-flight effect gate.

Usage:
    python scripts/build_seg_ablation.py +experiment=more_osc_clusters \
        +embedding=dino_sp +seg_ablation=moderate

Prints, before spending 150 epochs on the result:
  * realized per-kind rates and episode lengths (the planner drops an episode
    when the contact graph offers no free adjacent window, and on a sparse
    cache that shortfall is large),
  * the stratification audit — per-slot counts in both roles, and the
    correlation between each per-slot count and every varied simulation
    parameter, which is the confound the stratification exists to remove,
  * position / velocity SNR against the true motion,
  * a **pre-flight effect gate**: how far each kind moves the representation
    it acts through, against a between-cell reference. A kind below ~5% of
    that reference is not perturbing anything and should be cut or reclassified
    before it costs a full grid.

Patterned on scripts/generate_cache.py: no Hydra output directory, no MLflow.
"""

import os
import sys

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from graph_dyno.seg_ablation import ops
from graph_dyno.seg_ablation.build import ensure_sidecars, sidecar_paths
from graph_dyno.seg_ablation.plan import abl_id, normalize_spec

PROBE_CELLS = 192


# ---------------------------------------------------------------------------
# Pre-flight effect gate
# ---------------------------------------------------------------------------

@torch.no_grad()
def preflight(cfg, cache_dir, cache_id, spec, encoder, traj_idx=0, seed=0):
    """Measure how far each kind moves the representation it acts through.

    Each kind is reported in its *own* channel, because they do not all act on
    the embedding:

        partial / merge  ||de|| / between-cell reference   (shape + scale)
        scale            ||de|| / reference, plus |d crop_side| / sd. Both,
                         because the design originally assumed erode/dilate was
                         a pure scale perturbation; it is not (measured 0.33 at
                         2 px, 0.43 at 3 px), which is why v2 re-encodes it.
                         delta <= 1 px is a literal no-op at 128 resolution and
                         will show 0.000 here.
        miss             ||de|| / reference                (the empty-mask row)
        id_swap          teleport distance / median per-frame displacement
        centroid_noise   velocity SNR
    """
    rng = np.random.default_rng(seed)
    traj = torch.load(os.path.join(cache_dir, f"{cache_id}_{traj_idx:04d}.pt"),
                      map_location="cpu", mmap=True)
    images = traj["channeled_images"]
    states = traj["states"]
    W = traj["W_interactions"]
    T, N = images.shape[0], images.shape[1]
    r_max = float(traj["sim_metadata"]["r_max"])
    px = ops.scale_px(int(images.shape[-1]), r_max)
    pad_factor = int(spec["kinds"]["merge"]["pad_factor"])
    S = int(images.shape[-1])

    # Sample probe cells from frames where a contacting pair exists, so the
    # merge probe is measured on real adjacency rather than a fabricated one.
    idx = W._indices().numpy()
    sel = rng.choice(idx.shape[1], size=min(PROBE_CELLS, idx.shape[1]),
                     replace=False)
    ts, isrc, jsrc = idx[0][sel], idx[1][sel], idx[2][sel]

    clean, cut, blob, morphed = [], [], [], []
    d_crop = []
    for t, i, j in zip(ts, isrc, jsrc):
        m = (images[int(t), int(i)].numpy() > 0.5).astype(np.uint8)
        mj = (images[int(t), int(j)].numpy() > 0.5).astype(np.uint8)
        clean.append(ops.pad_to(m, pad_factor))
        k = spec["kinds"]["partial"]
        cut.append(ops.pad_to(ops.half_plane_cut(
            m, float(rng.uniform(0, np.pi)),
            float(rng.uniform(k["area_frac_min"], k["area_frac_max"])),
            float(rng.integers(0, 2))), pad_factor))
        off = (states[int(t), int(j), :2] - states[int(t), int(i), :2]).numpy() * px
        blob.append(ops.merge_pair(m, mj, off, pad_factor,
                                   spec['kinds']['merge']['close_px'])[0])
        delta = float(spec["kinds"]["scale"]["delta_px"])
        mm = ops.morph_mask(m, delta)
        morphed.append(ops.pad_to(mm, pad_factor))
        d_crop.append(ops.crop_side_of(mm) - ops.crop_side_of(m))

    def enc(stack):
        e, c = encoder.encode_masks(
            torch.from_numpy(np.stack(stack).astype(np.uint8)), batch_size=64)
        return e.float().cpu().numpy(), c.cpu().numpy()

    e_clean, c_clean = enc(clean)
    e_cut, _ = enc(cut)
    e_blob, _ = enc(blob)
    e_morph, _ = enc(morphed)
    e_empty, _ = enc([np.zeros((pad_factor * S, pad_factor * S), np.uint8)])

    # Between-cell reference: how far apart two *different* cells sit in the
    # same space. Everything is reported as a fraction of this.
    perm = rng.permutation(len(e_clean))
    ref = float(np.linalg.norm(e_clean - e_clean[perm], axis=-1).mean())

    def frac(a, b):
        return float(np.linalg.norm(a - b, axis=-1).mean() / max(ref, 1e-12))

    pos = states[..., :2].numpy().astype(np.float64)
    med_disp_px = float(np.median(np.linalg.norm(pos[1:] - pos[:-1], axis=-1)) * px)
    speed = float(np.mean(np.linalg.norm(
        states[..., 2:4].numpy(), axis=-1)))
    dt = float(cfg.data.dt)
    sigma_units = spec["centroid_noise_sigma_rel"] * med_disp_px / px

    # Teleport size of an ID swap: how far the two cells of a contacting pair
    # actually are, in units of a per-frame step.
    jump = np.linalg.norm(
        (states[ts, jsrc, :2] - states[ts, isrc, :2]).numpy(), axis=-1)

    return {
        "reference": ref,
        "partial": frac(e_cut, e_clean),
        "merge": frac(e_blob, e_clean),
        "miss": frac(np.repeat(e_empty, len(e_clean), axis=0), e_clean),
        "scale_dL": frac(e_morph, e_clean),
        "scale_dcrop_sd": float(np.mean(np.abs(d_crop)) / max(c_clean.std(), 1e-12)),
        "id_swap_jump_disp": float(jump.mean() * px / max(med_disp_px, 1e-12)),
        "median_disp_px": med_disp_px,
        "mean_speed": speed,
        "vel_noise_sigma": sigma_units * np.sqrt(2.0) / dt,
        "vel_snr": (speed / (sigma_units * np.sqrt(2.0) / dt))
        if sigma_units > 0 else float("inf"),
    }


def _report_preflight(pf, spec):
    print("\n" + "=" * 72)
    print("PRE-FLIGHT EFFECT GATE  (between-cell embedding reference "
          f"= {pf['reference']:.3f})")
    print("=" * 72)
    rows = [
        ("partial", spec["kinds"]["partial"]["rate"], f"{pf['partial']:.3f}",
         "||dE|| / between-cell"),
        ("merge", spec["kinds"]["merge"]["rate"], f"{pf['merge']:.3f}",
         "||dE|| / between-cell"),
        ("miss", spec["kinds"]["miss"]["rate"], f"{pf['miss']:.3f}",
         "||dE|| / between-cell (empty-mask row)"),
        ("scale", spec["kinds"]["scale"]["rate"], f"{pf['scale_dL']:.3f}",
         f"||dE|| / between-cell  (|d crop_side| / sd = "
         f"{pf['scale_dcrop_sd']:.3f})"),
        ("id_swap", spec["kinds"]["id_swap"]["rate"],
         f"{pf['id_swap_jump_disp']:.1f}",
         "teleport / median per-frame displacement"),
        ("centroid_noise", spec["centroid_noise_sigma_rel"],
         f"{pf['vel_snr']:.2f}", "velocity SNR (true speed / noise sigma)"),
    ]
    print(f"{'kind':15s} {'rate':>7s} {'effect':>8s}   channel")
    for name, rate, eff, chan in rows:
        flag = ""
        if rate and name in ("partial", "merge", "miss", "scale") and float(eff) < 0.05:
            flag = "  <-- BELOW 5% OF REFERENCE: not perturbing the representation"
        if not rate:
            flag = "  (off)"
        print(f"{name:15s} {rate:>7.4g} {eff:>8s}   {chan}{flag}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    from graph_dyno.encoders.factory import get_companion_suffix, create_shape_encoder
    from graph_dyno.models.model_factory import resolve_shape_config
    from graph_dyno.data.trajectory_generator import generate_config_hash

    if cfg.get("seg_ablation", None) is None:
        raise SystemExit("pass +seg_ablation=<level> (see configs/seg_ablation/)")

    resolve_shape_config(cfg)
    spec = normalize_spec(cfg.seg_ablation)
    abl = abl_id(spec)
    cache_id = generate_config_hash(cfg.data)
    cache_dir = cfg.data.generator.cache_dir
    companion_suffix = get_companion_suffix(cfg)
    n_traj = cfg.data.generator.n_trajectories
    traj_indices = list(range(n_traj))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        p = torch.cuda.get_device_properties(0)
        print(f"GPU: {p.name} ({round(p.total_memory / 1024 ** 3)} GB)")

    print(f"\ncache_dir       {cache_dir}")
    print(f"cache_id        {cache_id}   (unchanged by seg_ablation, by design)")
    print(f"abl_id          {abl}")
    print(f"companion       {companion_suffix}")
    print(OmegaConf.to_yaml(OmegaConf.create(spec)))

    if cfg.get("preflight", True):
        encoder = create_shape_encoder(cfg, device=device)
        pf = preflight(cfg, cache_dir, cache_id, spec, encoder)
        _report_preflight(pf, spec)
        del encoder
        torch.cuda.empty_cache()

    stats = ensure_sidecars(cache_dir, cfg, cache_id=cache_id,
                            companion_suffix=companion_suffix,
                            traj_indices=traj_indices, device=device,
                            force=cfg.get("force", False))

    print("\n" + "=" * 72)
    print("REALIZED")
    print("=" * 72)
    for s in stats:
        if s is None:
            continue
        print(f"\ntrajectory {s['traj_idx']}  ({s['n_events']} episodes, "
              f"{s['n_reencoded']} re-encoded, "
              f"{s['n_invalid_cell_frames']} invalid cell-frames = "
              f"{s['invalid_frac'] * 100:.2f}%)")
        for kind, d in s["per_kind"].items():
            if not d.get("n_episodes"):
                continue
            line = (f"  {kind:9s} rate {d['realized_rate']:.4f}  "
                    f"episodes {d['n_episodes']}/{d['n_requested']} requested  "
                    f"len {d.get('mean_len', 0):.1f}  "
                    f"spread init {d.get('slot_count_spread', 0)}")
            if "partner_count_spread" in d:
                # `slots touched` matters more than the spread when there are
                # fewer episodes than slots: a realistic ID-swap rate gives ~10
                # episodes over ~380 eligible slots, so most slots getting
                # nothing is arithmetic, not a sampler failure.
                line += (f" partner {d['partner_count_spread']}"
                         f"  slots touched {d.get('n_slots_touched', 0)}"
                         f"/{d.get('n_feasible_slots', 0)}"
                         f" ({d.get('n_infeasible_slots', 0)} infeasible"
                         + (f", {d['n_stranded_slots']} STRANDED"
                            if d.get("n_stranded_slots") else "") + ")")
            print(line)
        print(f"  edges dropped {(1 - s['edges_kept_frac']) * 100:.2f}%   "
              f"centroid sigma {s['centroid_sigma_px']:.3f} px   "
              f"velocity SNR {s['vel_snr']:.2f}   "
              f"merge clipped {s['n_merge_clipped']}")
        w = s.get("worst_label_correlation")
        if w is not None:
            print(f"  worst per-slot/label correlation: "
                  f"r({w[0]}, {w[1]}) = {w[2]:+.3f}  "
                  f"(limit {spec['max_label_correlation']})")

    for i in traj_indices:
        g, e = sidecar_paths(cache_dir, abl, companion_suffix, cache_id, i)
        print(f"\n  {g}  ({os.path.getsize(g) / 1e6:.0f} MB)"
              f"\n  {e}  ({os.path.getsize(e) / 1e6:.0f} MB)")


if __name__ == "__main__":
    sys.argv.extend([
        "hydra.run.dir=.",
        "hydra.output_subdir=null",
        "hydra/job_logging=none",
    ])
    main()
