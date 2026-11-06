"""GraphModelSegAblation — GraphModelReal with bounded position/velocity losses.

Selected by `cfg.model.type: segablation`. `unified_dino.yaml` has no `type`
key, so every existing run and checkpoint is bit-identical and
`graph_model_dino.py` / `graph_model_real.py` get a zero diff.

Why a separate class at all
---------------------------
The masked losses (`GraphModelReal._masked_mse`, `_step_mask`) are exactly what
the oracle-masked-miss convention needs, and `GraphModelDINO` ignores `valid`
entirely (`training_step:450-495` uses plain `mse_loss`). But the paper's
`loss_terms` weight `position_mse` / `velocity_mse` / `dino_mse` / `scale_mse`
at 1.0 each, and against the FINISHED baselines
(`val_position_mse = 2e-6`, `val_velocity_mse = 0.0174`, `val_dino_mse = 0.021`)
a single ID-swapped transition teleports the observed position ~60 px = 3.2
units, contributing `3.2^2 ~ 10` to `position_mse` and `(3.2/0.01)^2 ~ 1.0e5` to
`velocity_mse`. Two consequences:

1. `dino_mse` — the term that actually shapes `parameter_embedding` — becomes
   numerically invisible next to `velocity_mse`.
2. `configs/training/default.yaml:12` sets `grad_clip_norm: 10.0`, applied
   unconditionally at `train.py:501`. Every ablated arm would train permanently
   clipped while `none` never clips, so the arms would differ in *effective
   step size* and any measured separation drop would be partly an optimizer
   artifact.

So position and velocity use a Huber loss that is **exactly** `d^2` inside
`beta` and linear outside:

    huber(d) = d^2                    |d| <= beta
             = 2*beta*|d| - beta^2    otherwise

which bounds each transition's contribution and its gradient while keeping the
blind condition intact. It applies to **every arm including `none`**, so the
ladder is internally matched — and with an all-True `valid`, `_masked_mse` is
exactly `F.mse_loss` (`graph_model_real.py:135-137`), so `none` carries no
model-class confound beyond the (documented, matched) Huber.
"""

import torch

from graph_dyno.models.graph_model_real import GraphModelReal

#: Defaults from the measured baselines: position residual RMS ~1.4e-3 units
#: and velocity residual RMS ~0.13 units/time, so these sit ~35x above a
#: typical clean residual — never active in normal training, and a ~30x cut on
#: a swapped transition.
DEFAULT_HUBER_BETA_POSITION = 0.05
DEFAULT_HUBER_BETA_VELOCITY = 5.0


def huber(pred, target, beta):
    """Quadratic inside `beta`, linear outside, continuous in value and slope.

    Reduces to `(pred - target) ** 2` exactly for `|d| <= beta`, which is what
    keeps the `none` arm numerically comparable to the published plain-MSE runs.
    """
    d = pred - target
    a = d.abs()
    return torch.where(a <= beta, d * d, 2.0 * beta * a - beta * beta)


class GraphModelSegAblation(GraphModelReal):
    """Graph model over *observed* (segmentation-corrupted) simulation data."""

    def __init__(self, cfg, shape_decoder, device: str | None = None):
        m = cfg.model
        # Two guards in GraphModelReal are real-data-specific and must not fire
        # here, so the parent runs against a patched *copy* — the caller's cfg
        # is never mutated, and neither graph_model_real.py nor
        # graph_model_dino.py changes.
        #
        #  1. `allow_absolute_position`: the parent refuses positions as MLP
        #     inputs because on real data every video sits at a different FOV
        #     extent. Here every arm shares one simulation domain.
        #  2. `feature_scale` / `feature_offset`: with the guard opted out, the
        #     parent would *derive* a domain-size normalisation the paper's
        #     GraphModelDINO runs never had. Pinning them to ones/zeros keeps
        #     `x_features` bit-identical to the published model.
        super().__init__(_patched_cfg(cfg), shape_decoder=shape_decoder,
                         device=device)

        self.huber_beta_position = float(
            m.get("huber_beta_position", DEFAULT_HUBER_BETA_POSITION))
        self.huber_beta_velocity = float(
            m.get("huber_beta_velocity", DEFAULT_HUBER_BETA_VELOCITY))
        print(f"[GraphModelSegAblation] Huber betas: "
              f"position={self.huber_beta_position} "
              f"velocity={self.huber_beta_velocity}")

        # Rollout ships only `W_interactions[timestep_idx]` and reuses those
        # step-0 edges for the whole window (`graph_model_real.py:109-123`).
        # Misses and merges make slots go invalid *mid-window*, so a dead slot
        # keeps stale edges and feeds its autoregressively-diverging state into
        # real neighbours' messages. On real data that is a warning; here it is
        # a refusal, because every ablated arm would hit it.
        if cfg.get("training", None) is not None \
                and cfg.training.get("rollout", {}).get("enabled", False):
            raise ValueError(
                "training.rollout.enabled=true is not supported under "
                "seg_ablation: rollout reuses step-0 edges for the whole "
                "window, and miss/merge make slots go invalid mid-window. Set "
                "training.rollout.enabled=false."
            )
        if cfg.get("augmentation", None) is not None \
                and cfg.augmentation.get("enabled", False):
            raise ValueError(
                "augmentation.enabled=true is not supported under seg_ablation "
                "(rotate_sample is unaware of `valid`). Set "
                "augmentation.enabled=false."
            )

        # `loss_mask_mode: transition` drops the prev-frame term, which is what
        # keeps a fabricated boundary velocity out of the trained inputs. A
        # missed slot's reported position is `pad_pos`, so the finite-difference
        # velocity at a miss boundary is ~2000x the true mean speed; under
        # `strict` those transitions are all masked out, under `transition`
        # they are not.
        spec = cfg.get("seg_ablation", None)
        if spec is not None and self.loss_mask_mode != "strict":
            from graph_dyno.seg_ablation.plan import normalize_spec
            s = normalize_spec(spec)
            if s["kinds"]["miss"]["rate"] > 0 or s["kinds"]["merge"]["rate"] > 0:
                raise ValueError(
                    "seg_ablation with miss/merge requires "
                    "model.loss_mask_mode='strict': a missed slot reports "
                    "pad_pos, so the boundary velocity is ~2000x the true mean "
                    "speed and only `strict` masks those transitions out."
                )

        # Filled by train.py from the sidecar so the checkpoint and MLflow both
        # record which ablation produced it.
        self.seg_ablation_meta: dict = {}

    # ------------------------------------------------------------------

    def _masked_huber(self, pred, target, mask, beta):
        m = mask.unsqueeze(-1).to(pred.dtype)
        per = huber(pred, target, beta)
        loss = (per * m).sum() / (m.sum() * pred.shape[-1]).clamp_min(1.0)
        with torch.no_grad():
            linear = (((pred - target).abs() > beta).to(pred.dtype) * m).sum()
            frac = linear / (m.sum() * pred.shape[-1]).clamp_min(1.0)
        return loss, frac

    def training_step(self, batch: dict) -> dict:
        x = batch["states"]
        W_interactions = batch["W_interactions"]

        L = batch["L"]
        target_next = batch["next_L"]
        current_scale = batch["crop_sides"].float()

        x, L = self._noise_inputs(x, L, batch["next_states"], target_next)

        L_velocity = None
        scale_velocity = None
        if self.use_velocities:
            L_velocity = L - batch["prev_L"]
            scale_velocity = current_scale - batch["prev_crop_sides"].float()

        L_history = None
        if self.use_history:
            L_history = batch["L_history"].transpose(1, 2).contiguous()

        new_x, L_out, next_scale = self.forward(x, W_interactions, L, current_scale,
                                                L_velocity, scale_velocity, L_history)

        mask = self._step_mask(batch, x, "prev_valid", "valid", "next_valid")
        seen = batch["valid"] if "valid" in batch else None
        S = self.spatial_dim

        pos_loss, pos_lin = self._masked_huber(
            new_x[..., :S], batch["next_states"][..., :S], mask,
            self.huber_beta_position)
        vel_loss, vel_lin = self._masked_huber(
            new_x[..., S:S * 2], batch["next_states"][..., S:S * 2], mask,
            self.huber_beta_velocity)

        return {
            "position_mse": pos_loss,
            "velocity_mse": vel_loss,
            "dino_mse": self._masked_mse(L_out, target_next, mask),
            "scale_mse": self._masked_mse(
                (next_scale / self.image_size).unsqueeze(-1),
                (batch["next_crop_sides"].float() / self.image_size).unsqueeze(-1),
                mask,
            ),
            "parameter_embedding_l2": self._masked_embedding_l2(seen),
            # Diagnostics: not in `loss_terms`, so train.py never adds them to
            # the objective, but they do reach MLflow as `val_*`.
            "position_huber_linear_frac": pos_lin,
            "velocity_huber_linear_frac": vel_lin,
            "supervised_frac": mask.to(x.dtype).mean().detach(),
        }

    def clean_target_losses(self, batch: dict) -> dict:
        """Val losses against the CLEAN targets, for honest model selection.

        `train.py:622-625` saves `graph_decoder_model_best.pth` at
        `argmin(avg_val_loss)`, and with a val loss dominated by unpredictable
        corrupted transitions the argmin epoch is driven by noise, not
        embedding quality. The seg-ablation analysis therefore reads the FINAL
        checkpoint; this is the diagnostic that says whether that mattered.

        Inputs stay corrupted — the model is blind. Only the targets are the
        truth. Returns {} when the batch carries no clean targets.
        """
        if "next_L_clean" not in batch:
            return {}
        clean = {
            "next_states": batch["next_states_clean"],
            "next_L": batch["next_L_clean"],
            "next_crop_sides": batch["next_crop_sides_clean"],
        }
        x = batch["states"]
        L = batch["L"]
        current_scale = batch["crop_sides"].float()

        L_velocity = None
        scale_velocity = None
        if self.use_velocities:
            L_velocity = L - batch["prev_L"]
            scale_velocity = current_scale - batch["prev_crop_sides"].float()

        L_history = None
        if self.use_history:
            L_history = batch["L_history"].transpose(1, 2).contiguous()

        new_x, L_out, next_scale = self.forward(
            x, batch["W_interactions"], L, current_scale,
            L_velocity, scale_velocity, L_history)

        mask = self._step_mask(batch, x, "prev_valid", "valid", "next_valid")
        S = self.spatial_dim
        pos, _ = self._masked_huber(new_x[..., :S], clean["next_states"][..., :S],
                                    mask, self.huber_beta_position)
        vel, _ = self._masked_huber(new_x[..., S:S * 2],
                                    clean["next_states"][..., S:S * 2], mask,
                                    self.huber_beta_velocity)
        return {
            "clean_position_mse": pos,
            "clean_velocity_mse": vel,
            "clean_dino_mse": self._masked_mse(L_out, clean["next_L"], mask),
            "clean_scale_mse": self._masked_mse(
                (next_scale / self.image_size).unsqueeze(-1),
                (clean["next_crop_sides"].float() / self.image_size).unsqueeze(-1),
                mask,
            ),
        }

    def rollout_training_step(self, batch: dict, rollout_steps: int) -> dict:
        if rollout_steps == 1:
            return self.training_step(batch)
        raise RuntimeError(
            "rollout is refused under seg_ablation (see __init__)")


def _patched_cfg(cfg):
    """A copy of `cfg` with GraphModelReal's real-data-only guards opted out."""
    from omegaconf import OmegaConf, open_dict

    out = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    with open_dict(out):
        out.model.allow_absolute_position = True
        n_feat = len(cfg.model.indices_of_mlp_features)
        if cfg.model.get("feature_scale", None) is None:
            out.model.feature_scale = [1.0] * n_feat
        if cfg.model.get("feature_offset", None) is None:
            out.model.feature_offset = [0.0] * n_feat
    return out
