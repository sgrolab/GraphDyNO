"""GraphModelReal — GraphModelDINO with masked losses for real (CTC) data.

Real cells appear, divide and disappear, so a slot is only a real cell on part
of its trajectory. `valid` marks the real ones; every loss must be masked or
padded slots would train the model to predict `PAD_POS`.

This is a **pure subclass with method overrides**: `graph_model_dino.py` gets a
zero-diff. Extracting a `_compute_losses` hook in the parent would remove ~35
duplicated lines and is technically safe for checkpoint loading, but "old
results still load" in practice means nobody should have to re-audit the file
that produced the published numbers.

Three distinct masks; using the wrong one is the easiest bug here:

    valid[t]                  slot is a real cell — edge filtering, L2 mask
    valid[t] & valid[t+1]     next-step targets exist
    valid[t-1] & valid[t]     inputs well-defined (velocity is f(t-1, t))

`loss_mask_mode: strict` (the default) uses the conjunction of all three.
`x[..., 2:4]` (kinematic velocity) is *always* an input — it feeds `x_features`
and the residual `new_vel = x[..., 2:4] + dv` — so a fabricated birth-frame zero
velocity contaminates position/velocity prediction whether or not
`use_velocities` is on. Cost is ~2 frames per track. `transition` mode
(`valid & next_valid`) is the escape hatch for very short tracks.
"""

import torch

from graph_dyno.models.graph_model_dino import GraphModelDINO

LOSS_MASK_MODES = ("strict", "transition")


class GraphModelReal(GraphModelDINO):
    """Graph model over real microscopy data with variable cell counts."""

    def __init__(self, cfg, shape_decoder, device: str | None = None):
        super().__init__(cfg, shape_decoder=shape_decoder, device=device)

        self.loss_mask_mode = cfg.model.get("loss_mask_mode", "strict")
        if self.loss_mask_mode not in LOSS_MASK_MODES:
            raise ValueError(
                f"model.loss_mask_mode must be one of {LOSS_MASK_MODES}, "
                f"got {self.loss_mask_mode!r}"
            )

        # Absolute position must stay structurally unavailable to every
        # prediction head, or the FOV extent of each video leaks straight into
        # the embedding and fakes the separation result. states are
        # [x, y, vx, vy], so indices_of_mlp_features: [2, 3] means x_features
        # is *velocities*; gnn.py feeds only relative x_ij and d_ij. This
        # property is load-bearing, hence the assert.
        # `model.allow_absolute_position: true` opts out, for deliberately
        # probing what position adds. Any separation number from such a run is
        # NOT comparable to a guarded run: PhC-C2DL-PSC's FOV is 576x720 and
        # Fluo-N2DL-HeLa's is 700x1100, and center_fov puts both at
        # domain_size/2, so radial distance from centre alone partly labels the
        # dataset. Treat the result as an upper bound contaminated by FOV
        # geometry, not as evidence of learned separation.
        feature_indices = set(int(i) for i in cfg.model.indices_of_mlp_features)
        leaked = feature_indices & {0, 1}
        if leaked and not cfg.model.get("allow_absolute_position", False):
            raise ValueError(
                f"model.indices_of_mlp_features includes absolute position "
                f"{sorted(leaked)}. On real data every video sits at a "
                f"different FOV extent, so absolute position is a trivial "
                f"per-video fingerprint that would fake the separation result. "
                f"Set model.allow_absolute_position=true to override "
                f"deliberately."
            )
        if leaked:
            print(f"[GraphModelReal] WARNING: absolute position "
                  f"{sorted(leaked)} is an MLP input (allow_absolute_position). "
                  f"Per-video FOV extent leaks into the embedding — separation "
                  f"metrics from this run are contaminated by construction.")
            # Position is ~1000x the magnitude of velocity in domain units
            # (measured on ctc_mix_b_late: mean |x| 556 vs mean |v| 0.55), which
            # swamps the first MLP layer: the first attempt at this left
            # `parameter_embedding` at its initialisation (per-dim std 0.100,
            # probe AUC 0.491 = chance) with train loss 4.65 vs 0.80 for the
            # velocity-only run. Map position to [-0.5, 0.5] and leave every
            # other channel alone, unless the config states values explicitly.
            if cfg.model.get("feature_scale", None) is None and \
                    cfg.model.get("feature_offset", None) is None:
                scale, offset = [], []
                for i in cfg.model.indices_of_mlp_features:
                    is_pos = int(i) < self.spatial_dim
                    scale.append(1.0 / self.domain_size if is_pos else 1.0)
                    offset.append(self.domain_size / 2.0 if is_pos else 0.0)
                self.feature_scale = torch.as_tensor(
                    scale, dtype=torch.float32, device=self.device)
                self.feature_offset = torch.as_tensor(
                    offset, dtype=torch.float32, device=self.device)
                print(f"[GraphModelReal] normalising x_features: "
                      f"offset={offset} scale={[round(s, 6) for s in scale]}")

        # augmentation.rotate_sample rotates states and channeled_images about
        # domain_size/2 with no notion of validity: it moves cells outside the
        # imaged region of a rectangular FOV, rotates PAD_POS slots to arbitrary
        # in-canvas positions, and never touches `valid`.
        if cfg.get("augmentation", None) is not None \
                and cfg.augmentation.get("enabled", False):
            raise ValueError(
                "augmentation is not usable on real data (rotate_sample is "
                "unaware of `valid` and of the rectangular FOV). "
                "Set augmentation.enabled=false."
            )

        # Rollout reuses step-0 edges for the whole k-step window
        # (dataloader ships only W_interactions[timestep_idx]). A slot that
        # goes invalid at step 2 keeps its step-0 edges, and its
        # autoregressively-diverging state feeds x_ij/d_ij into *real*
        # neighbours' messages. The "padded slots have no incident edges =>
        # harmless" argument holds only for the single-step path. Masked losses
        # do not fix corrupted inputs. Enabling rollout here requires per-step
        # edge re-masking, which this implementation does not carry.
        if cfg.get("training", None) is not None \
                and cfg.training.get("rollout", {}).get("enabled", False):
            print("[GraphModelReal] WARNING: training.rollout.enabled=true. "
                  "Rollout reuses step-0 edges for the whole window, so slots "
                  "that die mid-window keep stale edges and corrupt their "
                  "neighbours' messages. Masked losses do not fix that — "
                  "enabling rollout needs per-step edge re-masking.")

    # ------------------------------------------------------------------
    # Masked reductions
    # ------------------------------------------------------------------

    @staticmethod
    def _masked_mse(pred, target, mask):
        """MSE over masked entries only. mask is (B, N) bool.

        With an all-True mask this is exactly `F.mse_loss` — the guarantee that
        nothing about the simulation path changes.
        """
        m = mask.unsqueeze(-1).to(pred.dtype)
        return ((pred - target) ** 2 * m).sum() / (m.sum() * pred.shape[-1]).clamp_min(1.0)

    def _masked_embedding_l2(self, seen):
        """Mean squared embedding norm over slots this batch actually supervises.

        With disjoint slots and windowing, slot i is *supervised* in only a
        fraction of batches, but an unmasked mean(pe**2) regularises it in
        *every* batch — so rarely-seen slots get systematically over-shrunk and
        the embedding-norm gradient becomes a direct function of video size.
        That is a video-correlated artifact in the exact quantity being
        clustered. (The default config sets this term to 0, so this is
        defensive — but defensive in the direction that matters.)
        """
        pe = self.parameter_embedding
        if seen is None:
            return torch.mean(pe ** 2)
        w = seen.to(pe.dtype).unsqueeze(-1)
        return (pe ** 2 * w).sum() / (w.sum() * pe.shape[-1]).clamp_min(1.0)

    def _ones_mask(self, ref):
        return torch.ones(ref.shape[:2], dtype=torch.bool, device=ref.device)

    def _step_mask(self, batch, ref, prev_key, cur_key, next_key):
        """Build the loss mask for one prediction step.

        Falls back to all-True when the batch carries no validity keys, so the
        class still behaves exactly like GraphModelDINO on simulation caches.
        """
        cur = batch.get(cur_key, None)
        if cur is None:
            return self._ones_mask(ref)
        mask = cur & batch[next_key]
        if self.loss_mask_mode == "strict":
            prev = batch.get(prev_key, None)
            if prev is not None:
                mask = mask & prev
        return mask

    def loss_mask_for(self, batch: dict) -> torch.Tensor:
        """The single-step mask `training_step` will apply, exposed read-only.

        The Risk 10 check in train.py needs this to tell two very different
        things apart: a slot that never had a trainable transition at all (a
        data property — a short track landing at a window edge, where
        prev/cur/next are never all valid) from a slot that had one and still
        received no gradient (a wiring bug). `valid` alone cannot distinguish
        them, because `valid` is strictly broader than this mask.
        """
        return self._step_mask(batch, batch["states"],
                               "prev_valid", "valid", "next_valid")

    # ------------------------------------------------------------------
    # Training steps (mirrors GraphModelDINO, with masked losses)
    # ------------------------------------------------------------------

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

        return {
            "position_mse": self._masked_mse(
                new_x[..., :S], batch["next_states"][..., :S], mask,
            ),
            "velocity_mse": self._masked_mse(
                new_x[..., S:S * 2], batch["next_states"][..., S:S * 2], mask,
            ),
            "dino_mse": self._masked_mse(L_out, target_next, mask),
            # (B, N) tensors — unsqueeze both sides so the shape[-1] denominator
            # stays 1 and the value matches the unmasked mse_loss convention.
            "scale_mse": self._masked_mse(
                (next_scale / self.image_size).unsqueeze(-1),
                (batch["next_crop_sides"].float() / self.image_size).unsqueeze(-1),
                mask,
            ),
            "parameter_embedding_l2": self._masked_embedding_l2(seen),
        }

    def rollout_training_step(self, batch: dict, rollout_steps: int) -> dict:
        """Masked multi-step rollout.

        NOTE: the losses are masked, but the *inputs* are not corrected — step-0
        edges are reused for the whole window (see the warning in __init__).
        Keep `training.rollout.enabled: false` on real data.
        """
        if rollout_steps == 1:
            return self.training_step(batch)

        W_interactions = batch["W_interactions"]

        all_L = batch["L_seq"]
        all_L_target = batch["next_L_seq"]

        x = batch["states_seq"][:, 0]
        L = all_L[:, 0]
        current_scale = batch["crop_sides_seq"][:, 0].float()

        x, L = self._noise_inputs(x, L, batch["next_states_seq"][:, 0], all_L_target[:, 0])

        if self.use_velocities:
            L_velocity = L - batch["prev_L"]
            scale_velocity = current_scale - batch["prev_crop_sides"].float()
        else:
            L_velocity = None
            scale_velocity = None

        L_history = None
        if self.use_history:
            L_history = batch["L_history"].transpose(1, 2).contiguous()

        valid_seq = batch.get("valid_seq", None)
        next_valid_seq = batch.get("next_valid_seq", None)
        seen = valid_seq.any(dim=1) if valid_seq is not None else None

        loss_accum = {"parameter_embedding_l2": self._masked_embedding_l2(seen)}
        S = self.spatial_dim

        for step in range(rollout_steps):
            new_x, L_out, next_scale = self.forward(x, W_interactions, L, current_scale,
                                                    L_velocity, scale_velocity, L_history)

            if valid_seq is None:
                mask = self._ones_mask(x)
            else:
                mask = valid_seq[:, step] & next_valid_seq[:, step]
                if self.loss_mask_mode == "strict":
                    prev = batch["prev_valid"] if step == 0 else valid_seq[:, step - 1]
                    mask = mask & prev

            step_losses = {
                "position_mse": self._masked_mse(
                    new_x[..., :S],
                    batch["next_states_seq"][:, step, ..., :S], mask,
                ),
                "velocity_mse": self._masked_mse(
                    new_x[..., S:S * 2],
                    batch["next_states_seq"][:, step, ..., S:S * 2], mask,
                ),
                "dino_mse": self._masked_mse(L_out, all_L_target[:, step], mask),
                "scale_mse": self._masked_mse(
                    (next_scale / self.image_size).unsqueeze(-1),
                    (batch["next_crop_sides_seq"][:, step].float()
                     / self.image_size).unsqueeze(-1),
                    mask,
                ),
            }

            for k, v in step_losses.items():
                loss_accum[k] = loss_accum.get(k, 0.0) + v

            if self.use_velocities:
                L_velocity = L_out - L
                scale_velocity = next_scale - current_scale

            if self.use_history:
                L_history = torch.cat(
                    [L_history[:, :, 1:, :], L.detach().unsqueeze(2)], dim=2
                )

            x = new_x
            L = L_out
            current_scale = next_scale

        # Same averaging convention as GraphModelDINO.rollout_training_step
        # (which also divides the embedding L2 term).
        return {k: v / rollout_steps for k, v in loss_accum.items()}
