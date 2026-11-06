#%%
import torch
from torch.nn.functional import mse_loss
from omegaconf import DictConfig
from graph_dyno.models.mlp import MLP
from graph_dyno.models.gnn import GNN
from graph_dyno.decoders.base import ShapeDecoder


class GraphModelDINO(torch.nn.Module):
    """Graph neural network model using frozen shape decoder bottleneck representations.

    Shape representations come from a frozen pretrained ShapeDecoder:
    - get_bottleneck() maps raw embeddings to bottleneck vectors
    - shape_mlp evolves bottleneck vectors forward in time
    - No learned shape embedding, no rotation augmentation
    - Image decoding is a visualization-only utility (not in training loop)
    """

    def __init__(self, cfg: DictConfig, shape_decoder: ShapeDecoder, device: str | None = None):
        super().__init__()
        m = cfg.model

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        seed = m.get("seed", 0)
        torch.manual_seed(seed)
        self._noise_gen = torch.Generator(device=self.device)
        self._noise_gen.manual_seed(seed)

        # Check for periodic boundary conditions from data config
        self.periodic_boundary = False
        self.domain_size = cfg.data.domain_size
        if cfg.data.type == "simulation":
            self.periodic_boundary = cfg.data.params.periodic_boundary

        # Store common attributes
        self.embedding_dim = m.embedding_dim
        self.num_nodes = m.num_nodes
        self.spatial_dim = m.spatial_dim
        self.skip_connection = m.skip_connection
        # L (bottleneck) skip can be disabled independently to avoid compounding
        # rollout error; falls back to skip_connection for back-compat.
        self.L_skip_connection = m.get("L_skip_connection", m.skip_connection)
        self.embedding_noise_std = m.embedding_noise_std
        self.input_noise_std = m.get("input_noise_std", 0.0)
        if self.input_noise_std > 0:
            print(f"[GraphModelDINO] input_noise_std={self.input_noise_std} "
                  f"(noise on x, v, L scaled by per-group displacement-magnitude std)")
        # Per-cell whole-L dropout applied to shape-MLP inputs only (shallow).
        # Forces per-cell b to carry signal when L is masked.
        self.L_dropout_prob = m.get("L_dropout_prob", 0.0)
        if self.L_dropout_prob > 0:
            print(f"[GraphModelDINO] L_dropout_prob={self.L_dropout_prob} "
                  f"(per-cell whole-L drop to shape MLPs only)")
        # Per-cell whole-scale dropout applied to scale-MLP inputs only.
        # Forces per-cell b/interactions to carry signal when scale is masked.
        self.scale_dropout_prob = m.get("scale_dropout_prob", 0.0)
        if self.scale_dropout_prob > 0:
            print(f"[GraphModelDINO] scale_dropout_prob={self.scale_dropout_prob} "
                  f"(per-cell whole-scale drop to scale MLPs only)")

        # Dataset-level displacement-magnitude std (scalars) for positions,
        # velocities, and bottleneck. Populated externally before training;
        # None → fall back to per-batch estimate.
        self.x_disp_std: torch.Tensor | None = None
        self.v_disp_std: torch.Tensor | None = None
        self.L_disp_std: torch.Tensor | None = None
        self.dt = cfg.data.dt
        self.image_size = cfg.data.image_size

        # Convert activation string to torch activation module
        activation_map = {
            "ReLU": torch.nn.ReLU,
            "PReLU": torch.nn.PReLU,
            "LeakyReLU": torch.nn.LeakyReLU,
            "ELU": torch.nn.ELU,
            "GELU": torch.nn.GELU,
            "Tanh": torch.nn.Tanh,
            "Sigmoid": torch.nn.Sigmoid,
        }
        if m.activation not in activation_map:
            raise ValueError(f"Unknown activation: {m.activation}. Available: {list(activation_map.keys())}")
        activation_module = activation_map[m.activation]

        # --- Frozen shape decoder (created externally via factory) ---
        self.shape_decoder = shape_decoder
        self.shape_decoder.to(self.device)
        self.shape_decoder.eval()
        for p in self.shape_decoder.parameters():
            p.requires_grad_(False)

        self.dino_dim = self.shape_decoder.input_dim
        self.bottleneck_dim = self.shape_decoder.bottleneck_dim

        # Past-K-frame bottleneck history fed as a flattened (K*bn_dim) vector
        # to any MLP that lists 'L_history' in its inputs. 0 disables it.
        self.history_window = int(m.get("history_window", 0))

        # --- Config-driven MLP input vectors ---
        # Each name maps to a tensor dimension; MLPs are built by summing dims.
        self._vector_dims = {
            'b': m.embedding_dim,
            'L': self.bottleneck_dim,
            'scale': 1,
            'interactions': m.interaction_dim,
            'x_features': len(m.indices_of_mlp_features),
            'L_velocity': self.bottleneck_dim,
            'scale_velocity': 1,
            'L_history': self.history_window * self.bottleneck_dim,
        }

        # Read vector lists from config
        self.gnn_inputs = list(m.gnn_inputs)
        self.position_inputs = list(m.position_inputs)
        self.shape_mlp_inputs = [list(v) for v in m.shape_mlps]
        self.scale_mlp_inputs = [list(v) for v in m.scale_mlps]

        # Derive use_velocities from which vectors are referenced
        all_vectors = set(self.gnn_inputs + self.position_inputs)
        for inputs in self.shape_mlp_inputs + self.scale_mlp_inputs:
            all_vectors.update(inputs)
        self.use_velocities = bool({'L_velocity', 'scale_velocity'} & all_vectors)
        self.use_history = 'L_history' in all_vectors and self.history_window > 0

        # Parameter embedding. The init spread defaults to embedding_noise_std;
        # embedding_init_std sets it on its own (same RNG draw, so 0 gives all-ones).
        init_std = m.get("embedding_init_std", None)
        init_std = self.embedding_noise_std if init_std is None else float(init_std)
        self.parameter_embedding = torch.nn.Parameter(
            torch.normal(1, init_std, size=(m.num_nodes, m.embedding_dim), device=self.device)
        )

        # Feature indices for position MLP
        self.feature_indices = torch.asarray(m.indices_of_mlp_features, device=self.device)
        if len(self.feature_indices) > m.feature_dim:
            raise ValueError("Number of feature indices exceeds feature dimension")

        # Optional affine normalisation of x_features: (x - offset) * scale.
        # Defaults are exactly 0.0 / 1.0, so unless a config sets them this is
        # bit-identical to indexing x directly and no existing run changes.
        # It exists because the state channels are on wildly different scales
        # when absolute position is included: on ctc_mix_b_late, mean |position|
        # is 556 domain units against mean |velocity| 0.55 — a 1016x ratio that
        # swamps the first MLP layer and left `parameter_embedding` stuck at its
        # initialisation (per-dim std 0.100) with train loss 6x worse than the
        # velocity-only run. GraphModelReal derives sensible values; see there.
        fs = m.get("feature_scale", None)
        fo = m.get("feature_offset", None)
        n_feat = len(self.feature_indices)
        self.register_buffer(
            "feature_scale",
            torch.ones(n_feat, device=self.device) if fs is None
            else torch.as_tensor(list(fs), dtype=torch.float32, device=self.device))
        self.register_buffer(
            "feature_offset",
            torch.zeros(n_feat, device=self.device) if fo is None
            else torch.as_tensor(list(fo), dtype=torch.float32, device=self.device))
        for nm, buf in (("feature_scale", self.feature_scale),
                        ("feature_offset", self.feature_offset)):
            if buf.numel() != n_feat:
                raise ValueError(
                    f"model.{nm} has {buf.numel()} entries but "
                    f"indices_of_mlp_features has {n_feat}")

        # GNN
        self.gnn = GNN(
            output_dim=m.interaction_dim,
            spatial_dim=m.spatial_dim,
            embedding_dim=self._resolve_input_dim(self.gnn_inputs),
            hidden_dim=m.hidden_dim,
            hidden_layers=m.hidden_layers,
            activation=activation_module,
            periodic_boundary=self.periodic_boundary,
            domain_size=self.domain_size,
            device=self.device,
        )

        # Velocity MLP: predicts velocity residual; position is integrated
        # from the predicted velocity (new_pos = pos + new_vel * dt) so the
        # skip can't give a near-identity free lunch on position.
        self.mlp_position = MLP(
            input_size=self._resolve_input_dim(self.position_inputs),
            hidden_size=m.hidden_dim,
            num_hidden_layers=m.hidden_layers,
            output_size=m.spatial_dim,
            activation=activation_module,
        ).to(self.device)

        # Shape MLPs (outputs are summed)
        self.shape_mlps = torch.nn.ModuleList()
        for inputs in self.shape_mlp_inputs:
            self.shape_mlps.append(MLP(
                input_size=self._resolve_input_dim(inputs),
                hidden_size=m.hidden_dim,
                num_hidden_layers=m.hidden_layers,
                output_size=self.bottleneck_dim,
                activation=activation_module,
            ).to(self.device))

        # Scale MLPs (outputs are summed)
        self.scale_mlps = torch.nn.ModuleList()
        for inputs in self.scale_mlp_inputs:
            self.scale_mlps.append(MLP(
                input_size=self._resolve_input_dim(inputs),
                hidden_size=m.hidden_dim,
                num_hidden_layers=m.hidden_layers,
                output_size=1,
                activation=activation_module,
            ).to(self.device))

        # Metadata storage
        self._metadata_keys: list[str] = []

    @property
    def dino_decoder(self):
        """Backward-compatible alias for shape_decoder."""
        return self.shape_decoder

    def set_metadata(self, key: str = None, value: torch.Tensor = None, **kwargs) -> None:
        """Set particle/simulation metadata for visualization and validation."""
        if key is not None and value is not None:
            kwargs[key] = value

        for k, v in kwargs.items():
            if not isinstance(v, torch.Tensor):
                v = torch.tensor(v, dtype=torch.float32)
            self.register_buffer(f'_meta_{k}', v.to(self.device))
            if k not in self._metadata_keys:
                self._metadata_keys.append(k)

    def get_metadata(self, key: str) -> torch.Tensor | None:
        """Get a metadata tensor by key."""
        return getattr(self, f'_meta_{key}', None)

    @property
    def metadata(self) -> dict[str, torch.Tensor]:
        """Return all metadata as a dictionary."""
        return {k: self.get_metadata(k) for k in self._metadata_keys}

    def _resolve_input_dim(self, names):
        """Compute total input dimension for a list of vector names."""
        total = 0
        for name in names:
            if name not in self._vector_dims:
                raise ValueError(
                    f"Unknown vector name: '{name}'. "
                    f"Available: {sorted(self._vector_dims.keys())}"
                )
            total += self._vector_dims[name]
        return total

    def _noise_inputs(self, x, L, x_target, L_target):
        """Add Gaussian noise to positions, velocities, and bottleneck.

        Each group's scale is `input_noise_std * sigma_group`, where
        `sigma_group` is the stddev of that group's per-frame displacement
        magnitude (||Δ||₂ across the component axis). Uses precomputed
        dataset-level sigmas if set, else per-batch estimates. Targets are
        never noised — only the values fed into forward().
        """
        if not (self.training and self.input_noise_std > 0):
            return x, L
        S = self.spatial_dim
        if self.x_disp_std is not None:
            sigma_x = self.x_disp_std
            sigma_v = self.v_disp_std
            sigma_L = self.L_disp_std
        else:
            dx = x_target[..., :S] - x[..., :S]
            dv = x_target[..., S:2 * S] - x[..., S:2 * S]
            dL = L_target - L
            sigma_x = dx.norm(dim=-1).std().detach()
            sigma_v = dv.norm(dim=-1).std().detach()
            sigma_L = dL.norm(dim=-1).std().detach()

        x = x.clone()
        x[..., :S] = x[..., :S] + torch.empty_like(x[..., :S]).normal_(generator=self._noise_gen) * self.input_noise_std * sigma_x
        x[..., S:2 * S] = x[..., S:2 * S] + torch.empty_like(x[..., S:2 * S]).normal_(generator=self._noise_gen) * self.input_noise_std * sigma_v
        L = L + torch.empty_like(L).normal_(generator=self._noise_gen) * self.input_noise_std * sigma_L
        return x, L

    def _build_vectors(self, b, L, current_scale, L_velocity=None,
                       scale_velocity=None, L_history=None):
        """Build named vector dict for config-driven MLP input gathering.

        When history_window=0, configs can still list 'L_history' in their
        input vectors; we emit a zero-width placeholder so _gather /
        downstream MLPs (whose L_history slice is 0-wide by construction)
        keep working without a special case.
        """
        vectors = {
            'b': b,
            'L': L,
            'scale': current_scale.unsqueeze(-1),
        }
        if L_velocity is not None:
            vectors['L_velocity'] = L_velocity
        if scale_velocity is not None:
            vectors['scale_velocity'] = scale_velocity.unsqueeze(-1)
        if L_history is not None:
            # (B, N, K, bn) → (B, N, K*bn) so it concatenates like other vectors.
            vectors['L_history'] = L_history.flatten(start_dim=-2)
        elif self.history_window == 0:
            vectors['L_history'] = L.new_empty(*L.shape[:-1], 0)
        return vectors

    def _gather(self, names, vectors):
        """Concatenate named vectors into a single tensor.

        With an empty `names` list, returns a zero-width tensor shaped like
        the leading dims of any vector in `vectors` so downstream `torch.cat`
        / MLP calls (sized for input_dim=0) work without a special case.
        """
        if not names:
            ref = next(iter(vectors.values()))
            return ref.new_empty(*ref.shape[:-1], 0)
        return torch.cat([vectors[n] for n in names], dim=-1)

    def _evolve_shape_and_scale(self, vectors, L, current_scale):
        """Compute next bottleneck vectors and scale using config-driven MLPs.

        Each shape/scale MLP processes its configured input vectors.
        When multiple MLPs exist, their outputs are summed.
        Applies skip connections if enabled.
        """
        # Per-cell whole-L dropout on shape-MLP inputs only: some cells see
        # L = 0, L_velocity = 0, and L_history = 0, forcing b + interactions
        # to carry the prediction. Skip connection (below) still uses the
        # true L. L_history is masked with the same per-cell mask so the
        # history can't back-door past the dropout.
        # Symmetric whole-scale dropout on scale-MLP inputs only.
        shape_vectors = vectors
        scale_vectors = vectors
        B, N = L.shape[:2]
        if self.training and self.L_dropout_prob > 0:
            B, N = L.shape[:2]
            keep = (torch.empty(B, N, 1, device=L.device).uniform_(0, 1, generator=self._noise_gen) >= self.L_dropout_prob).to(L.dtype)
            shape_vectors = dict(vectors)
            shape_vectors['L'] = vectors['L'] * keep
            if 'L_velocity' in vectors:
                shape_vectors['L_velocity'] = vectors['L_velocity'] * keep
            if 'L_history' in vectors:
                shape_vectors['L_history'] = vectors['L_history'] * keep
        if self.training and self.scale_dropout_prob > 0:
            keep = (torch.empty(B, N, 1, device=L.device).uniform_(0, 1, generator=self._noise_gen) >= self.scale_dropout_prob).to(L.dtype)
            scale_vectors = dict(vectors)
            scale_vectors['scale'] = vectors['scale'] * keep
            if 'scale_velocity' in vectors:
                scale_vectors['scale_velocity'] = vectors['scale_velocity'] * keep

        # Shape MLPs
        L_update = 0
        for mlp, input_names in zip(self.shape_mlps, self.shape_mlp_inputs):
            L_update = L_update + mlp(self._gather(input_names, shape_vectors))

        # Scale MLPs
        scale_delta = 0
        for mlp, input_names in zip(self.scale_mlps, self.scale_mlp_inputs):
            scale_delta = scale_delta + mlp(self._gather(input_names, scale_vectors)).squeeze(-1)

        L_out = L + L_update if self.L_skip_connection else L_update
        next_scale = current_scale + scale_delta if self.skip_connection else scale_delta
        return L_out, next_scale

    def forward(self, x, W_interactions, L, current_scale, L_velocity=None,
                scale_velocity=None, L_history=None):
        """
        Args:
            x: (batch_size, num_nodes, feature_dim)
            W_interactions: Stacked sparse COO tensor (batch_size, num_nodes, num_nodes)
            L: (batch_size, num_nodes, bottleneck_dim) pre-computed bottleneck vectors
            current_scale: (batch_size, num_nodes) current crop side per cell
            L_velocity: (batch_size, num_nodes, bottleneck_dim) L(t) - L(t-1), optional
            scale_velocity: (batch_size, num_nodes) scale(t) - scale(t-1), optional
            L_history: (batch_size, num_nodes, history_window, bottleneck_dim) oldest-first, optional

        Returns:
            new_x: Updated positions (batch_size, num_nodes, feature_dim)
            L_out: Evolved bottleneck vectors (batch_size, num_nodes, bottleneck_dim)
            next_scale: Predicted crop side per cell (batch_size, num_nodes)
        """
        batch_size, num_nodes, feature_dim = x.shape

        # Expand parameter embedding to match batch size
        b = self.parameter_embedding.unsqueeze(0).expand(batch_size, -1, -1)
        if self.training and self.embedding_noise_std > 0:
            b = b + torch.empty_like(b).normal_(generator=self._noise_gen) * self.embedding_noise_std * b.std().detach()

        vectors = self._build_vectors(b, L, current_scale, L_velocity,
                                      scale_velocity, L_history)

        # GNN interactions
        interactions = self.gnn(x, self._gather(self.gnn_inputs, vectors), W_interactions)
        vectors['interactions'] = interactions
        vectors['x_features'] = ((x[..., self.feature_indices] - self.feature_offset)
                                 * self.feature_scale)

        # Velocity prediction → integrated position
        S = self.spatial_dim
        dv = self.mlp_position(self._gather(self.position_inputs, vectors))
        new_vel = x[..., S:2 * S] + dv if self.skip_connection else dv
        new_pos = x[..., :S] + new_vel * self.dt
        if self.periodic_boundary:
            new_pos = new_pos % self.domain_size
        new_x = torch.cat([new_pos, new_vel], dim=-1)

        # Shape and scale
        L_out, next_scale = self._evolve_shape_and_scale(vectors, L, current_scale)

        return new_x, L_out, next_scale

    def forward_from_batch(self, batch: dict):
        """Unpack a dataloader batch and run forward(). Returns model outputs —
        no loss computation. Suitable for plotting / eval. Noise injection
        self-gates on self.training, so callers should set eval mode as needed.

        Bottleneck vectors (L, prev_L, L_history) are precomputed at dataloader
        setup time; no shape_decoder calls happen here.
        """
        x = batch["states"]
        W_interactions = batch["W_interactions"]

        L = batch["L"]
        current_scale = batch["crop_sides"].float()

        L_velocity = None
        scale_velocity = None
        if self.use_velocities:
            L_velocity = L - batch["prev_L"]
            scale_velocity = current_scale - batch["prev_crop_sides"].float()

        L_history = None
        if self.use_history:
            # (B, K, N, bn) → (B, N, K, bn) to match training layout.
            L_history = batch["L_history"].transpose(1, 2).contiguous()

        return self.forward(x, W_interactions, L, current_scale,
                            L_velocity, scale_velocity, L_history)

    def training_step(self, batch: dict) -> dict:
        """Single training step.

        Args:
            batch: Dictionary containing:
                - states: (batch_size, num_nodes, feature_dim)
                - W_interactions: Stacked sparse COO tensor
                - next_states: (batch_size, num_nodes, feature_dim) targets
                - L: (batch_size, num_nodes, bottleneck_dim) bottleneck at time t
                - next_L: (batch_size, num_nodes, bottleneck_dim) bottleneck at t+1
                - prev_L, L_history: optional, used when velocities/history enabled

        Returns:
            Dictionary of loss terms
        """
        x = batch["states"]
        W_interactions = batch["W_interactions"]

        L = batch["L"]
        target_next = batch["next_L"]

        current_scale = batch["crop_sides"].float()

        x, L = self._noise_inputs(x, L, batch["next_states"], target_next)

        # Compute velocities from previous timestep if enabled (uses noised L
        # so velocity reflects the corruption the model sees on its input).
        L_velocity = None
        scale_velocity = None
        if self.use_velocities:
            L_velocity = L - batch["prev_L"]
            scale_velocity = current_scale - batch["prev_crop_sides"].float()

        # Past-K-frame bottleneck history, transposed to (B, N, K, bn) for MLPs
        L_history = None
        if self.use_history:
            L_history = batch["L_history"].transpose(1, 2).contiguous()

        new_x, L_out, next_scale = self.forward(x, W_interactions, L, current_scale,
                                                L_velocity, scale_velocity, L_history)

        losses = {
            "position_mse": mse_loss(
                new_x[..., :self.spatial_dim],
                batch["next_states"][..., :self.spatial_dim],
            ),
            "velocity_mse": mse_loss(
                new_x[..., self.spatial_dim:self.spatial_dim * 2],
                batch["next_states"][..., self.spatial_dim:self.spatial_dim * 2],
            ),
            "dino_mse": mse_loss(L_out, target_next),
            "scale_mse": mse_loss(next_scale / self.image_size, batch["next_crop_sides"].float() / self.image_size),
            "parameter_embedding_l2": torch.mean(self.parameter_embedding**2),
        }

        return losses

    def rollout_training_step(self, batch: dict, rollout_steps: int) -> dict:
        """Multi-step rollout training step with autoregressive prediction.

        Backpropagates through a chain of predictions, accumulating loss at
        each step against ground truth. W_interactions from step 0 is reused.

        Args:
            batch: Dictionary with sequence keys (*_seq) from the dataloader
            rollout_steps: Number of autoregressive steps

        Returns:
            Dictionary of averaged loss terms
        """
        if rollout_steps == 1:
            return self.training_step(batch)

        W_interactions = batch["W_interactions"]

        # Bottleneck vectors for the full sequence (precomputed at data load).
        all_L = batch["L_seq"]                # (B, rollout_steps, N, bn)
        all_L_target = batch["next_L_seq"]    # (B, rollout_steps, N, bn)

        # Initialize from ground truth at step 0
        x = batch["states_seq"][:, 0]
        L = all_L[:, 0]
        current_scale = batch["crop_sides_seq"][:, 0].float()

        # Step-0 noise injection; the rollout's own accumulated error plays
        # the same role for step > 0.
        x, L = self._noise_inputs(x, L, batch["next_states_seq"][:, 0], all_L_target[:, 0])

        # Compute initial velocity from previous timestep (uses noised L).
        if self.use_velocities:
            L_velocity = L - batch["prev_L"]
            scale_velocity = current_scale - batch["prev_crop_sides"].float()
        else:
            L_velocity = None
            scale_velocity = None

        # Seed the rolling L-history buffer from past ground-truth frames.
        # Shape (B, N, K, bn), oldest-first. At each rollout step we shift a
        # slot and append the last input L so the buffer tracks actual inputs.
        L_history = None
        if self.use_history:
            L_history = batch["L_history"].transpose(1, 2).contiguous()

        loss_accum = {"parameter_embedding_l2": torch.mean(self.parameter_embedding**2)}

        for step in range(rollout_steps):
            new_x, L_out, next_scale = self.forward(x, W_interactions, L, current_scale,
                                                    L_velocity, scale_velocity, L_history)

            step_losses = {
                "position_mse": mse_loss(
                    new_x[..., :self.spatial_dim],
                    batch["next_states_seq"][:, step, ..., :self.spatial_dim],
                ),
                "velocity_mse": mse_loss(
                    new_x[..., self.spatial_dim:self.spatial_dim * 2],
                    batch["next_states_seq"][:, step, ..., self.spatial_dim:self.spatial_dim * 2],
                ),
                "dino_mse": mse_loss(L_out, all_L_target[:, step]),
                "scale_mse": mse_loss(
                    next_scale / self.image_size,
                    batch["next_crop_sides_seq"][:, step].float() / self.image_size,
                ),
            }


            for k, v in step_losses.items():
                loss_accum[k] = loss_accum.get(k, 0.0) + v

            # Autoregressive: compute velocity before advancing state
            if self.use_velocities:
                L_velocity = L_out - L
                scale_velocity = next_scale - current_scale

            # Slide the history buffer: drop oldest, append the L we just
            # consumed. Detach — the buffer is meant as an observed-history
            # summary, and letting gradients flow through the K-step window
            # would make memory grow linearly with rollout depth.
            if self.use_history:
                L_history = torch.cat(
                    [L_history[:, :, 1:, :], L.detach().unsqueeze(2)], dim=2
                )

            x = new_x
            L = L_out
            current_scale = next_scale

        return {k: v / rollout_steps for k, v in loss_accum.items()}

    def generate(self, x, W_interactions, b, L, current_scale, L_velocity=None,
                 scale_velocity=None, L_history=None):
        """Single-step generation for inference (unbatched).

        Args:
            x: (num_nodes, feature_dim)
            W_interactions: Sparse COO tensor (num_nodes, num_nodes)
            b: (num_nodes, embedding_dim) - the parameter embedding
            L: (num_nodes, bottleneck_dim) - current bottleneck vectors
            current_scale: (num_nodes,) - current crop side per cell
            L_velocity: (num_nodes, bottleneck_dim) L(t) - L(t-1), optional
            scale_velocity: (num_nodes,) scale(t) - scale(t-1), optional
            L_history: (num_nodes, history_window, bottleneck_dim) oldest-first, optional

        Returns:
            new_x: Updated positions (num_nodes, feature_dim)
            L_out: Evolved bottleneck vectors (num_nodes, bottleneck_dim)
            next_scale: Predicted crop side per cell (num_nodes,)
        """
        vectors = self._build_vectors(b, L, current_scale, L_velocity,
                                      scale_velocity, L_history)

        # GNN interactions
        interactions = self.gnn.generate(x, self._gather(self.gnn_inputs, vectors), W_interactions)
        vectors['interactions'] = interactions
        vectors['x_features'] = ((x[..., self.feature_indices] - self.feature_offset)
                                 * self.feature_scale)

        # Velocity prediction → integrated position
        S = self.spatial_dim
        dv = self.mlp_position(self._gather(self.position_inputs, vectors))
        new_vel = x[..., S:2 * S] + dv if self.skip_connection else dv
        new_pos = x[..., :S] + new_vel * self.dt
        if self.periodic_boundary:
            new_pos = new_pos % self.domain_size
        new_x = torch.cat([new_pos, new_vel], dim=-1)

        # Shape and scale
        L_out, next_scale = self._evolve_shape_and_scale(vectors, L, current_scale)

        return new_x, L_out, next_scale

    def decode_to_image(self, L_out):
        """Visualization utility: decode bottleneck vectors to shape images.

        Bypasses get_bottleneck since input is already in bottleneck space.

        Args:
            L_out: (..., bottleneck_dim) bottleneck vectors

        Returns:
            images: (..., output_size, output_size) shape images
        """
        leading_shape = L_out.shape[:-1]
        flat = L_out.reshape(-1, self.bottleneck_dim)

        with torch.no_grad():
            images = self.shape_decoder.decode_from_bottleneck(flat)

        return images.view(*leading_shape, images.shape[-2], images.shape[-1])


