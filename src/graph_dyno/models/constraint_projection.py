#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
from graph_dyno.models.mlp import MLP


def _safe_norm(x):
    """Compute vector norm with clamped squared norm to avoid NaN grad at zero."""
    return (x * x).sum(dim=-1).clamp(min=1e-12).sqrt()


class ConstraintProjection(nn.Module):
    """Neural projection operator for non-overlap enforcement.

    Inspired by Yang et al. (NeurIPS 2020) "Learning Physical Constraints
    with Neural Projections", using per-edge Newton steps with Jacobi
    accumulation (analogous to their GroupProjection for collision).

    Each edge gets its own Newton step:
        λ_e = k_iter · C_e / (‖∇_i C_e‖² + ‖∇_j C_e‖²)
        δx_i = −λ_e · ∇_i C_e,   δx_j = −λ_e · ∇_j C_e

    Corrections are accumulated per node (Jacobi-style) and applied once
    per iteration. This gives each pairwise constraint its own properly
    scaled correction, rather than one blended step size per sample.

    Key design choices:
      - Per-edge Newton normalization: each constraint gets its own step
        scale, so large and small violations are corrected proportionally.
      - PBD iteration-dependent stiffness: k_iter = 1 - (1-k)^(1/(i+1))
        from Muller et al. 2007. When stiffness=1, hard constraint (full
        projection). When stiffness<1, soft/elastic constraint.
    """

    def __init__(
        self,
        bottleneck_dim: int,
        spatial_dim: int,
        hidden_dim: int = 32,
        hidden_layers: int = 1,
        num_passes: int = 3,
        periodic_boundary: bool = False,
        domain_size: float | None = None,
        activation: type[nn.Module] = nn.PReLU,
        stiffness: float = 1.0,
        stiffness_warmup_epochs: int = 0,
    ):
        super().__init__()
        self.bottleneck_dim = bottleneck_dim
        self.spatial_dim = spatial_dim
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.num_passes = num_passes
        self.periodic_boundary = periodic_boundary
        self.domain_size = domain_size
        self.stiffness_target = stiffness
        self.stiffness_warmup_epochs = stiffness_warmup_epochs
        self.register_buffer("_epoch", torch.tensor(0, dtype=torch.long))

        # Per-edge constraint MLP
        # Input: [x_ij (sd), d_ij (1), L_i (bd), L_j (bd), scale_i (1), scale_j (1)]
        constraint_input_dim = spatial_dim + 1 + 2 * bottleneck_dim + 2
        self.constraint_mlp = MLP(
            input_size=constraint_input_dim,
            hidden_size=hidden_dim,
            num_hidden_layers=hidden_layers,
            output_size=1,
            activation=activation,
        )

        # Per-node shape correction from [L, total_violation, edge_count, mean_dir(sd)]
        # Total: bottleneck_dim + 2 + spatial_dim
        self.shape_mlp = MLP(
            input_size=bottleneck_dim + 2 + spatial_dim,
            hidden_size=hidden_dim,
            num_hidden_layers=hidden_layers,
            output_size=bottleneck_dim,
            activation=activation,
        )

    def step_epoch(self):
        """Advance the epoch counter (call once per epoch from training loop)."""
        self._epoch += 1

    @property
    def stiffness(self) -> float:
        """Current stiffness, linearly ramped from 0 to target over warmup epochs."""
        if self.stiffness_warmup_epochs <= 0:
            return self.stiffness_target
        t = min(self._epoch.item() / self.stiffness_warmup_epochs, 1.0)
        return self.stiffness_target * t

    def _iteration_stiffness(self, i: int) -> float:
        """PBD iteration-dependent stiffness (Yang et al. Eq. from Muller 2007).

        k_iter = 1 - (1 - stiffness)^(1/(i+1))

        When stiffness=1 (hard constraint), returns 1.0 for all iterations.
        When stiffness<1 (soft constraint), returns decreasing values so early
        iterations correct more and later iterations refine.
        """
        k = self.stiffness
        if k >= 1.0:
            return 1.0
        return 1.0 - (1.0 - k) ** (1.0 / (i + 1))

    def _compute_relative_position(self, pos_i, pos_j):
        """Compute relative position with periodic BC handling (mirrors gnn.py:40-41)."""
        x_ij = pos_j - pos_i
        if self.periodic_boundary and self.domain_size is not None:
            # Non-in-place periodic wrapping (safe for autograd)
            wrapped = x_ij[..., :2] - self.domain_size * torch.round(
                x_ij[..., :2] / self.domain_size
            )
            if x_ij.shape[-1] > 2:
                x_ij = torch.cat([wrapped, x_ij[..., 2:]], dim=-1)
            else:
                x_ij = wrapped
        return x_ij

    def _compute_edge_constraint(self, positions, L, scale, src_idx, tgt_idx):
        """Compute per-edge constraint values via the learned constraint MLP.

        Returns per-edge softplus(constraint_mlp(...)) values.
        """
        pos_i = positions[src_idx]
        pos_j = positions[tgt_idx]
        x_ij = self._compute_relative_position(pos_i, pos_j)
        d_ij = _safe_norm(x_ij)

        edge_features = torch.cat([
            x_ij,
            d_ij.unsqueeze(-1),
            L[src_idx],
            L[tgt_idx],
            scale[src_idx].unsqueeze(-1),
            scale[tgt_idx].unsqueeze(-1),
        ], dim=-1)

        per_edge_C = F.softplus(self.constraint_mlp(edge_features).squeeze(-1))
        return per_edge_C

    def _compute_edge_constraint_from_pos(self, pos_i, pos_j, L_i, L_j, scale_i, scale_j):
        """Compute per-edge constraint from already-gathered endpoint tensors.

        Args:
            pos_i, pos_j: (num_edges, sd) endpoint positions
            L_i, L_j: (num_edges, bottleneck_dim) endpoint shape embeddings
            scale_i, scale_j: (num_edges,) endpoint scales

        Returns:
            per_edge_C: (num_edges,) softplus constraint values
        """
        x_ij = self._compute_relative_position(pos_i, pos_j)
        d_ij = _safe_norm(x_ij)

        edge_features = torch.cat([
            x_ij,
            d_ij.unsqueeze(-1),
            L_i, L_j,
            scale_i.unsqueeze(-1),
            scale_j.unsqueeze(-1),
        ], dim=-1)

        return F.softplus(self.constraint_mlp(edge_features).squeeze(-1))

    def _apply_projection(self, new_x, L, scale, src_idx, tgt_idx, edge_weights,
                          batch_size=1, nodes_per_sample=None):
        """Core projection via per-edge Newton steps with Jacobi accumulation.

        Each edge gets its own Newton step λ_e = C_e / ‖∇C_e‖², and the
        resulting corrections are accumulated per node (Jacobi-style) before
        being applied. This mirrors GroupProjection from Yang et al. where
        each constraint group gets independently normalized Newton steps.

        Args:
            new_x: (total_nodes, feature_dim)
            L: (total_nodes, bottleneck_dim)
            scale: (total_nodes,)
            src_idx, tgt_idx: (total_edges,) flat node indices
            edge_weights: (total_edges,)
            batch_size: number of samples in the batch
            nodes_per_sample: nodes per sample (N); None = total_nodes (unbatched)
        """
        sd = self.spatial_dim
        total_nodes = new_x.shape[0]
        num_edges = src_idx.shape[0]

        if nodes_per_sample is None:
            nodes_per_sample = total_nodes

        x_corrected = new_x

        if num_edges == 0:
            shape_input = torch.cat([
                L,
                torch.zeros(total_nodes, 2 + sd, device=L.device),
            ], dim=-1)
            shape_delta = self.shape_mlp(shape_input)
            return x_corrected, L + shape_delta

        # Accumulate contact info across all passes for shape correction
        total_violation_per_node = torch.zeros(total_nodes, 1, device=new_x.device)
        edge_count_per_node = torch.zeros(total_nodes, 1, device=new_x.device)
        mean_dir_accum = torch.zeros(total_nodes, sd, device=new_x.device)

        for i in range(self.num_passes):
            k_iter = self._iteration_stiffness(i)

            # --- Per-edge Newton steps via autograd ---
            with torch.enable_grad():
                # Gather endpoint positions as differentiable leaf tensors
                positions = x_corrected[:, :sd]
                pos_i = positions[src_idx]
                pos_j = positions[tgt_idx]
                if not pos_i.requires_grad:
                    pos_i = pos_i.detach().clone().requires_grad_(True)
                    pos_j = pos_j.detach().clone().requires_grad_(True)

                per_edge_C = self._compute_edge_constraint_from_pos(
                    pos_i, pos_j,
                    L[src_idx], L[tgt_idx],
                    scale[src_idx], scale[tgt_idx],
                )

                # Per-edge gradients: ∇_{pos_i} C_e and ∇_{pos_j} C_e
                grad_i, grad_j = torch.autograd.grad(
                    per_edge_C.sum(), (pos_i, pos_j),
                    create_graph=self.training,
                    retain_graph=True,
                )  # each (num_edges, sd)

            # Per-edge Newton step: λ_e = k_iter · C_e / (‖∇_i C_e‖² + ‖∇_j C_e‖²)
            gnorm_sq_e = (grad_i ** 2).sum(dim=-1) + (grad_j ** 2).sum(dim=-1) + 1e-7
            lam_e = (k_iter * per_edge_C / gnorm_sq_e).unsqueeze(-1)  # (num_edges, 1)

            # Per-edge corrections for each endpoint
            delta_i = -lam_e * grad_i  # (num_edges, sd)
            delta_j = -lam_e * grad_j  # (num_edges, sd)

            # Jacobi accumulation: scatter-add corrections to node positions
            delta_pos = torch.zeros(total_nodes, sd, device=new_x.device)
            src_expand = src_idx.unsqueeze(-1).expand(-1, sd)
            tgt_expand = tgt_idx.unsqueeze(-1).expand(-1, sd)
            delta_pos.scatter_add_(0, src_expand, delta_i)
            delta_pos.scatter_add_(0, tgt_expand, delta_j)

            # Apply position correction (functional, no in-place ops)
            x_corrected = torch.cat([
                x_corrected[:, :sd] + delta_pos,
                x_corrected[:, sd:],
            ], dim=-1)

            # Periodic BC
            if self.periodic_boundary:
                wrapped = x_corrected[:, :2] % self.domain_size
                if x_corrected.shape[-1] > 2:
                    x_corrected = torch.cat(
                        [wrapped, x_corrected[:, 2:]], dim=-1
                    )
                else:
                    x_corrected = wrapped

            # Accumulate per-node contact info (detached — only for shape features)
            with torch.no_grad():
                pos_i_d = x_corrected[:, :sd][src_idx]
                pos_j_d = x_corrected[:, :sd][tgt_idx]
                x_ij_d = self._compute_relative_position(pos_i_d, pos_j_d)
                d_ij_d = torch.norm(x_ij_d, dim=-1, keepdim=True).clamp(min=1e-8)
                r_hat = x_ij_d / d_ij_d

                violation = per_edge_C.detach()

                total_violation_per_node.scatter_add_(
                    0, src_idx.unsqueeze(-1), violation.unsqueeze(-1)
                )
                edge_count_per_node.scatter_add_(
                    0, src_idx.unsqueeze(-1), torch.ones_like(violation.unsqueeze(-1))
                )
                dir_contrib = violation.unsqueeze(-1) * (-r_hat)
                mean_dir_accum.scatter_add_(0, src_expand, dir_contrib)

        # Normalize mean direction (zero for nodes with no contacts)
        dir_norm = mean_dir_accum.norm(dim=-1, keepdim=True)
        has_contact = (dir_norm > 1e-8).float()
        mean_dir = mean_dir_accum / dir_norm.clamp(min=1e-8) * has_contact

        # Shape correction
        shape_input = torch.cat([
            L, total_violation_per_node, edge_count_per_node, mean_dir
        ], dim=-1)
        shape_delta = self.shape_mlp(shape_input)
        L_corrected = L + shape_delta

        return x_corrected, L_corrected

    def forward(self, new_x, L, scale, W_interactions):
        """Batched projection.

        Args:
            new_x: (B, N, feature_dim) predicted positions/velocities
            L: (B, N, bottleneck_dim) bottleneck vectors
            scale: (B, N) scale values
            W_interactions: Stacked sparse COO tensor (B, N, N) with 3-index

        Returns:
            corrected_x: (B, N, feature_dim)
            corrected_L: (B, N, bottleneck_dim)
        """
        B, N, feat = new_x.shape

        # Extract edges from stacked sparse tensor (3-index: batch, src, tgt)
        indices = W_interactions._indices()  # (3, total_edges)
        edge_weights = W_interactions._values()

        if indices.shape[1] > 0:
            batch_idx = indices[0]
            src_flat = batch_idx * N + indices[1]
            tgt_flat = batch_idx * N + indices[2]
        else:
            src_flat = torch.zeros(0, dtype=torch.long, device=new_x.device)
            tgt_flat = torch.zeros(0, dtype=torch.long, device=new_x.device)
            edge_weights = torch.zeros(0, device=new_x.device)

        x_flat = new_x.reshape(B * N, feat)
        L_flat = L.reshape(B * N, -1)
        scale_flat = scale.reshape(B * N)

        x_out, L_out = self._apply_projection(
            x_flat, L_flat, scale_flat, src_flat, tgt_flat, edge_weights,
            batch_size=B, nodes_per_sample=N,
        )

        return x_out.view(B, N, feat), L_out.view(B, N, -1)

    def project(self, new_x, L, scale, W_interactions):
        """Unbatched projection for inference.

        Args:
            new_x: (N, feature_dim)
            L: (N, bottleneck_dim)
            scale: (N,)
            W_interactions: Sparse COO tensor (N, N) with 2-index

        Returns:
            corrected_x: (N, feature_dim)
            corrected_L: (N, bottleneck_dim)
        """
        indices = W_interactions._indices()  # (2, total_edges)
        edge_weights = W_interactions._values()

        if indices.shape[1] > 0:
            src = indices[0]
            tgt = indices[1]
        else:
            src = torch.zeros(0, dtype=torch.long, device=new_x.device)
            tgt = torch.zeros(0, dtype=torch.long, device=new_x.device)
            edge_weights = torch.zeros(0, device=new_x.device)

        return self._apply_projection(
            new_x, L, scale, src, tgt, edge_weights,
            batch_size=1, nodes_per_sample=new_x.shape[0],
        )

    def compute_violation_loss(self, new_x, L, scale, W_interactions):
        """Mean per-edge constraint violation (monitoring only, no MLP gradients).

        The constraint_mlp learns exclusively through the position loss
        backpropagating through the Newton projection (create_graph=True).
        This method evaluates the current constraint magnitude for logging
        but runs under torch.no_grad() so it cannot be trivially minimised
        by driving MLP outputs to -∞.

        Args:
            new_x: (B, N, feature_dim) corrected positions
            L: (B, N, bottleneck_dim) corrected bottleneck vectors
            scale: (B, N) scale values
            W_interactions: Stacked sparse COO tensor (B, N, N)

        Returns:
            Scalar loss (mean per-edge violation, detached from constraint MLPs)
        """
        B, N, feat = new_x.shape
        sd = self.spatial_dim

        indices = W_interactions._indices()

        if indices.shape[1] == 0:
            return torch.tensor(0.0, device=new_x.device)

        batch_idx = indices[0]
        src = indices[1]
        tgt = indices[2]

        # Flatten batched tensors to (B*N, ...) so _compute_edge_constraint works
        positions_flat = new_x[..., :sd].reshape(B * N, sd)
        L_flat = L.reshape(B * N, -1)
        scale_flat = scale.reshape(B * N)
        src_flat = batch_idx * N + src
        tgt_flat = batch_idx * N + tgt

        with torch.no_grad():
            per_edge_C = self._compute_edge_constraint(
                positions_flat, L_flat, scale_flat, src_flat, tgt_flat
            )
        return per_edge_C.mean()


if __name__ == "__main__":
    print("=" * 60)
    print("ConstraintProjection smoke test")
    print("=" * 60)

    bottleneck_dim = 64
    spatial_dim = 2
    feature_dim = 4
    N = 10
    B = 3

    cp = ConstraintProjection(
        bottleneck_dim=bottleneck_dim,
        spatial_dim=spatial_dim,
        hidden_dim=32,
        num_passes=3,
    )
    n_params = sum(p.numel() for p in cp.parameters())
    print(f"\n[1] Instantiation: OK ({n_params} parameters)")

    # --- Batched forward ---
    x = torch.randn(B, N, feature_dim)
    L = torch.randn(B, N, bottleneck_dim)
    scale = torch.rand(B, N) * 100

    W_list = []
    for _ in range(B):
        idx = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
        vals = torch.ones(4)
        W_list.append(torch.sparse_coo_tensor(idx, vals, (N, N)))
    W_batch = torch.stack(W_list)

    x_out, L_out = cp(x, L, scale, W_batch)
    assert x_out.shape == (B, N, feature_dim)
    assert L_out.shape == (B, N, bottleneck_dim)
    print(f"[2] Batched forward: OK  x={x_out.shape}, L={L_out.shape}")

    # --- Unbatched project ---
    x_s = torch.randn(N, feature_dim)
    L_s = torch.randn(N, bottleneck_dim)
    scale_s = torch.rand(N) * 100
    idx_s = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    W_s = torch.sparse_coo_tensor(idx_s, torch.ones(3), (N, N))

    x_p, L_p = cp.project(x_s, L_s, scale_s, W_s)
    assert x_p.shape == (N, feature_dim)
    assert L_p.shape == (N, bottleneck_dim)
    print(f"[3] Unbatched project: OK  x={x_p.shape}, L={L_p.shape}")

    # --- Violation loss ---
    loss = cp.compute_violation_loss(x, L, scale, W_batch)
    assert loss.ndim == 0 and loss >= 0
    print(f"[4] Violation loss: {loss.item():.6f} (non-negative: OK)")

    # --- Gradient flow ---
    cp.train()
    x_g = torch.randn(B, N, feature_dim, requires_grad=True)
    L_g = torch.randn(B, N, bottleneck_dim, requires_grad=True)
    scale_g = torch.rand(B, N) * 100
    x_out_g, L_out_g = cp(x_g, L_g, scale_g, W_batch)
    total = x_out_g.sum() + L_out_g.sum()
    total.backward()
    assert x_g.grad is not None
    print(f"[5] Gradient flow: OK")

    # --- Stiffness warmup ---
    cp2 = ConstraintProjection(
        bottleneck_dim=bottleneck_dim,
        spatial_dim=spatial_dim,
        hidden_dim=32,
        num_passes=3,
        stiffness=0.8,
        stiffness_warmup_epochs=10,
    )
    assert cp2.stiffness == 0.0
    for _ in range(5):
        cp2.step_epoch()
    assert abs(cp2.stiffness - 0.4) < 1e-6
    for _ in range(5):
        cp2.step_epoch()
    assert abs(cp2.stiffness - 0.8) < 1e-6
    print(f"[6] Stiffness warmup: OK")

    # --- Empty edges passthrough ---
    W_empty = torch.sparse_coo_tensor(
        torch.zeros(3, 0, dtype=torch.long), torch.zeros(0), (B, N, N)
    )
    x_e, L_e = cp(x, L, scale, W_empty)
    assert torch.allclose(x_e, x)
    print(f"[7] Empty edges: positions unchanged OK")

    print("\n" + "=" * 60)
    print("All smoke tests passed!")
    print("=" * 60)
