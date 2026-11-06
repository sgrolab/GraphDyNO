
from abc import ABC, abstractmethod
import torch
from graph_dyno.models.mlp import MLP

class ShapeEmbedding(ABC, torch.nn.Module):
    """Abstract base class for shape embeddings.

    Shape embeddings represent time-varying learned representations of object shapes.
    Subclasses must implement the access method to retrieve embeddings at specific times.
    """

    def __init__(self, num_nodes: int, shape_embedding_dim: int, device: str | None = None):
        """
        Args:
            num_nodes: Number of nodes (spheres) in the simulation
            shape_embedding_dim: Dimension of the shape embedding
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
        """
        super().__init__()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.num_nodes = num_nodes
        self.shape_embedding_dim = shape_embedding_dim

    @property
    @abstractmethod
    def embedding(self) -> torch.nn.Parameter:
        """Return the underlying learnable parameter."""
        pass

    @abstractmethod
    def access(
        self,
        t_idx: torch.Tensor,
        mask: torch.Tensor | None = None,
        rotation_angle: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Access the shape embedding at time t.

        Args:
            t_idx: Time index tensor of shape (batch_size,) or (batch_size, 1)
            mask: Optional boolean mask to filter the output
            rotation_angle: Optional rotation angle in radians of shape (batch_size,).
                           May be ignored by implementations that don't support it.

        Returns:
            Shape embedding tensor, filtered if mask is provided
        """
        pass

    def forward(
        self,
        t_idx: torch.Tensor,
        mask: torch.Tensor | None = None,
        rotation_angle: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass - delegates to access method."""
        return self.access(t_idx, mask, rotation_angle)

class ArrayShapeEmbedding(ShapeEmbedding):
    """A shape embedding that stores embeddings for all timepoints in an array.

    This is a simple wrapper around a torch Parameter to hold the
    shape embeddings for all timepoints. The embedding is indexed by
    computing t/dt to get the timepoint index.
    """

    def __init__(self, num_timepoints: int, num_nodes: int, shape_embedding_dim: int, device: str | None = None):
        """
        Args:
            num_timepoints: Number of timepoints to store embeddings for
            num_nodes: Number of nodes (spheres) in the simulation
            shape_embedding_dim: Dimension of the shape embedding
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
        """
        super().__init__(
            num_nodes=num_nodes,
            shape_embedding_dim=shape_embedding_dim,
            device=device
        )

        self.num_timepoints = num_timepoints

        self.shape_embedding = torch.nn.Parameter(
            torch.normal(1, 0.0, size=(num_timepoints, num_nodes, shape_embedding_dim), device=self.device)
        )

    @property
    def embedding(self) -> torch.nn.Parameter:
        """Return the shape embedding parameter.

        Returns:
            Parameter of shape (num_timepoints, num_nodes, shape_embedding_dim)
        """
        return self.shape_embedding

    def __getitem__(self, index):
        """Allow array-like access to the embedding."""
        return self.shape_embedding[index]

    def access(
        self,
        t_idx: torch.Tensor,
        mask: torch.Tensor | None = None,
        rotation_angle: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Access the shape embedding at time t.

        Args:
            t_idx: Time index tensor of shape (batch_size,) or (batch_size, 1)
            mask: Optional boolean mask to filter the output
            rotation_angle: Ignored - ArrayShapeEmbedding does not support rotation-aware mode

        Returns:
            Shape embedding at time t of shape (batch_size, num_nodes, shape_embedding_dim),
            or filtered shape if mask is provided
        """
        if mask is not None:
            t_idx = t_idx[mask]
        return self.shape_embedding[t_idx]


class MLPShapeEmbedding(ShapeEmbedding):
    """A shape embedding computed as a function of time and parameter embedding.

    Uses an MLP to map (normalized_time, parameter_embedding) -> shape_embedding.
    This allows the shape embedding to vary smoothly with time rather than
    being stored discretely for each timepoint.

    Optionally supports rotation-aware mode where the MLP also takes a rotation
    angle as input, allowing the embedding to adapt to rotated objects.
    """

    def __init__(
        self,
        parameter_embedding: torch.nn.Parameter,
        num_nodes: int,
        shape_embedding_dim: int,
        hidden_dim: int = 64,
        hidden_layers: int = 2,
        num_timepoints: int | None = None,
        activation: type = torch.nn.ReLU,
        device: str | None = None,
        rotation_aware: bool = False,
    ):
        """
        Args:
            parameter_embedding: Reference to the parameter embedding from GraphShapecoder
            num_nodes: Number of nodes in the simulation
            shape_embedding_dim: Dimension of the output shape embedding
            hidden_dim: Hidden dimension of the MLP
            hidden_layers: Number of hidden layers in the MLP
            num_timepoints: Number of timepoints (used for time normalization)
            activation: Activation function class
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
            rotation_aware: If True, the MLP takes rotation angle as additional input
        """
        super().__init__(
            num_nodes=num_nodes,
            shape_embedding_dim=shape_embedding_dim,
            device=device
        )

        # Store reference to parameter embedding (no copy)
        self._parameter_embedding = parameter_embedding
        self.parameter_embedding_dim = parameter_embedding.shape[-1]
        self.num_timepoints = num_timepoints
        self.rotation_aware = rotation_aware

        # MLP takes [time_normalized, parameter_embedding] as input
        # If rotation_aware, also takes [sin(angle), cos(angle)] for smooth angle representation
        input_dim = 1 + self.parameter_embedding_dim
        if rotation_aware:
            input_dim += 2  # sin(angle) and cos(angle)
        self.mlp = MLP(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_hidden_layers=hidden_layers,
            output_size=shape_embedding_dim,
            activation=activation,
        ).to(self.device)

    @property
    def embedding(self) -> torch.nn.Parameter:
        """Return MLP weights as proxy for the embedding parameter."""
        return self.mlp.layers[0].weight

    def _normalize_time(self, t_idx: torch.Tensor) -> torch.Tensor:
        """Normalize time index to [0, 1] range."""
        if self.num_timepoints is not None and self.num_timepoints > 1:
            return t_idx.float() / (self.num_timepoints - 1)
        return t_idx.float()

    def _compute_embedding(
        self, t_idx: torch.Tensor, rotation_angle: torch.Tensor | None = None,
        detach_param: bool = False,
    ) -> torch.Tensor:
        """Compute shape embedding for given time indices.

        Args:
            t_idx: Time index tensor of shape (batch_size,)
            rotation_angle: Optional rotation angle in radians of shape (batch_size,).
                           Only used if rotation_aware=True.
            detach_param: If True, detach the parameter embedding (cell_id_embedding)
                         to prevent gradients flowing back to it.

        Returns:
            Shape embedding of shape (batch_size, num_nodes, shape_embedding_dim)
        """
        batch_size = t_idx.shape[0]
        t_normalized = self._normalize_time(t_idx)  # (batch_size,)

        # Expand parameter embedding to batch size: (num_nodes, param_dim) -> (batch, num_nodes, param_dim)
        param = self._parameter_embedding.detach() if detach_param else self._parameter_embedding
        param_emb = param.unsqueeze(0).expand(batch_size, -1, -1)

        # Expand time to match nodes: (batch,) -> (batch, num_nodes, 1)
        t_expanded = t_normalized.unsqueeze(-1).unsqueeze(-1).expand(-1, self.num_nodes, 1)

        if self.rotation_aware:
            # Use sin/cos encoding for smooth angle representation
            if rotation_angle is None:
                # Default to zero rotation if not provided
                rotation_angle = torch.zeros(batch_size, device=t_idx.device)
            sin_angle = torch.sin(rotation_angle).unsqueeze(-1).unsqueeze(-1).expand(-1, self.num_nodes, 1)
            cos_angle = torch.cos(rotation_angle).unsqueeze(-1).unsqueeze(-1).expand(-1, self.num_nodes, 1)
            # Concatenate: (batch, num_nodes, 1 + param_dim + 2)
            mlp_input = torch.cat([t_expanded, param_emb, sin_angle, cos_angle], dim=-1)
        else:
            # Concatenate: (batch, num_nodes, 1 + param_dim)
            mlp_input = torch.cat([t_expanded, param_emb], dim=-1)

        # Flatten, compute, reshape: (batch * num_nodes, input_dim) -> (batch, num_nodes, shape_dim)
        mlp_input_flat = mlp_input.view(-1, mlp_input.shape[-1])
        shape_emb_flat = self.mlp(mlp_input_flat)
        shape_emb = shape_emb_flat.view(batch_size, self.num_nodes, self.shape_embedding_dim)

        return shape_emb

    def __getitem__(self, index):
        """Compute shape embedding for given time index/indices."""
        if isinstance(index, int):
            t_idx = torch.tensor([index], dtype=torch.long, device=self.device)
            return self._compute_embedding(t_idx).squeeze(0)
        elif isinstance(index, slice):
            start = index.start or 0
            stop = index.stop or self.num_timepoints
            step = index.step or 1
            t_idx = torch.arange(start, stop, step, dtype=torch.long, device=self.device)
            return self._compute_embedding(t_idx)
        else:
            t_idx = index if isinstance(index, torch.Tensor) else torch.tensor(index, device=self.device)
            if t_idx.dtype != torch.long:
                t_idx = t_idx.long()
            return self._compute_embedding(t_idx)

    def access(
        self,
        t_idx: torch.Tensor,
        mask: torch.Tensor | None = None,
        rotation_angle: torch.Tensor | None = None,
        detach_param: bool = False,
    ) -> torch.Tensor:
        """Access the shape embedding at time t.

        Args:
            t_idx: Time index tensor of shape (batch_size,) or (batch_size, 1)
            mask: Optional boolean mask to filter the output
            rotation_angle: Optional rotation angle in radians of shape (batch_size,).
                           Only used if rotation_aware=True.
            detach_param: If True, detach the parameter embedding (cell_id_embedding)
                         to prevent gradients flowing back to it.

        Returns:
            Shape embedding of shape (batch_size, num_nodes, shape_embedding_dim)
        """
        if t_idx.dim() > 1:
            t_idx = t_idx.squeeze(-1)
        if mask is not None:
            t_idx = t_idx[mask]
            if rotation_angle is not None:
                rotation_angle = rotation_angle[mask]
        return self._compute_embedding(t_idx, rotation_angle, detach_param=detach_param)

