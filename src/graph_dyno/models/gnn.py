import torch
from omegaconf import DictConfig
from graph_dyno.models.mlp import MLP

class GNN(torch.nn.Module):
    def __init__(self, output_dim, spatial_dim, embedding_dim, hidden_dim, hidden_layers, activation, periodic_boundary=False, domain_size=None, device: str | None = None):
        super(GNN, self).__init__()

        # MLP for interactions
        self.output_dim = output_dim
        self.spatial_dim = spatial_dim

        self.periodic_boundary = periodic_boundary
        self.domain_size = domain_size

        # MLP for interactions between particles
        # input is (x_ij, b_i, b_j, d_ij)
        self.mlp_interaction = MLP(
            input_size= spatial_dim + 2 * embedding_dim + 1,
            hidden_size=hidden_dim,
            num_hidden_layers=hidden_layers,
            output_size=output_dim,
            activation=activation,
        ).to(device)



    def _compute_pairwise_distance(self, x_i, x_j):
        """Compute pairwise distance and relative position vectors.

        Args:
            x_i: Tensor of shape (num_edges, spatial_dim)
            x_j: Tensor of shape (num_edges, spatial_dim)
        Returns:
            x_ij: Relative position vectors (num_edges, spatial_dim)
            d_ij: Pairwise distances (num_edges, 1)
        """
        x_ij = x_j - x_i  # (num_edges, spatial_dim)

        if self.periodic_boundary:
            x_ij[..., :2] = x_ij[..., :2] - self.domain_size * torch.round(x_ij[..., :2] / self.domain_size)

        d_ij = torch.norm(x_ij, dim=-1, keepdim=True)  # (num_edges, 1)
        return x_ij, d_ij


    def forward(self, x, b, W_interactions):

        # Extract all edges from all batches at once
        # W_interactions is a stacked sparse tensor with shape (batch_size, num_nodes, num_nodes)
        # For stacked sparse tensors, sparse_dim=3, and indices has shape (3, total_edges)
        # where indices[0] = batch_idx, indices[1] = src_node, indices[2] = tgt_node
        batch_size, num_nodes, feature_dim = x.shape

        interactions = torch.zeros(batch_size, num_nodes, self.output_dim, device=x.device)

        indices = W_interactions._indices()  # (3, total_edges)
        edge_values = W_interactions._values()  # (total_edges,)

        if indices.shape[1] > 0:  # if there are any edges
            # Extract batch, source, and target indices
            batch_indices = indices[0]  # (total_edges,)
            src_nodes = indices[1]      # (total_edges,)
            tgt_nodes = indices[2]      # (total_edges,)
            
            # Get positions and embeddings for all edges at once using advanced indexing
            x_i = x[batch_indices, src_nodes, :]  # (total_edges, feature_dim)
            x_j = x[batch_indices, tgt_nodes, :]  # (total_edges, feature_dim)
            b_i = b[batch_indices, src_nodes, :]  # (total_edges, embedding_dim)
            b_j = b[batch_indices, tgt_nodes, :]  # (total_edges, embedding_dim)
            
            x_ij, d_ij = self._compute_pairwise_distance(x_i[..., :self.spatial_dim], x_j[..., :self.spatial_dim])  # (total_edges, spatial_dim), (total_edges, 1)
            # Concatenate features for interaction MLP
            input_interaction = torch.cat([x_ij, b_i, b_j, d_ij], dim=-1)  # (total_edges, input_size)

            # Compute interactions for all edges in one forward pass
            interaction_features = self.mlp_interaction(input_interaction)  # (total_edges, self.interaction_dim)

            # Weight by edge values
            weighted_interactions = interaction_features * edge_values.unsqueeze(-1)  # (total_edges, self.interaction_dim)

            # Aggregate interactions back to nodes using scatter_add
            # Create flat index: batch_idx * num_nodes + src_node
            flat_indices = batch_indices * num_nodes + src_nodes  # (total_edges,)
            
            # Flatten interactions tensor for scatter_add
            interactions_flat = interactions.view(-1, self.output_dim)  # (batch_size * num_nodes, interaction_dim)
            
            # Scatter add the weighted interactions
            interactions_flat.scatter_add_(0, flat_indices.unsqueeze(-1).expand(-1, self.output_dim), weighted_interactions)
            
            # Reshape back
            interactions = interactions_flat.view(batch_size, num_nodes, self.output_dim)

        return interactions

    def generate(self, x, b, W_interactions):

        num_nodes, feature_dim = x.shape

        # Initialize interactions tensor
        interactions = torch.zeros(num_nodes, self.output_dim, device=x.device)

        indices = W_interactions._indices()  # (2, total_edges)
        edge_values = W_interactions._values()  # (total_edges,)

        if indices.shape[1] > 0:  # if there are any edges
            # Extract batch, source, and target indices
            src_nodes = indices[0]      # (total_edges,)
            tgt_nodes = indices[1]      # (total_edges,)
            
            # Get positions and embeddings for all edges at once using advanced indexing
            x_i = x[src_nodes, :]  # (total_edges, feature_dim)
            x_j = x[tgt_nodes, :]  # (total_edges, feature_dim)
            b_i = b[src_nodes, :]  # (total_edges, embedding_dim)
            b_j = b[tgt_nodes, :]  # (total_edges, embedding_dim)
            
            x_ij, d_ij = self._compute_pairwise_distance(x_i[..., :self.spatial_dim], x_j[..., :self.spatial_dim])  # (total_edges, 1)
            # Concatenate features for interaction MLP
            input_interaction = torch.cat([x_ij, b_i, b_j, d_ij], dim=-1)  # (total_edges, output_dim)

            # Compute interactions for all edges in one forward pass
            interaction_features = self.mlp_interaction(input_interaction)  # (total_edges, output_dim)

            # Weight by edge values
            weighted_interactions = interaction_features * edge_values.unsqueeze(-1)  # (total_edges, output_dim)

            # Aggregate interactions back to nodes using scatter_add
            # Create flat index: batch_idx * num_nodes + src_node
            flat_indices = src_nodes
            
            # Flatten interactions tensor for scatter_add
            interactions_flat = interactions.view(-1, self.output_dim)  # (num_nodes, output_dim)
            
            # Scatter add the weighted interactions
            interactions_flat.scatter_add_(0, flat_indices.unsqueeze(-1).expand(-1, self.output_dim), weighted_interactions)
            
            # Reshape back
            interactions = interactions_flat.view(num_nodes, self.output_dim)

        return interactions