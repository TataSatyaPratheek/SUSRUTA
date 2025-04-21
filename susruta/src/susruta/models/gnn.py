# susruta/src/susruta/models/gnn.py
"""
Graph Neural Network models for glioma treatment outcome prediction.

Implements memory-efficient GNN architectures for heterogeneous graphs.
"""

from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, HeteroConv


class GliomaGNN(torch.nn.Module):
    """Memory-efficient GNN for glioma treatment outcome prediction."""

    def __init__(self,
                node_feature_dims: Dict[str, int],
                edge_feature_dims: Optional[Dict[Tuple[str, str, str], int]] = None,
                hidden_channels: int = 32,
                dropout: float = 0.3):
        """
        Initialize the GNN model.

        Args:
            node_feature_dims: Dictionary mapping node types to feature dimensions
            edge_feature_dims: Dictionary mapping edge types to feature dimensions
            hidden_channels: Number of hidden channels
            dropout: Dropout rate
        """
        super().__init__()

        self.node_feature_dims = node_feature_dims
        self.hidden_channels = hidden_channels
        self.dropout = dropout
        self.edge_types = list(edge_feature_dims.keys()) if edge_feature_dims else []

        # Node type-specific encoders
        self.node_encoders = nn.ModuleDict()
        for node_type, dim in node_feature_dims.items():
             # Ensure input dimension is at least 1
            in_dim = max(dim, 1)
            self.node_encoders[node_type] = nn.Linear(in_dim, hidden_channels)

        # GNN layers using HeteroConv for clarity
        self.conv1 = HeteroConv({
            edge_type: GATConv((-1, -1), hidden_channels, heads=4, dropout=dropout, add_self_loops=False)
            for edge_type in self.edge_types
        }, aggr='sum') # Use 'sum' aggregation

        self.conv2 = HeteroConv({
             edge_type: GATConv((-1, -1), hidden_channels, heads=1, dropout=dropout, add_self_loops=False)
             for edge_type in self.edge_types
        }, aggr='sum')


        # Prediction heads for different tasks
        self.response_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2), # Smaller intermediate layer
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 1),
            nn.Sigmoid()
        )

        self.survival_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 1),
            nn.ReLU()  # Survival time is non-negative
        )

        # Uncertainty estimation head
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 1),
            nn.Softplus()  # For positive uncertainty values
        )

    def _ensure_input_dict(self,
                           input_dict: Dict[str, torch.Tensor],
                           num_nodes_dict: Dict[str, int],
                           device: torch.device
                           ) -> Dict[str, torch.Tensor]:
        """Ensure input dict has entries for all node types involved in edges."""
        # Determine all node types involved in the model's edge types
        required_node_types = set()
        for edge_type in self.edge_types:
            required_node_types.add(edge_type[0]) # Source
            required_node_types.add(edge_type[2]) # Destination

        output_dict = input_dict.copy()
        for node_type in required_node_types:
            if node_type not in output_dict:
                # If node type is missing entirely, add a zero tensor
                num_nodes = num_nodes_dict.get(node_type, 0) # Get num nodes if available
                # Determine expected feature dimension (usually hidden_channels for intermediate)
                # For the initial input (h_dict), it's hidden_channels after encoding
                expected_dim = self.hidden_channels
                print(f"Warning: Adding zero tensor for missing node type '{node_type}' in GNN forward pass (shape: ({num_nodes}, {expected_dim})).")
                output_dict[node_type] = torch.zeros((num_nodes, expected_dim), device=device)
            elif output_dict[node_type].shape[0] > 0 and output_dict[node_type].shape[1] == 0:
                 # If node type exists but has 0 features, add dummy feature
                 expected_dim = self.hidden_channels
                 print(f"Warning: Adding zero tensor for node type '{node_type}' with 0 features in GNN forward pass (shape: ({output_dict[node_type].shape[0]}, {expected_dim})).")
                 output_dict[node_type] = torch.zeros((output_dict[node_type].shape[0], expected_dim), device=device)


        return output_dict


    def forward(self,
            x_dict: Dict[str, torch.Tensor],
            edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor] # Corrected type hint
            ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Forward pass for memory-efficient heterogeneous GNN.

        Args:
            x_dict: Dictionary of node features by type
            edge_index_dict: Dictionary of edge indices by type

        Returns:
            Tuple of (predictions, node_embeddings)
        """
        # Store number of nodes for potential zero tensor creation
        num_nodes_dict = {nt: x.shape[0] for nt, x in x_dict.items()}
        device = next(iter(x_dict.values())).device if x_dict else self.conv1.convs[self.edge_types[0]].lin.weight.device # Get device

        # Encode node features by type
        h_dict = {}
        for node_type, x in x_dict.items():
            if node_type not in self.node_encoders: continue # Skip if no encoder defined

            # Ensure input dimension matches encoder, handle 0-dim input
            expected_dim = self.node_encoders[node_type].in_features
            if x.shape[1] == 0 and expected_dim == 1:
                 x = torch.zeros((x.shape[0], 1), device=x.device, dtype=x.dtype)
            elif x.shape[1] != expected_dim:
                 if x.shape[1] < expected_dim:
                     padding = torch.zeros((x.shape[0], expected_dim - x.shape[1]), device=x.device, dtype=x.dtype)
                     x = torch.cat([x, padding], dim=1)
                 else:
                     x = x[:, :expected_dim]

            h_dict[node_type] = self.node_encoders[node_type](x)

        # --- Start Fix: Ensure h_dict has entries for all required node types ---
        h_dict = self._ensure_input_dict(h_dict, num_nodes_dict, device)
        # --- End Fix ---

        # Apply first HeteroConv layer
        h_dict_1 = self.conv1(h_dict, edge_index_dict)

        # Apply activation and dropout
        h_dict_1 = {node_type: F.relu(h) for node_type, h in h_dict_1.items()}
        h_dict_1 = {node_type: F.dropout(h, p=self.dropout, training=self.training)
                for node_type, h in h_dict_1.items()}

        # --- Start Fix: Ensure h_dict_1 has entries for all required node types ---
        h_dict_1 = self._ensure_input_dict(h_dict_1, num_nodes_dict, device)
        # --- End Fix ---

        # Apply second HeteroConv layer
        h_dict_2 = self.conv2(h_dict_1, edge_index_dict)

        # Create predictions
        predictions = {}

        # Predict treatment outcomes if treatment nodes exist
        if 'treatment' in h_dict_2:
            treatment_embeddings = h_dict_2['treatment']
            if treatment_embeddings.shape[0] > 0: # Only predict if embeddings exist
                response_pred = self.response_head(treatment_embeddings)
                predictions['response'] = response_pred
                survival_pred = self.survival_head(treatment_embeddings)
                predictions['survival'] = survival_pred
                uncertainty = self.uncertainty_head(treatment_embeddings)
                predictions['uncertainty'] = uncertainty

        return predictions, h_dict_2

    def predict(self, data: HeteroData, device: Optional[torch.device] = None) -> Dict[str, torch.Tensor]:
        """
        Make predictions on a HeteroData object.

        Args:
            data: PyTorch Geometric HeteroData object
            device: Optional device to run prediction on

        Returns:
            Dictionary of predictions
        """
        # Select device
        if device is None:
            try:
                device = next(self.parameters()).device
            except StopIteration: # Handle case with no parameters (unlikely)
                 device = torch.device('cpu')
        else:
             self.to(device) # Ensure model is on the specified device

        # Set model to evaluation mode
        self.eval()

        # Move data to device
        x_dict = {node_type: data[node_type].x.to(device)
                  for node_type in data.node_types if hasattr(data[node_type], 'x')}

        edge_index_dict = {}
        for edge_type_tuple in data.edge_types:
             if hasattr(data[edge_type_tuple], 'edge_index'):
                 edge_index_dict[edge_type_tuple] = data[edge_type_tuple].edge_index.to(device)


        # Make predictions
        with torch.no_grad():
            predictions, _ = self.forward(x_dict, edge_index_dict)

        return predictions
