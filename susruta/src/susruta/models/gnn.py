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

# --- START FIX: Remove circular import ---
# Remove the following line which causes the circular import error:
# from ..models.gnn import GliomaGNN # This line is incorrect and causes the error
# --- END FIX ---


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
            Tuple of (predictions, node_embeddings after GNN layers)
        """
        # Store number of nodes for potential zero tensor creation
        num_nodes_dict = {nt: x.shape[0] for nt, x in x_dict.items()}
        # --- START FIX: Robust device detection ---
        if x_dict:
            # Get device from the first input tensor
            device = next(iter(x_dict.values())).device
        else:
            # Fallback: get device from the model's parameters
            try:
                device = next(self.parameters()).device
            except StopIteration:
                # If model has no parameters (very unlikely), default to CPU
                print("Warning: Could not determine device from input or parameters. Defaulting to CPU.")
                device = torch.device('cpu')
        # --- END FIX ---

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

        # Ensure h_dict has entries for all required node types before conv1
        h_dict = self._ensure_input_dict(h_dict, num_nodes_dict, device)

        # Filter edge_index_dict before conv1 (Enhanced)
        filtered_edge_index_dict1 = {}
        for edge_type, index in edge_index_dict.items():
            src_type, _, dst_type = edge_type
            # Check if BOTH types exist AND have nodes in the current feature dict
            if src_type in h_dict and dst_type in h_dict and \
            h_dict[src_type].shape[0] > 0 and h_dict[dst_type].shape[0] > 0:
                filtered_edge_index_dict1[edge_type] = index
        # Apply first HeteroConv layer using the filtered dict
        if filtered_edge_index_dict1:
            h_dict_1 = self.conv1(h_dict, filtered_edge_index_dict1)
        else:
            # If no valid edges, conv1 output is just the input features for relevant types
            # Only copy types that were actually destinations in the original edge_types
            dest_types_in_edges = {et[2] for et in self.edge_types}
            h_dict_1 = {nt: h.clone() for nt, h in h_dict.items() if nt in dest_types_in_edges}

        # Apply activation and dropout
        h_dict_1 = {node_type: F.relu(h) for node_type, h in h_dict_1.items()}
        h_dict_1 = {node_type: F.dropout(h, p=self.dropout, training=self.training)
                for node_type, h in h_dict_1.items()}

        # Ensure h_dict_1 has entries for all required node types before conv2
        h_dict_1 = self._ensure_input_dict(h_dict_1, num_nodes_dict, device)

        # Filter edge_index_dict before conv2 (Enhanced)
        filtered_edge_index_dict2 = {}
        for edge_type, index in edge_index_dict.items():
            src_type, _, dst_type = edge_type
            # Check if BOTH types exist AND have nodes in the feature dict for this layer
            if src_type in h_dict_1 and dst_type in h_dict_1 and \
            h_dict_1[src_type].shape[0] > 0 and h_dict_1[dst_type].shape[0] > 0:
                filtered_edge_index_dict2[edge_type] = index
        # Apply second HeteroConv layer using the filtered dict
        if filtered_edge_index_dict2:
            h_dict_2 = self.conv2(h_dict_1, filtered_edge_index_dict2)
        else:
            # If no valid edges, conv2 output is just the input features for relevant types
            dest_types_in_edges = {et[2] for et in self.edge_types}
            h_dict_2 = {nt: h.clone() for nt, h in h_dict_1.items() if nt in dest_types_in_edges}

        # Create predictions (based on h_dict_2, as heads operate on treatment embeddings)
        predictions = {}
        if 'treatment' in h_dict_2: # Check h_dict_2 for treatment embeddings specifically
            treatment_embeddings = h_dict_2['treatment']
            if treatment_embeddings.shape[0] > 0: # Only predict if embeddings exist
                response_pred = self.response_head(treatment_embeddings)
                predictions['response'] = response_pred
                survival_pred = self.survival_head(treatment_embeddings)
                predictions['survival'] = survival_pred
                uncertainty = self.uncertainty_head(treatment_embeddings)
                predictions['uncertainty'] = uncertainty

        # --- START FIX: Return h_dict_2 (embeddings after GNN layers) ---
        # Return predictions and the embeddings dictionary after GNN layers
        return predictions, h_dict_2
        # --- END FIX ---


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
