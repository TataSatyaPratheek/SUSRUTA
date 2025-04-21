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
from torch_geometric.nn import GATConv, GCNConv, SAGEConv


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
        
        # Node type-specific encoders
        self.node_encoders = nn.ModuleDict()
        for node_type, dim in node_feature_dims.items():
            self.node_encoders[node_type] = nn.Linear(dim, hidden_channels)
        
        # GNN layers - use GAT for better inductive bias
        self.conv1 = nn.ModuleDict()
        self.conv2 = nn.ModuleDict()
        
        # Create convolutions for each edge type
        if edge_feature_dims:
            for edge_type in edge_feature_dims.keys():
                # Format: (src_type, edge_type, dst_type)
                src_type, _, dst_type = edge_type
                
                # First convolution
                self.conv1[str(edge_type)] = GATConv(
                    (hidden_channels, hidden_channels), 
                    hidden_channels, 
                    heads=4,
                    dropout=dropout,
                    add_self_loops=False
                )
                
                # Second convolution
                self.conv2[str(edge_type)] = GATConv(
                    (hidden_channels * 4, hidden_channels * 4),  # 4 attention heads
                    hidden_channels,
                    heads=1,
                    dropout=dropout,
                    add_self_loops=False
                )
        
        # Prediction heads for different tasks
        self.response_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
            nn.Sigmoid()
        )
        
        self.survival_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
            nn.ReLU()  # Survival time is non-negative
        )
        
        # Uncertainty estimation head
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
            nn.Softplus()  # For positive uncertainty values
        )
    
    def forward(self, 
               x_dict: Dict[str, torch.Tensor], 
               edge_indices_dict: Dict[Tuple[str, str, str], torch.Tensor]
               ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Forward pass for memory-efficient heterogeneous GNN.
        
        Args:
            x_dict: Dictionary of node features by type
            edge_indices_dict: Dictionary of edge indices by type
            
        Returns:
            Tuple of (predictions, node_embeddings)
        """
        # Encode node features by type
        h_dict = {}
        for node_type, x in x_dict.items():
            if node_type in self.node_encoders:
                h_dict[node_type] = self.node_encoders[node_type](x)
            else:
                # Handle unexpected node types
                dim = x.size(1)
                self.node_encoders[node_type] = nn.Linear(dim, self.hidden_channels).to(x.device)
                h_dict[node_type] = self.node_encoders[node_type](x)
        
        # First message-passing layer with residual connection
        h_dict_1 = {node_type: h for node_type, h in h_dict.items()}
        
        for edge_type, edge_index in edge_indices_dict.items():
            src_type, _, dst_type = edge_type
            
            # Skip if edge type not in convolution dictionary
            if str(edge_type) not in self.conv1:
                continue
                
            # Apply convolution
            h_dict_1[dst_type] = h_dict_1.get(dst_type, 0) + self.conv1[str(edge_type)](
                (h_dict[src_type], h_dict[dst_type]), 
                edge_index
            )
        
        # Apply activation and dropout
        h_dict_1 = {node_type: F.relu(h) for node_type, h in h_dict_1.items()}
        h_dict_1 = {node_type: F.dropout(h, p=0.3, training=self.training) 
                   for node_type, h in h_dict_1.items()}
        
        # Second message-passing layer
        h_dict_2 = {node_type: h for node_type, h in h_dict_1.items()}
        
        for edge_type, edge_index in edge_indices_dict.items():
            src_type, _, dst_type = edge_type
            
            # Skip if edge type not in convolution dictionary
            if str(edge_type) not in self.conv2:
                continue
                
            # Apply convolution
            h_dict_2[dst_type] = h_dict_2.get(dst_type, 0) + self.conv2[str(edge_type)](
                (h_dict_1[src_type], h_dict_1[dst_type]), 
                edge_index
            )
        
        # Apply final activation
        h_dict_2 = {node_type: F.relu(h) for node_type, h in h_dict_2.items()}
        
        # Create predictions
        predictions = {}
        
        # Predict treatment outcomes if treatment nodes exist
        if 'treatment' in h_dict_2:
            treatment_embeddings = h_dict_2['treatment']
            
            # Response prediction (binary classification)
            response_pred = self.response_head(treatment_embeddings)
            predictions['response'] = response_pred
            
            # Survival prediction (regression)
            survival_pred = self.survival_head(treatment_embeddings)
            predictions['survival'] = survival_pred
            
            # Uncertainty estimation
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
            device = next(self.parameters()).device
        
        # Set model to evaluation mode
        self.eval()
        
        # Move data to device
        x_dict = {node_type: data[node_type].x.to(device) for node_type in data.node_types()}
        edge_indices_dict = {
            edge_type: data[edge_type].edge_index.to(device) 
            for edge_type in data.edge_types()
        }
        
        # Make predictions
        with torch.no_grad():
            predictions, _ = self.forward(x_dict, edge_indices_dict)
        
        return predictions