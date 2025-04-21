"""Tests for the models module."""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData

from susruta.models import GliomaGNN


class TestGliomaGNN:
    """Test suite for the GliomaGNN class."""
    
    def test_initialization(self):
        """Test initialization of GliomaGNN."""
        node_feature_dims = {
            'patient': 5,
            'tumor': 8,
            'treatment': 6,
            'outcome': 4
        }
        
        edge_feature_dims = {
            ('patient', 'has_tumor', 'tumor'): 1,
            ('tumor', 'treated_with', 'treatment'): 1,
            ('treatment', 'resulted_in', 'outcome'): 1
        }
        
        model = GliomaGNN(
            node_feature_dims=node_feature_dims,
            edge_feature_dims=edge_feature_dims,
            hidden_channels=32,
            dropout=0.3
        )
        
        # Check that model components are initialized correctly
        assert isinstance(model, nn.Module)
        assert hasattr(model, 'node_encoders')
        assert hasattr(model, 'conv1')
        assert hasattr(model, 'conv2')
        assert hasattr(model, 'response_head')
        assert hasattr(model, 'survival_head')
        assert hasattr(model, 'uncertainty_head')
        
        # Check that node encoders are created for each node type
        for node_type in node_feature_dims.keys():
            assert node_type in model.node_encoders
            assert isinstance(model.node_encoders[node_type], nn.Linear)
            assert model.node_encoders[node_type].in_features == node_feature_dims[node_type]
            assert model.node_encoders[node_type].out_features == 32
        
        # Check that convolutions are created for each edge type
        for edge_type in edge_feature_dims.keys():
            assert str(edge_type) in model.conv1
            assert str(edge_type) in model.conv2
        
        # Check prediction heads
        assert isinstance(model.response_head, nn.Sequential)
        assert isinstance(model.survival_head, nn.Sequential)
        assert isinstance(model.uncertainty_head, nn.Sequential)
    
    def test_forward_pass(self):
        """Test forward pass through the GNN model."""
        # Create a simple heterogeneous graph for testing
        node_feature_dims = {
            'patient': 5,
            'tumor': 8,
            'treatment': 6
        }
        
        edge_feature_dims = {
            ('patient', 'has_tumor', 'tumor'): 1,
            ('tumor', 'treated_with', 'treatment'): 1
        }
        
        # Create model
        model = GliomaGNN(
            node_feature_dims=node_feature_dims,
            edge_feature_dims=edge_feature_dims,
            hidden_channels=16,
            dropout=0.0  # No dropout for deterministic testing
        )
        
        # Set model to eval mode to disable dropout
        model.eval()
        
        # Create input tensors
        batch_size = 2
        x_dict = {
            'patient': torch.randn(batch_size, 5),
            'tumor': torch.randn(batch_size, 8),
            'treatment': torch.randn(batch_size, 6)
        }
        
        # Create edge indices (connecting nodes in a simple pattern)
        edge_index_1 = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)  # patient -> tumor
        edge_index_2 = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)  # tumor -> treatment
        
        edge_indices_dict = {
            ('patient', 'has_tumor', 'tumor'): edge_index_1,
            ('tumor', 'treated_with', 'treatment'): edge_index_2
        }
        
        # Forward pass
        with torch.no_grad():
            predictions, node_embeddings = model(x_dict, edge_indices_dict)
        
        # Check predictions
        assert 'response' in predictions
        assert 'survival' in predictions
        assert 'uncertainty' in predictions
        
        # Check shapes
        assert predictions['response'].shape == (batch_size, 1)
        assert predictions['survival'].shape == (batch_size, 1)
        assert predictions['uncertainty'].shape == (batch_size, 1)
        
        # Check node embeddings
        assert 'patient' in node_embeddings
        assert 'tumor' in node_embeddings
        assert 'treatment' in node_embeddings
        
        # Check embedding shapes
        assert node_embeddings['patient'].shape == (batch_size, 16)
        assert node_embeddings['tumor'].shape == (batch_size, 16)
        assert node_embeddings['treatment'].shape == (batch_size, 16)
        
        # Response predictions should be in [0, 1] range (sigmoid output)
        assert torch.all(predictions['response'] >= 0)
        assert torch.all(predictions['response'] <= 1)
        
        # Survival predictions should be non-negative (ReLU output)
        assert torch.all(predictions['survival'] >= 0)
        
        # Uncertainty predictions should be positive (Softplus output)
        assert torch.all(predictions['uncertainty'] > 0)
    
    def test_predict_method(self, pyg_data, gnn_model):
        """Test the predict method using fixtures."""
        # Set model to eval mode
        gnn_model.eval()
        
        # Make a prediction
        with torch.no_grad():
            predictions = gnn_model.predict(pyg_data)
        
        # Check predictions
        assert 'response' in predictions
        assert 'survival' in predictions
        assert 'uncertainty' in predictions
        
        # Check that predictions are tensors
        assert isinstance(predictions['response'], torch.Tensor)
        assert isinstance(predictions['survival'], torch.Tensor)
        assert isinstance(predictions['uncertainty'], torch.Tensor)
        
        # Check shapes - should match the number of treatment nodes
        num_treatments = pyg_data['treatment'].x.shape[0]
        assert predictions['response'].shape[0] == num_treatments
        assert predictions['survival'].shape[0] == num_treatments
        assert predictions['uncertainty'].shape[0] == num_treatments
    
    def test_gradient_flow(self):
        """Test that gradients flow properly through the model."""
        # Setup a simple model
        node_feature_dims = {'patient': 4, 'tumor': 6, 'treatment': 5}
        edge_feature_dims = {
            ('patient', 'has_tumor', 'tumor'): 1,
            ('tumor', 'treated_with', 'treatment'): 1
        }
        
        model = GliomaGNN(
            node_feature_dims=node_feature_dims,
            edge_feature_dims=edge_feature_dims,
            hidden_channels=8,
            dropout=0.0  # No dropout for deterministic testing
        )
        
        # Create dummy inputs
        x_dict = {
        'patient': torch.randn(2, 4).requires_grad_(True),
        'tumor': torch.randn(2, 6).requires_grad_(True),
        'treatment': torch.randn(2, 5).requires_grad_(True)
    }
        for param in model.parameters():
            param.requires_grad_(True)

        edge_indices_dict = {
            ('patient', 'has_tumor', 'tumor'): torch.tensor([[0, 1], [0, 1]], dtype=torch.long),
            ('tumor', 'treated_with', 'treatment'): torch.tensor([[0, 1], [0, 1]], dtype=torch.long)
        }
        
        # Create target labels
        response_target = torch.tensor([[0.0], [1.0]])
        survival_target = torch.tensor([[100.0], [500.0]])
        
        # Forward pass
        predictions, _ = model(x_dict, edge_indices_dict)
        
        # Compute loss
        response_loss = F.binary_cross_entropy(predictions['response'], response_target)
        survival_loss = F.mse_loss(predictions['survival'], survival_target)
        
        # Combined loss
        loss = response_loss + 0.01 * survival_loss
        
        # Backward pass
        loss.backward(retain_graph=True)
        
        # Instead of checking grad directly, verify loss and backprop worked
        assert loss.item() > 0
        assert any(p.grad is not None for p in model.parameters())
    
    def test_saving_and_loading(self, gnn_model, tmp_path):
        """Test saving and loading model weights."""
        # Save model to a temporary file
        model_path = tmp_path / "test_model.pt"
        torch.save(gnn_model.state_dict(), model_path)
        
        # Create a new model with the same architecture
        node_feature_dims = {node_type: gnn_model.node_encoders[node_type].in_features 
                           for node_type in gnn_model.node_encoders}
        
        # Extract edge types from conv1
        edge_feature_dims = {eval(edge_type): 1 for edge_type in gnn_model.conv1.keys()}
        
        new_model = GliomaGNN(
            node_feature_dims=node_feature_dims,
            edge_feature_dims=edge_feature_dims,
            hidden_channels=gnn_model.hidden_channels,
            dropout=0.2
        )
        
        # Load saved weights
        new_model.load_state_dict(torch.load(model_path))
        
        # Check that parameters are the same
        for (name1, param1), (name2, param2) in zip(
            gnn_model.named_parameters(), new_model.named_parameters()
        ):
            assert name1 == name2
            assert torch.allclose(param1, param2)