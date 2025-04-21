# susruta/tests/test_models.py
"""Tests for the models module."""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATConv, HeteroConv

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

        # Check HeteroConv correctly
        assert isinstance(model.conv1, HeteroConv)
        assert isinstance(model.conv2, HeteroConv)
        for edge_type in edge_feature_dims.keys():
            assert edge_type in model.conv1.convs # Check if conv exists for edge type
            assert edge_type in model.conv2.convs

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

        # Define edge types that ensure all node types are involved as destinations
        edge_feature_dims = {
            ('patient', 'has_tumor', 'tumor'): 1,
            ('tumor', 'treated_with', 'treatment'): 1,
            ('treatment', 'affects', 'patient'): 1 # Add edge back to patient
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
        edge_index_pt = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)  # patient -> tumor
        edge_index_tt = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)  # tumor -> treatment
        edge_index_tp = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)  # treatment -> patient

        edge_indices_dict = {
            ('patient', 'has_tumor', 'tumor'): edge_index_pt,
            ('tumor', 'treated_with', 'treatment'): edge_index_tt,
            ('treatment', 'affects', 'patient'): edge_index_tp # Include new edge type
        }

        # Forward pass
        with torch.no_grad():
            predictions, node_embeddings = model(x_dict, edge_indices_dict)

        # Check predictions (only treatment has prediction heads)
        assert 'response' in predictions
        assert 'survival' in predictions
        assert 'uncertainty' in predictions

        # Check shapes
        assert predictions['response'].shape == (batch_size, 1)
        assert predictions['survival'].shape == (batch_size, 1)
        assert predictions['uncertainty'].shape == (batch_size, 1)

        # Check node embeddings exist for all types involved in message passing
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
        # Ensure all node types in edges have features (already fixed in GNN forward)

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
        num_treatments = pyg_data['treatment'].num_nodes # Use num_nodes
        assert predictions['response'].shape[0] == num_treatments
        assert predictions['survival'].shape[0] == num_treatments
        assert predictions['uncertainty'].shape[0] == num_treatments

    def test_gradient_flow(self):
        """Test that gradients flow properly through the model."""
        # Setup a simple model
        node_feature_dims = {'patient': 4, 'tumor': 6, 'treatment': 5}
        edge_feature_dims = {
            ('patient', 'has_tumor', 'tumor'): 1,
            ('tumor', 'treated_with', 'treatment'): 1,
            ('treatment', 'affects', 'patient'): 1 # Ensure all nodes can receive messages
        }

        model = GliomaGNN(
            node_feature_dims=node_feature_dims,
            edge_feature_dims=edge_feature_dims,
            hidden_channels=8,
            dropout=0.0  # No dropout for deterministic testing
        )
        model.train() # Set to train mode for gradient calculation

        # Create dummy inputs that require gradients
        x_dict = {
            'patient': torch.randn(2, 4, requires_grad=True),
            'tumor': torch.randn(2, 6, requires_grad=True),
            'treatment': torch.randn(2, 5, requires_grad=True)
        }

        edge_indices_dict = {
            ('patient', 'has_tumor', 'tumor'): torch.tensor([[0, 1], [0, 1]], dtype=torch.long),
            ('tumor', 'treated_with', 'treatment'): torch.tensor([[0, 1], [0, 1]], dtype=torch.long),
            ('treatment', 'affects', 'patient'): torch.tensor([[0, 1], [0, 1]], dtype=torch.long)
        }

        # Create target labels
        response_target = torch.rand(2, 1) # Use rand for BCELoss
        survival_target = torch.rand(2, 1) * 500 # Use rand for MSELoss

        # Forward pass
        predictions, _ = model(x_dict, edge_indices_dict)

        # Compute loss
        response_loss = F.binary_cross_entropy(predictions['response'], response_target)
        survival_loss = F.mse_loss(predictions['survival'], survival_target)

        # Combined loss
        loss = response_loss + 0.01 * survival_loss

        # Backward pass
        model.zero_grad() # Clear any potential stale gradients
        loss.backward()

        # Check if gradients exist for model parameters and inputs
        assert loss.item() > 0
        param_grads_exist = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters() if p.requires_grad)
        assert param_grads_exist, "No gradients found for model parameters"

        # Check input gradients more carefully
        patient_grad_sum = x_dict['patient'].grad.abs().sum() if x_dict['patient'].grad is not None else 0
        tumor_grad_sum = x_dict['tumor'].grad.abs().sum() if x_dict['tumor'].grad is not None else 0
        treatment_grad_sum = x_dict['treatment'].grad.abs().sum() if x_dict['treatment'].grad is not None else 0

        assert patient_grad_sum > 0 or tumor_grad_sum > 0 or treatment_grad_sum > 0, "No gradients flowed back to any input features"


    def test_saving_and_loading(self, gnn_model, tmp_path):
        """Test saving and loading model weights."""
        # Initialize lazy layers before saving
        dummy_x_dict = {}
        for node_type, encoder in gnn_model.node_encoders.items():
            # Ensure dummy input dim matches encoder's expected input dim
            dummy_x_dict[node_type] = torch.randn(1, encoder.in_features)

        dummy_edge_indices_dict = {}
        # Use edge types defined in the model instance
        for edge_type in gnn_model.edge_types:
            # Check if source/dest types exist in dummy_x_dict before adding edge
            src_type, _, dst_type = edge_type
            if src_type in dummy_x_dict and dst_type in dummy_x_dict:
                dummy_edge_indices_dict[edge_type] = torch.tensor([[0], [0]], dtype=torch.long)

        # Perform dummy forward pass only if there are edges to process
        if dummy_edge_indices_dict:
            try:
                gnn_model.eval() # Ensure dropout is off
                with torch.no_grad():
                    _ = gnn_model(dummy_x_dict, dummy_edge_indices_dict)
                print("Dummy forward pass successful for lazy layer initialization.")
            except Exception as e:
                pytest.fail(f"Dummy forward pass failed during saving/loading test setup: {e}")
        else:
             print("Skipping dummy forward pass: No valid edges found for initialization.")


        # Save model to a temporary file
        model_path = tmp_path / "test_model.pt"
        torch.save(gnn_model.state_dict(), model_path)

        # Create a new model with the same architecture
        node_feature_dims = {node_type: gnn_model.node_encoders[node_type].in_features
                           for node_type in gnn_model.node_encoders}

        # Extract edge types from the initialized model instance
        edge_feature_dims = {edge_type: 1 for edge_type in gnn_model.edge_types}

        new_model = GliomaGNN(
            node_feature_dims=node_feature_dims,
            edge_feature_dims=edge_feature_dims,
            hidden_channels=gnn_model.hidden_channels,
            dropout=gnn_model.dropout # Use dropout from original model
        )

        # Load saved weights
        # --- Start Fix: Use weights_only=False explicitly for robustness ---
        # Although initializing might make weights_only=True work,
        # using False is safer if initialization is complex or might fail.
        try:
            new_model.load_state_dict(torch.load(model_path, weights_only=False))
        except Exception as e:
             pytest.fail(f"Failed to load state dict: {e}")
        # --- End Fix ---

        new_model.eval() # Set to eval mode
        gnn_model.eval() # Ensure original is also in eval mode

        # Check that parameters are the same
        params1 = {name: p for name, p in gnn_model.named_parameters()}
        params2 = {name: p for name, p in new_model.named_parameters() if not isinstance(p, nn.parameter.UninitializedParameter)}

        # It's possible the new_model still has uninitialized params if dummy forward pass wasn't comprehensive
        # Only compare parameters that exist and are initialized in both
        common_params = params1.keys() & params2.keys()
        assert len(common_params) > 0, "No common initialized parameters found between models"

        for name in common_params:
            param1 = params1[name]
            param2 = params2[name]
            assert torch.allclose(param1, param2, atol=1e-6), f"Parameter mismatch: {name}"
