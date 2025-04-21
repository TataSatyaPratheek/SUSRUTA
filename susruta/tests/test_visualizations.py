# susruta/tests/test_visualizations.py
"""Tests for the visualization and explanation module."""

import pytest
import torch
import numpy as np
import networkx as nx
from unittest.mock import patch, MagicMock

# Import classes directly (fixtures are now in conftest.py)
from susruta.viz import ExplainableGliomaTreatment
from susruta.models import GliomaGNN
from torch_geometric.data import HeteroData # Import HeteroData

# Fixtures like gnn_model, pyg_data, knowledge_graph, treatment_explainer
# are automatically injected by pytest from conftest.py


class TestExplainableGliomaTreatment:
    """Test suite for the ExplainableGliomaTreatment class."""

    def test_initialization(self, treatment_explainer, gnn_model, pyg_data, knowledge_graph):
        """Test initialization of ExplainableGliomaTreatment."""
        assert treatment_explainer.model == gnn_model
        # Check data object identity or structure
        assert isinstance(treatment_explainer.data, HeteroData) # Check type
        assert isinstance(treatment_explainer.original_graph, nx.MultiDiGraph)
        assert hasattr(treatment_explainer, 'device')
        assert not treatment_explainer.model.training # Should be in eval mode

    def test_explain_treatment_prediction(self, treatment_explainer, pyg_data):
        """Test explanation generation for treatment prediction."""
        treatment_idx_to_test = 0
        if 'treatment' not in treatment_explainer.data.node_types or treatment_explainer.data['treatment'].num_nodes <= treatment_idx_to_test:
             pytest.skip("Skipping explanation test: No valid treatment node at index 0.")
        if not hasattr(treatment_explainer.data['treatment'], 'x') or treatment_explainer.data['treatment'].x is None:
             pytest.skip("Skipping explanation test: Treatment node features are missing.")
        if treatment_explainer.data['treatment'].num_node_features == 0:
             pytest.skip("Skipping explanation test: Treatment node features have dimension 0.")

        treatment_feature_dim = treatment_explainer.data['treatment'].num_node_features

        # Patch model's forward and zero_grad.
        with patch.object(treatment_explainer.model, 'zero_grad') as mock_zero_grad, \
             patch.object(treatment_explainer.model, 'forward') as mock_forward, \
             patch('torch.Tensor.backward') as mock_backward: # Mock backward globally

            # Configure mock_forward to return predictions requiring grad
            num_treatments = treatment_explainer.data['treatment'].num_nodes
            mock_response_preds_all = torch.rand(num_treatments, 1, requires_grad=True, device=treatment_explainer.device)
            mock_survival_preds_all = torch.rand(num_treatments, 1, requires_grad=True, device=treatment_explainer.device) * 500
            mock_uncertainty_preds_all = torch.rand(num_treatments, 1, requires_grad=True, device=treatment_explainer.device) * 0.5

            predictions_output = {
                'response': mock_response_preds_all,
                'survival': mock_survival_preds_all,
                'uncertainty': mock_uncertainty_preds_all
            }
            mock_forward.return_value = (predictions_output, {}) # Return dummy embeddings

            # Simulate gradient assignment during backward mock
            # This is simplified: assumes backward is called and assigns *some* non-zero grad
            def assign_grad_effect(*args, **kwargs):
                 input_tensor = args[0] # The tensor backward is called on
                 if hasattr(input_tensor, 'grad_fn') and input_tensor.grad_fn is not None:
                     # Find the treatment input tensor in the graph history (simplified)
                     # In a real scenario, this is complex. We rely on the tested function's logic.
                     # For the test, we just need to ensure the grad attribute exists later.
                     if treatment_explainer.data['treatment'].x.grad is None:
                          treatment_explainer.data['treatment'].x.grad = torch.rand_like(treatment_explainer.data['treatment'].x) * 1e-3 # Small random grads
                     else:
                          treatment_explainer.data['treatment'].x.grad += torch.rand_like(treatment_explainer.data['treatment'].x) * 1e-3
            mock_backward.side_effect = assign_grad_effect

            treatment_id = treatment_explainer.data['treatment'].original_ids[treatment_idx_to_test]

            # --- Generate explanation ---
            explanation = treatment_explainer.explain_treatment_prediction(treatment_idx_to_test, k=treatment_feature_dim) # Get all features

            # --- Assertions ---
            assert explanation is not None
            assert explanation['treatment_id'] == treatment_id
            assert 'predictions' in explanation
            assert 'feature_importance' in explanation
            assert explanation['predictions']['response_probability'] == pytest.approx(mock_response_preds_all[treatment_idx_to_test].item())
            assert explanation['predictions']['survival_days'] == pytest.approx(mock_survival_preds_all[treatment_idx_to_test].item())
            assert explanation['predictions']['uncertainty'] == pytest.approx(mock_uncertainty_preds_all[treatment_idx_to_test].item())

            assert 'response' in explanation['feature_importance']
            assert 'survival' in explanation['feature_importance']
            # Check if importance values are generated (might be zero due to mock)
            assert len(explanation['feature_importance']['response']) <= treatment_feature_dim
            assert len(explanation['feature_importance']['survival']) <= treatment_feature_dim

            # --- Start Fix: Get expected feature names from model ---
            # This assumes the explain method uses names consistent with the model's input layer
            expected_treatment_dim = treatment_explainer.model.node_encoders['treatment'].in_features
            # Get sorted numerical attribute names from the *first* treatment node in the original graph
            # This is still an approximation, assuming all treatment nodes have the same attributes
            expected_feature_names = []
            if treatment_explainer.original_graph:
                 first_treatment_node = next((n for n, d in treatment_explainer.original_graph.nodes(data=True) if d.get('type') == 'treatment'), None)
                 if first_treatment_node:
                     node_data = treatment_explainer.original_graph.nodes[first_treatment_node]
                     numeric_keys = sorted([
                         key for key, value in node_data.items()
                         if key != 'type' and isinstance(value, (int, float, np.number)) and not np.isnan(value)
                     ])
                     expected_feature_names = numeric_keys

            # Pad/truncate to match model's expected dimension
            if len(expected_feature_names) < expected_treatment_dim:
                 expected_feature_names.extend([f'feature_{i}' for i in range(len(expected_feature_names), expected_treatment_dim)])
            elif len(expected_feature_names) > expected_treatment_dim:
                 expected_feature_names = expected_feature_names[:expected_treatment_dim]
            # --- End Fix ---

            response_feat_names = [f[0] for f in explanation['feature_importance']['response']]
            survival_feat_names = [f[0] for f in explanation['feature_importance']['survival']]

            # Check if the extracted names match the expected names
            assert set(response_feat_names).issubset(set(expected_feature_names))
            assert set(survival_feat_names).issubset(set(expected_feature_names))

            # Check mock calls
            mock_forward.assert_called_once()
            assert mock_zero_grad.call_count >= 2
            assert mock_backward.call_count >= 2


    def test_get_treatment_comparison(self, treatment_explainer):
        """Test generation of treatment comparisons."""
        if treatment_explainer.data['treatment'].num_nodes < 2:
             pytest.skip("Skipping comparison test: Need at least 2 treatment nodes.")

        with patch.object(treatment_explainer, 'explain_treatment_prediction') as mock_explain:
            mock_exp1 = {'treatment_id': 'treatment_1', 'predictions': {'response_probability': 0.8, 'survival_days': 500, 'uncertainty': 0.1}, 'feature_importance': {'response': [('feat_A', 0.5), ('feat_B', 0.3)], 'survival': [('feat_B', 0.6)]}, 'treatment_attributes': {}}
            mock_exp2 = {'treatment_id': 'treatment_2', 'predictions': {'response_probability': 0.7, 'survival_days': 450, 'uncertainty': 0.2}, 'feature_importance': {'response': [('feat_B', 0.6), ('feat_C', 0.1)], 'survival': [('feat_D', 0.3)]}, 'treatment_attributes': {}}
            mock_explain.side_effect = [mock_exp1, mock_exp2]

            idx1, idx2 = 0, 1
            comparison = treatment_explainer.get_treatment_comparison([idx1, idx2])

            assert mock_explain.call_count == 2
            mock_explain.assert_any_call(idx1)
            mock_explain.assert_any_call(idx2)

            assert 'treatments' in comparison and len(comparison['treatments']) == 2
            assert comparison['treatments'][0]['treatment_id'] == 'treatment_1'
            assert comparison['treatments'][1]['treatment_id'] == 'treatment_2'

            # Check corrected comparison logic
            assert 'common_response_features' in comparison
            assert comparison['common_response_features'] == {'feat_B'}
            assert 'common_survival_features' in comparison
            assert comparison['common_survival_features'] == set()

            assert 'all_response_features' in comparison
            assert comparison['all_response_features'] == {'feat_A', 'feat_B', 'feat_C'}
            assert 'all_survival_features' in comparison
            assert comparison['all_survival_features'] == {'feat_B', 'feat_D'}

            assert 'differentiating_response_features' in comparison
            assert set(comparison['differentiating_response_features']) == {'feat_A', 'feat_C'}
            assert 'differentiating_survival_features' in comparison
            assert set(comparison['differentiating_survival_features']) == {'feat_B', 'feat_D'}

            assert 'common_features' in comparison
            assert comparison['common_features'] == {'feat_B'}
            assert 'differentiating_features' in comparison
            assert comparison['differentiating_features'] == {'feat_A', 'feat_C', 'feat_B', 'feat_D'}


    def test_generate_natural_language_explanation(self, treatment_explainer):
        """Test generation of natural language explanations."""
        explanation = {
            'treatment_id': 'treatment_1',
            'treatment_attributes': {'category': 'surgery', 'dose': None, 'duration': 1},
            'patient_id': 'patient_1',
            'tumor_id': 'tumor_1',
            'predictions': {'response_probability': 0.8, 'survival_days': 500, 'uncertainty': 0.1},
            'feature_importance': {
                'response': [('intensity', 0.5), ('duration', 0.1)],
                'survival': [('intensity', 0.4)]
            }
        }
        explanation_text = treatment_explainer.generate_natural_language_explanation(explanation)
        assert isinstance(explanation_text, str) and len(explanation_text) > 0
        assert 'surgery treatment' in explanation_text.lower()
        assert '80.0%' in explanation_text
        assert '500 days' in explanation_text
        assert 'high confidence' in explanation_text.lower()
        assert 'uncertainty score: 0.100' in explanation_text
        # --- Start Fix: Add period to assertion ---
        assert 'influencing response prediction: intensity (0.500), duration (0.100).' in explanation_text
        assert 'influencing survival prediction: intensity (0.400).' in explanation_text
        # --- End Fix ---
        assert 'Treatment details considered: duration: 1.' in explanation_text
        assert 'dose' not in explanation_text
