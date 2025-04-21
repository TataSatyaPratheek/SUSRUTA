"""Tests for the visualization and explanation module."""

import pytest
import torch
import numpy as np
import networkx as nx
from unittest.mock import patch, MagicMock

from susruta.viz import ExplainableGliomaTreatment


class TestExplainableGliomaTreatment:
    """Test suite for the ExplainableGliomaTreatment class."""
    
    def test_initialization(self, gnn_model, pyg_data, knowledge_graph):
        """Test initialization of ExplainableGliomaTreatment."""
        explainer = ExplainableGliomaTreatment(gnn_model, pyg_data, knowledge_graph)
        
        assert explainer.model == gnn_model
        assert explainer.data == pyg_data
        assert explainer.original_graph == knowledge_graph
        assert hasattr(explainer, 'device')
    
    @patch('torch.autograd.grad')
    def test_explain_treatment_prediction(self, mock_grad, treatment_explainer, pyg_data):
        """Test explanation generation for treatment prediction."""
        # Create a mock clone tensor with grad
        mock_x = torch.zeros(10, requires_grad=True)
        mock_grad_value = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        
        # Set up mocks for tensor operations
        with patch('torch.clone', return_value=mock_x), \
             patch.object(treatment_explainer.model, 'zero_grad') as mock_zero_grad, \
             patch.object(treatment_explainer.model, 'forward') as mock_forward, \
             patch.object(mock_x, 'grad', create=True, new=mock_grad_value):
            
            # Mock forward pass outputs
            predictions = {
                'response': torch.tensor([[0.75]]),
                'survival': torch.tensor([[450.0]]),
                'uncertainty': torch.tensor([[0.2]])
            }
            node_embeddings = {}
            mock_forward.return_value = (predictions, node_embeddings)
            
            # Create mapping for original IDs
            pyg_data['treatment'].original_ids = [f"treatment_{i}" for i in range(1, 6)]
            
            # Add some nodes to the graph for lookup
            treatment_id = "treatment_1"
            tumor_id = "tumor_1"
            patient_id = "patient_1"
            
            # Add nodes to the graph
            G = treatment_explainer.original_graph
            G.add_node(treatment_id, type='treatment', category='surgery', name='Test Surgery')
            G.add_node(tumor_id, type='tumor', grade='IV')
            G.add_node(patient_id, type='patient', age=50)
            
            # Add edges
            G.add_edge(tumor_id, treatment_id, relation='treated_with')
            G.add_edge(patient_id, tumor_id, relation='has_tumor')
            
            # Generate explanation
            explanation = treatment_explainer.explain_treatment_prediction(0, k=3)
            
            # Check explanation structure
            assert 'treatment_id' in explanation
            assert 'treatment_attributes' in explanation
            assert 'patient_id' in explanation
            assert 'tumor_id' in explanation
            assert 'predictions' in explanation
            assert 'feature_importance' in explanation
            
            # Check prediction values
            assert explanation['predictions']['response_probability'] == 0.75
            assert explanation['predictions']['survival_days'] == 450.0
            assert explanation['predictions']['uncertainty'] == 0.2
            
            # Check feature importance
            assert 'response' in explanation['feature_importance']
            assert 'survival' in explanation['feature_importance']
            assert len(explanation['feature_importance']['response']) == 3  # top k=3
            assert len(explanation['feature_importance']['survival']) == 3  # top k=3
            
            # Check identifiers
            assert explanation['treatment_id'] == 'treatment_1'
            assert explanation['tumor_id'] == 'tumor_1'
            assert explanation['patient_id'] == 'patient_1'
    
    def test_get_treatment_comparison(self, treatment_explainer):
        """Test generation of treatment comparisons."""
        # Create mock explain_treatment_prediction method
        with patch.object(treatment_explainer, 'explain_treatment_prediction') as mock_explain:
            # Set up mock return values for two different treatments
            mock_explain.side_effect = [
                {
                    'treatment_id': 'treatment_1',
                    'predictions': {
                        'response_probability': 0.8,
                        'survival_days': 500,
                        'uncertainty': 0.1
                    },
                    'feature_importance': {
                        'response': [('feature_A', 0.5), ('feature_B', 0.3), ('feature_C', 0.2)],
                        'survival': [('feature_B', 0.6), ('feature_A', 0.3), ('feature_D', 0.1)]
                    }
                },
                {
                    'treatment_id': 'treatment_2',
                    'predictions': {
                        'response_probability': 0.7,
                        'survival_days': 450,
                        'uncertainty': 0.2
                    },
                    'feature_importance': {
                        'response': [('feature_B', 0.6), ('feature_A', 0.3), ('feature_E', 0.1)],
                        'survival': [('feature_B', 0.5), ('feature_D', 0.3), ('feature_A', 0.2)]
                    }
                }
            ]
            
            # Generate comparison
            comparison = treatment_explainer.get_treatment_comparison([0, 1])
            
            # Check comparison structure
            assert 'treatments' in comparison
            assert 'predictions' in comparison
            assert 'common_features' in comparison
            assert 'differentiating_features' in comparison
            
            # Check treatments
            assert len(comparison['treatments']) == 2
            assert comparison['treatments'][0]['treatment_id'] == 'treatment_1'
            assert comparison['treatments'][1]['treatment_id'] == 'treatment_2'
            
            # Check predictions
            assert len(comparison['predictions']) == 2
            assert comparison['predictions'][0]['response_probability'] == 0.8
            assert comparison['predictions'][1]['response_probability'] == 0.7
            
            # Check common features - features A and B appear in both treatments' important features
            assert 'feature_A' in comparison['common_features']
            assert 'feature_B' in comparison['common_features']
            
            # Check differentiating features - C, D, E appear in only one treatment
            assert 'feature_C' in comparison['differentiating_features']
            assert 'feature_D' in comparison['differentiating_features']
            assert 'feature_E' in comparison['differentiating_features']
    
    def test_generate_natural_language_explanation(self, treatment_explainer):
        """Test generation of natural language explanations."""
        # Create a sample explanation
        explanation = {
            'treatment_id': 'treatment_1',
            'treatment_attributes': {
                'type': 'treatment',
                'category': 'surgery',
                'name': 'Gross total resection',
                'duration': 1
            },
            'patient_id': 'patient_1',
            'tumor_id': 'tumor_1',
            'predictions': {
                'response_probability': 0.8,
                'survival_days': 500,
                'uncertainty': 0.1
            },
            'feature_importance': {
                'response': [('category_surgery', 0.5), ('duration', 0.3), ('tumor_grade', 0.2)],
                'survival': [('patient_age', 0.4), ('tumor_location', 0.3), ('duration', 0.3)]
            }
        }
        
        # Generate explanation text
        explanation_text = treatment_explainer.generate_natural_language_explanation(explanation)
        
        # Check that explanation is a non-empty string
        assert isinstance(explanation_text, str)
        assert len(explanation_text) > 0
        
        # Check that key elements are included in the text
        assert 'surgery' in explanation_text.lower()
        assert '80' in explanation_text  # 80% probability (0.8)
        assert '500' in explanation_text  # 500 days survival
        assert 'high confidence' in explanation_text.lower()  # low uncertainty (0.1)
        
        # Check that feature importance is mentioned
        assert 'key factors' in explanation_text.lower()
        assert 'category_surgery' in explanation_text
        assert 'duration' in explanation_text
        assert 'patient_age' in explanation_text
        
        # Test with different uncertainty level
        explanation['predictions']['uncertainty'] = 0.25
        explanation_text = treatment_explainer.generate_natural_language_explanation(explanation)
        assert 'moderate confidence' in explanation_text.lower()
        
        # Test with high uncertainty
        explanation['predictions']['uncertainty'] = 0.4
        explanation_text = treatment_explainer.generate_natural_language_explanation(explanation)
        assert 'low confidence' in explanation_text.lower()