# susruta/tests/test_treatment.py
"""Tests for the treatment simulation module."""

import pytest
import torch
import networkx as nx
from torch_geometric.data import HeteroData
from unittest.mock import patch, MagicMock
import numpy as np # Import numpy

# Import classes directly (fixtures are now in conftest.py)
from susruta.treatment import TreatmentSimulator
from susruta.models import GliomaGNN
from susruta.graph import GliomaKnowledgeGraph

# Fixtures like gnn_model, knowledge_graph, pyg_data, treatment_options, etc.,
# are automatically injected by pytest from conftest.py


class TestTreatmentSimulator:
    """Test suite for the TreatmentSimulator class."""

    def test_initialization(self, treatment_simulator, gnn_model, knowledge_graph):
        """Test initialization of TreatmentSimulator."""
        assert treatment_simulator.model == gnn_model
        # Check graph structure if needed, comparing nodes/edges might be complex
        assert isinstance(treatment_simulator.graph, nx.MultiDiGraph)
        assert hasattr(treatment_simulator, 'device')
        assert not treatment_simulator.model.training

    def test_encode_treatment(self, treatment_simulator, treatment_options):
        """Test encoding of treatment configurations."""
        surgery_option = treatment_options[0]
        encoded = treatment_simulator._encode_treatment(surgery_option)
        # Expected length: 4 (category) + 3 (dose, duration, intensity) = 7
        assert len(encoded) == 7, f"Expected length 7, but got {len(encoded)}"
        # surgery category: [1.0, 0.0, 0.0, 0.0]
        assert encoded[0:4] == [1.0, 0.0, 0.0, 0.0], f"Category encoding mismatch: {encoded[0:4]}"
        # numerical: dose=0.0, duration=1.0, intensity=0.9
        assert encoded[4:] == pytest.approx([0.0, 1.0, 0.9]), f"Numerical encoding mismatch: {encoded[4:]}"

        chemo_option = treatment_options[2]
        encoded_chemo = treatment_simulator._encode_treatment(chemo_option)
        assert len(encoded_chemo) == 7, f"Expected length 7, but got {len(encoded_chemo)}"
        # chemo category: [0.0, 0.0, 1.0, 0.0]
        assert encoded_chemo[0:4] == [0.0, 0.0, 1.0, 0.0], f"Category encoding mismatch: {encoded_chemo[0:4]}"
        # numerical: dose=150.0, duration=120.0, intensity=0.0
        assert encoded_chemo[4:] == pytest.approx([150.0, 120.0, 0.0]), f"Numerical encoding mismatch: {encoded_chemo[4:]}"

    def test_find_similar_treatments(self, treatment_simulator, treatment_options, pyg_data, knowledge_graph):
        """Test finding similar treatments in the graph."""
        # Ensure the graph used by the simulator is the fixture graph
        treatment_simulator.graph = knowledge_graph

        # Determine the actual number of treatments and their categories from the graph fixture
        actual_treatments = {}
        if 'treatment' in pyg_data.node_types and hasattr(pyg_data['treatment'], 'original_ids'):
            for i, tid in enumerate(pyg_data['treatment'].original_ids):
                # Ensure tid is string for graph lookup
                tid_str = str(tid) if not isinstance(tid, str) else tid
                if tid_str in knowledge_graph:
                     actual_treatments[tid_str] = knowledge_graph.nodes[tid_str].get('category')
                # Also handle potential integer keys if graph was built differently
                elif isinstance(tid, int) and tid in knowledge_graph:
                     actual_treatments[tid] = knowledge_graph.nodes[tid].get('category')


        num_surgery = sum(1 for cat in actual_treatments.values() if cat == 'surgery')
        num_radiation = sum(1 for cat in actual_treatments.values() if cat == 'radiation')

        # Create dummy embeddings matching the actual number of treatments in pyg_data
        num_actual_treatments = pyg_data['treatment'].num_nodes
        # Ensure embeddings exist even if num_actual_treatments is 0
        if num_actual_treatments > 0:
            treatment_embeds = torch.randn(num_actual_treatments, treatment_simulator.model.hidden_channels)
        else:
            treatment_embeds = torch.empty((0, treatment_simulator.model.hidden_channels))

        node_embeddings = {
            'treatment': treatment_embeds
        }


        # Test finding surgery treatments
        surgery_option = treatment_options[0] # category: 'surgery'
        similar_treatments_surgery = treatment_simulator._find_similar_treatments(surgery_option, pyg_data, node_embeddings)
        assert len(similar_treatments_surgery) == num_surgery
        if num_surgery > 0:
            assert all(t.shape == (treatment_simulator.model.hidden_channels,) for t in similar_treatments_surgery)

        # Test finding radiation treatments
        radiation_option = treatment_options[1] # category: 'radiation'
        similar_treatments_radiation = treatment_simulator._find_similar_treatments(radiation_option, pyg_data, node_embeddings)
        assert len(similar_treatments_radiation) == num_radiation
        if num_radiation > 0:
             assert all(t.shape == (treatment_simulator.model.hidden_channels,) for t in similar_treatments_radiation)


        # Test for a category with no matches in graph
        no_match_option = {'category': 'experimental'}
        no_match_similar = treatment_simulator._find_similar_treatments(no_match_option, pyg_data, node_embeddings)
        assert len(no_match_similar) == 0


    @patch.object(TreatmentSimulator, '_find_similar_treatments')
    def test_simulate_treatments(self, mock_find_similar, treatment_simulator, treatment_options, pyg_data):
        """Test simulating outcomes for different treatments."""
        # --- Setup Mock Model Behavior ---
        mock_model_instance = MagicMock()
        mock_model_instance.hidden_channels = treatment_simulator.model.hidden_channels
        mock_model_instance.device = treatment_simulator.device

        # Mock the prediction heads
        mock_model_instance.response_head = MagicMock(return_value=torch.tensor([[0.8]], device=treatment_simulator.device))
        mock_model_instance.survival_head = MagicMock(return_value=torch.tensor([[500.0]], device=treatment_simulator.device))
        mock_model_instance.uncertainty_head = MagicMock(return_value=torch.tensor([[0.1]], device=treatment_simulator.device))

        # Mock the node encoder for 'treatment'
        mock_treatment_embedding = torch.randn(1, mock_model_instance.hidden_channels, device=treatment_simulator.device) # Ensure batch dim
        mock_encoder = MagicMock(return_value=mock_treatment_embedding)
        mock_encoder.in_features = 7 # Match the expected feature length from _encode_treatment
        mock_model_instance.node_encoders = {'treatment': mock_encoder} # Assign directly

        # Configure forward directly on the mock instance
        mock_embeddings = {
            'patient': torch.randn(pyg_data['patient'].num_nodes, mock_model_instance.hidden_channels, device=treatment_simulator.device),
            'tumor': torch.randn(pyg_data['tumor'].num_nodes, mock_model_instance.hidden_channels, device=treatment_simulator.device),
            'treatment': torch.randn(pyg_data['treatment'].num_nodes, mock_model_instance.hidden_channels, device=treatment_simulator.device)
            # Add other node types if necessary based on pyg_data
        }
        # Make the mock instance's forward method return the desired tuple
        mock_model_instance.return_value = (None, mock_embeddings)

        # Configure the mock _find_similar_treatments
        # Return an empty list to force fallback to encoder
        mock_find_similar.return_value = []

        # --- Replace the simulator's model with the configured mock ---
        original_model = treatment_simulator.model
        treatment_simulator.model = mock_model_instance
        treatment_simulator.model.eval = MagicMock()

        # --- Run the simulation ---
        # Use patient/tumor IDs known to be in the fixture (e.g., patient 3)
        patient_id = "patient_3"
        tumor_id = "tumor_3"

        # Ensure patient/tumor IDs exist in the data for indexing
        if not hasattr(pyg_data['patient'], 'original_ids') or patient_id not in pyg_data['patient'].original_ids:
             pytest.skip(f"Patient {patient_id} not found in pyg_data fixture for simulation test.")
        if not hasattr(pyg_data['tumor'], 'original_ids') or tumor_id not in pyg_data['tumor'].original_ids:
             pytest.skip(f"Tumor {tumor_id} not found in pyg_data fixture for simulation test.")


        results = treatment_simulator.simulate_treatments(patient_id, tumor_id, treatment_options, pyg_data)

        # --- Assertions ---
        assert len(results) == len(treatment_options)
        assert 'option_0' in results
        assert results['option_0']['response_prob'] == pytest.approx(0.8)
        assert results['option_0']['survival_days'] == pytest.approx(500.0)
        assert results['option_0']['uncertainty'] == pytest.approx(0.1)

        # Check mocks were called
        mock_model_instance.assert_called_once()
        assert mock_find_similar.call_count == len(treatment_options)
        # Check calls to the heads on the *mocked model instance*
        assert mock_model_instance.response_head.call_count == len(treatment_options)
        assert mock_model_instance.survival_head.call_count == len(treatment_options)
        assert mock_model_instance.uncertainty_head.call_count == len(treatment_options)
        # Check encoder was called (since find_similar returned empty)
        assert mock_model_instance.node_encoders['treatment'].call_count == len(treatment_options)


        # Restore original model if necessary
        treatment_simulator.model = original_model

    def test_rank_treatments(self, treatment_simulator):
        """Test ranking treatments based on simulated outcomes."""
        simulated_outcomes = {
            'option_0': {'response_prob': 0.8, 'survival_days': 500, 'uncertainty': 0.1},
            'option_1': {'response_prob': 0.7, 'survival_days': 450, 'uncertainty': 0.2},
            'option_2': {'response_prob': 0.6, 'survival_days': 400, 'uncertainty': 0.3}
        }
        ranked = treatment_simulator.rank_treatments(simulated_outcomes)
        assert ranked == ['option_0', 'option_1', 'option_2']

        conflicting_outcomes = {
            'option_A': {'response_prob': 0.9, 'survival_days': 400, 'uncertainty': 0.1},
            'option_B': {'response_prob': 0.6, 'survival_days': 600, 'uncertainty': 0.2}
        }
        # Default weights: {'response_prob': 0.4, 'survival_days': 0.4, 'uncertainty': -0.2}
        # Normalize A: resp=1.0, surv=0.0, unc=0.0 -> Score = 0.4*1.0 + 0.4*0.0 - 0.2*0.0 = 0.4
        # Normalize B: resp=0.0, surv=1.0, unc=1.0 -> Score = 0.4*0.0 + 0.4*1.0 - 0.2*1.0 = 0.2
        ranked_default_conflict = treatment_simulator.rank_treatments(conflicting_outcomes)
        assert ranked_default_conflict == ['option_A', 'option_B']

        # Survival heavy weights: {'response_prob': 0.1, 'survival_days': 0.8, 'uncertainty': -0.1}
        # Normalize A: resp=1.0, surv=0.0, unc=0.0 -> Score = 0.1*1.0 + 0.8*0.0 - 0.1*0.0 = 0.1
        # Normalize B: resp=0.0, surv=1.0, unc=1.0 -> Score = 0.1*0.0 + 0.8*1.0 - 0.1*1.0 = 0.7
        ranked_survival_heavy = treatment_simulator.rank_treatments(
            conflicting_outcomes,
            weights={'response_prob': 0.1, 'survival_days': 0.8, 'uncertainty': -0.1}
        )
        assert ranked_survival_heavy == ['option_B', 'option_A']
