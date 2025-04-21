"""Test fixtures for SUSRUTA test suite."""

import os
import pytest
import numpy as np
import pandas as pd
import torch
import networkx as nx

from susruta.data import ClinicalDataProcessor, EfficientMRIProcessor
from susruta.graph import GliomaKnowledgeGraph
from susruta.models import GliomaGNN
from susruta.treatment import TreatmentSimulator
from susruta.viz import ExplainableGliomaTreatment
from susruta.utils import MemoryTracker


@pytest.fixture(scope="session")
def synthetic_clinical_data():
    """Create synthetic clinical data for testing."""
    np.random.seed(42)  # For reproducibility
    
    data = pd.DataFrame({
        'patient_id': range(1, 11),
        'age': np.random.randint(30, 80, 10),
        'sex': np.random.choice(['M', 'F'], 10),
        'karnofsky_score': np.random.randint(60, 100, 10),
        'grade': np.random.choice(['II', 'III', 'IV'], 10),
        'histology': np.random.choice(['Astrocytoma', 'Oligodendroglioma', 'GBM'], 10),
        'location': np.random.choice(['Frontal', 'Temporal', 'Parietal'], 10),
        'idh_mutation': np.random.choice([0, 1], 10),
        'mgmt_methylation': np.random.choice([0, 1], 10)
    })
    
    return data


@pytest.fixture(scope="session")
def synthetic_treatment_data():
    """Create synthetic treatment data for testing."""
    np.random.seed(43)  # Different seed from clinical data
    
    treatments = []
    for patient_id in range(1, 11):
        # Each patient gets 1-3 treatments
        num_treatments = np.random.randint(1, 4)
        for i in range(num_treatments):
            treatment_id = len(treatments) + 1
            category = np.random.choice(['surgery', 'radiation', 'chemotherapy'])
            
            # Add treatment specifics based on category
            if category == 'surgery':
                treatment_name = np.random.choice(['Gross total resection', 'Subtotal resection'])
                dose = None
            elif category == 'radiation':
                treatment_name = 'External beam radiation'
                dose = np.random.choice([45.0, 54.0, 60.0])  # Gy
            else:  # chemotherapy
                treatment_name = np.random.choice(['Temozolomide', 'PCV', 'Bevacizumab'])
                dose = np.random.randint(100, 200)  # mg/m²
            
            duration_days = np.random.randint(1, 180)
            start_day = np.random.randint(0, 100)
            
            # Add outcome
            response = np.random.choice([0, 1], p=[0.3, 0.7])  # Binary outcome
            progression_free_days = np.random.randint(30, 1000)
            survival_days = progression_free_days + np.random.randint(0, 500)
            
            treatments.append({
                'patient_id': patient_id,
                'treatment_id': treatment_id,
                'category': category,
                'treatment_name': treatment_name,
                'dose': dose,
                'duration_days': duration_days,
                'start_day': start_day,
                'response': response,
                'progression_free_days': progression_free_days,
                'survival_days': survival_days
            })
    
    return pd.DataFrame(treatments)


@pytest.fixture(scope="session")
def synthetic_imaging_features():
    """Create synthetic imaging features for testing."""
    np.random.seed(44)  # Different seed
    
    imaging_features = {}
    for patient_id in range(1, 11):
        patient_features = {}
        
        # T1c features
        patient_features['t1c'] = {
            'mean': np.random.uniform(100, 200),
            'std': np.random.uniform(10, 50),
            'max': np.random.uniform(200, 300),
            'volume_voxels': np.random.randint(1000, 5000)
        }
        
        # T2 features
        patient_features['t2w'] = {
            'mean': np.random.uniform(150, 250),
            'std': np.random.uniform(20, 60),
            'max': np.random.uniform(250, 350),
            'volume_voxels': np.random.randint(1000, 5000)
        }
        
        # Tumor features
        patient_features['tumor'] = {
            'volume_mm3': np.random.uniform(1000, 30000),
            'surface_area': np.random.uniform(500, 5000),
            'elongation': np.random.uniform(0.2, 0.8),
            'roundness': np.random.uniform(0.3, 0.9)
        }
        
        imaging_features[patient_id] = patient_features
    
    return imaging_features


@pytest.fixture(scope="session")
def clinical_processor():
    """Create clinical data processor for testing."""
    return ClinicalDataProcessor()


@pytest.fixture(scope="session")
def mri_processor():
    """Create MRI processor for testing."""
    return EfficientMRIProcessor(memory_limit_mb=1000)


@pytest.fixture(scope="session")
def processed_clinical_data(synthetic_clinical_data, clinical_processor):
    """Create processed clinical data for testing."""
    return clinical_processor.preprocess_clinical_data(synthetic_clinical_data)


@pytest.fixture(scope="session")
def processed_treatment_data(synthetic_treatment_data, clinical_processor):
    """Create processed treatment data for testing."""
    return clinical_processor.process_treatment_data(synthetic_treatment_data)


@pytest.fixture(scope="session")
def knowledge_graph(synthetic_clinical_data, synthetic_treatment_data, synthetic_imaging_features):
    """Create knowledge graph for testing."""
    kg_builder = GliomaKnowledgeGraph(memory_limit_mb=2000)
    kg_builder.add_clinical_data(synthetic_clinical_data)
    kg_builder.add_imaging_features(synthetic_imaging_features)
    kg_builder.add_treatments(synthetic_treatment_data)
    kg_builder.add_similarity_edges(threshold=0.5)
    
    return kg_builder.G


@pytest.fixture(scope="session")
def pyg_data(knowledge_graph, synthetic_clinical_data, synthetic_treatment_data, synthetic_imaging_features):
    """Create PyTorch Geometric data for testing."""
    kg_builder = GliomaKnowledgeGraph(memory_limit_mb=2000)
    kg_builder.G = knowledge_graph  # Use the existing graph
    
    return kg_builder.to_pytorch_geometric()


@pytest.fixture(scope="session")
def gnn_model(pyg_data):
    """Create GNN model for testing."""
    node_feature_dims = {node_type: data.x.size(1) for node_type, data in pyg_data.items()}
    
    # Initialize edge_feature_dims
    edge_feature_dims = {}
    for edge_type in pyg_data.edge_types:
        edge_feature_dims[edge_type] = 1  # Default edge feature dimension
    
    # Create model
    torch.manual_seed(42)  # For reproducibility
    model = GliomaGNN(
        node_feature_dims=node_feature_dims,
        edge_feature_dims=edge_feature_dims,
        hidden_channels=16,  # Smaller for testing
        dropout=0.2
    )
    
    return model


@pytest.fixture(scope="session")
def treatment_simulator(gnn_model, knowledge_graph):
    """Create treatment simulator for testing."""
    return TreatmentSimulator(gnn_model, knowledge_graph)


@pytest.fixture(scope="session")
def treatment_explainer(gnn_model, pyg_data, knowledge_graph):
    """Create treatment explainer for testing."""
    return ExplainableGliomaTreatment(gnn_model, pyg_data, knowledge_graph)


@pytest.fixture(scope="session")
def treatment_options():
    """Create treatment options for testing."""
    return [
        {
            'category': 'surgery',
            'name': 'Gross total resection',
            'intensity': 0.9,
            'duration': 1
        },
        {
            'category': 'radiation',
            'name': 'Standard radiation therapy',
            'dose': 60.0,
            'duration': 30
        },
        {
            'category': 'chemotherapy',
            'name': 'Temozolomide',
            'dose': 150,
            'duration': 120
        }
    ]