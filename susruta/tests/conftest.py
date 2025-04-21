# susruta/tests/conftest.py
"""Test fixtures for SUSRUTA test suite."""

import os
import pytest
import numpy as np
import pandas as pd
import torch
import networkx as nx
from torch_geometric.data import HeteroData

from susruta.data import ClinicalDataProcessor, EfficientMRIProcessor
from susruta.graph import GliomaKnowledgeGraph
from susruta.models import GliomaGNN
from susruta.treatment import TreatmentSimulator
from susruta.viz import ExplainableGliomaTreatment
from susruta.utils import MemoryTracker

# --- Data Fixtures ---

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
    # Introduce some NaNs for imputation testing
    data.loc[0, 'age'] = np.nan
    data.loc[1, 'sex'] = np.nan
    data.loc[2, 'karnofsky_score'] = np.nan
    return data

@pytest.fixture(scope="session")
def synthetic_treatment_data():
    """Create synthetic treatment data for testing."""
    np.random.seed(43)  # Different seed
    treatments = []
    for patient_id in range(1, 11):
        num_treatments = np.random.randint(1, 4) # 1 to 3 treatments per patient
        for i in range(num_treatments):
            treatment_id = len(treatments) + 1
            category = np.random.choice(['surgery', 'radiation', 'chemotherapy'])
            dose = None
            duration_days = np.random.randint(1, 180)
            if category == 'radiation':
                dose = np.random.choice([45.0, 54.0, 60.0])
            elif category == 'chemotherapy':
                dose = np.random.randint(100, 200)

            response = np.random.choice([0, 1], p=[0.3, 0.7]) # Binary response
            survival_days = np.random.randint(30, 1500)
            treatments.append({
                'patient_id': patient_id,
                'treatment_id': treatment_id, # Use simple integer ID here
                'category': category,
                'dose': dose,
                'duration_days': duration_days,
                'response': response,
                'survival_days': survival_days,
                'start_day': np.random.randint(0, 100),
                'treatment_name': f"{category}_{treatment_id}"
            })
    return pd.DataFrame(treatments)

@pytest.fixture(scope="session")
def synthetic_imaging_features():
    """Create synthetic imaging features (dict format) for testing."""
    np.random.seed(44)
    imaging_features = {}
    for patient_id in range(1, 11):
        imaging_features[patient_id] = {
            't1c': {'mean': np.random.uniform(100, 200), 'std': np.random.uniform(10, 50)},
            't2w': {'mean': np.random.uniform(150, 250), 'std': np.random.uniform(20, 60)},
            'tumor': {'volume_mm3': np.random.uniform(1000, 30000), 'elongation': np.random.uniform(0.2, 0.8)}
        }
    return imaging_features

# --- Processor Fixtures ---

@pytest.fixture(scope="session")
def clinical_processor():
    """Create clinical data processor instance."""
    return ClinicalDataProcessor()

@pytest.fixture(scope="session")
def mri_processor():
    """Create MRI processor instance."""
    return EfficientMRIProcessor(memory_limit_mb=1000)

# --- Processed Data Fixtures ---

@pytest.fixture(scope="session")
def processed_clinical_data(synthetic_clinical_data, clinical_processor):
    """Create processed clinical data."""
    return clinical_processor.preprocess_clinical_data(synthetic_clinical_data)

@pytest.fixture(scope="session")
def processed_treatment_data(synthetic_treatment_data, clinical_processor):
    """Create processed treatment data."""
    return clinical_processor.process_treatment_data(synthetic_treatment_data)

# --- Graph Fixtures ---

@pytest.fixture(scope="session")
def knowledge_graph(synthetic_clinical_data, synthetic_treatment_data, synthetic_imaging_features):
    """Create a NetworkX knowledge graph."""
    kg_builder = GliomaKnowledgeGraph(memory_limit_mb=2000)
    # Add data - ensure order allows connections to be made
    kg_builder.add_clinical_data(synthetic_clinical_data)
    kg_builder.add_imaging_features(synthetic_imaging_features)
    kg_builder.add_treatments(synthetic_treatment_data) # Adds treatments and outcomes
    kg_builder.add_similarity_edges(threshold=0.5)
    return kg_builder.G

@pytest.fixture(scope="session")
def pyg_data(knowledge_graph): # Build pyg_data from the final knowledge_graph fixture
    """Create PyTorch Geometric HeteroData from the NetworkX graph."""
    kg_builder = GliomaKnowledgeGraph(memory_limit_mb=2000)
    kg_builder.G = knowledge_graph # Use the fully constructed graph
    data = kg_builder.to_pytorch_geometric()

    # --- Post-processing/Validation for PyG data ---
    required_node_types = ['patient', 'tumor', 'treatment', 'feature', 'outcome']
    for node_type in required_node_types:
        if node_type not in data.node_types:
            print(f"Warning: Node type '{node_type}' missing in pyg_data fixture. Adding empty.")
            data[node_type].x = torch.empty((0, 1), dtype=torch.float) # Shape (0, 1)
            data[node_type].original_ids = []
        elif not hasattr(data[node_type], 'x') or data[node_type].x is None:
             print(f"Warning: Node type '{node_type}' exists but has no 'x'. Adding empty tensor.")
             num_nodes = data[node_type].num_nodes if hasattr(data[node_type], 'num_nodes') else 0
             data[node_type].x = torch.empty((num_nodes, 1), dtype=torch.float) # Shape (N, 1)
        elif data[node_type].num_nodes > 0 and data[node_type].num_node_features == 0:
             # If nodes exist but features are 0-dim, add a dummy feature
             print(f"Warning: Node type '{node_type}' has 0 features. Adding dummy feature.")
             data[node_type].x = torch.zeros((data[node_type].num_nodes, 1), dtype=torch.float) # Shape (N, 1)

        # Ensure original_ids exist
        if node_type in data.node_types and not hasattr(data[node_type], 'original_ids'):
             print(f"Warning: Node type '{node_type}' missing 'original_ids'. Adding dummy list.")
             num_nodes = data[node_type].num_nodes if hasattr(data[node_type], 'num_nodes') else 0
             data[node_type].original_ids = [f"{node_type}_{i}" for i in range(num_nodes)]


    # Validate edge types
    for edge_type in data.edge_types:
         if not hasattr(data[edge_type], 'edge_index') or data[edge_type].edge_index is None:
              print(f"Warning: Edge type '{edge_type}' missing 'edge_index'.")
         # Add check for edge_attr if needed

    return data

# --- Model and Utility Fixtures ---

@pytest.fixture(scope="session")
def gnn_model(pyg_data):
    """Create GNN model instance based on pyg_data dimensions."""
    node_feature_dims = {}
    default_dim = 1 # Use 1 as default dim for nodes/edges without features

    for node_type in pyg_data.node_types:
        if hasattr(pyg_data[node_type], 'x') and pyg_data[node_type].x is not None:
            # Use actual dim if features exist, ensure it's at least default_dim
            dim = pyg_data[node_type].x.size(1)
            node_feature_dims[node_type] = max(dim, default_dim)
        else:
            # Handle cases where a node type might have no features tensor or no nodes
            node_feature_dims[node_type] = default_dim
            print(f"Warning: Using default dimension ({default_dim}) for node type '{node_type}' in gnn_model fixture.")

    edge_feature_dims = {}
    for edge_type in pyg_data.edge_types:
        # --- Start Fix: Use pyg_data instead of data ---
        if hasattr(pyg_data[edge_type], 'edge_attr') and pyg_data[edge_type].edge_attr is not None and pyg_data[edge_type].num_edges > 0:
             dim = pyg_data[edge_type].edge_attr.size(1)
             edge_feature_dims[edge_type] = max(dim, default_dim)
        else:
            edge_feature_dims[edge_type] = default_dim # Default if no edge attributes or no edges
        # --- End Fix ---

    torch.manual_seed(42)
    model = GliomaGNN(
        node_feature_dims=node_feature_dims,
        edge_feature_dims=edge_feature_dims,
        hidden_channels=16, # Smaller for testing
        dropout=0.2
    )
    return model

@pytest.fixture(scope="function") # Use function scope if simulator state matters per test
def treatment_simulator(gnn_model, knowledge_graph):
    """Create treatment simulator instance."""
    # Pass a copy of the graph if modifications are made during tests
    return TreatmentSimulator(gnn_model, knowledge_graph.copy())

@pytest.fixture(scope="function") # Use function scope if explainer state matters per test
def treatment_explainer(gnn_model, pyg_data, knowledge_graph):
    """Create treatment explainer instance."""
    # Pass copies if state needs isolation
    cloned_data = pyg_data.clone()

    # Ensure the cloned data has the necessary structure, especially for 'treatment'
    if 'treatment' not in cloned_data.node_types:
         print("Warning: Adding empty 'treatment' node type to data for explainer.")
         cloned_data['treatment'].x = torch.empty((0, max(gnn_model.node_feature_dims.get('treatment', 1), 1)))
         cloned_data['treatment'].original_ids = []
    elif not hasattr(cloned_data['treatment'], 'x') or cloned_data['treatment'].x is None:
         print("Warning: Adding empty 'x' tensor to 'treatment' node type for explainer.")
         num_nodes = cloned_data['treatment'].num_nodes if hasattr(cloned_data['treatment'], 'num_nodes') else 0
         cloned_data['treatment'].x = torch.empty((num_nodes, max(gnn_model.node_feature_dims.get('treatment', 1), 1)))
    elif cloned_data['treatment'].num_nodes > 0 and cloned_data['treatment'].num_node_features == 0:
         # Add dummy features if missing, matching model expectation (at least dim 1)
         feature_dim = max(gnn_model.node_feature_dims.get('treatment', 1), 1)
         print(f"Warning: Adding dummy features (dim {feature_dim}) to 'treatment' node for explainer.")
         cloned_data['treatment'].x = torch.randn(cloned_data['treatment'].num_nodes, feature_dim)

    if 'treatment' in cloned_data.node_types and not hasattr(cloned_data['treatment'], 'original_ids'):
         print("Warning: Adding dummy 'original_ids' to 'treatment' node type for explainer.")
         num_nodes = cloned_data['treatment'].num_nodes if hasattr(cloned_data['treatment'], 'num_nodes') else 0
         cloned_data['treatment'].original_ids = [f"treatment_{i}" for i in range(num_nodes)]


    return ExplainableGliomaTreatment(gnn_model, cloned_data, knowledge_graph.copy())


@pytest.fixture(scope="session")
def treatment_options():
    """Create sample treatment options."""
    # Ensure these match the encoding logic in TreatmentSimulator._encode_treatment
    return [
        {'category': 'surgery', 'name': 'Gross total resection', 'intensity': 0.9, 'duration': 1, 'dose': 0.0, 'description': 'Complete removal'},
        {'category': 'radiation', 'name': 'Standard radiation therapy', 'dose': 60.0, 'duration': 30, 'intensity': 0.0, 'description': 'Standard RT'},
        {'category': 'chemotherapy', 'name': 'Temozolomide', 'dose': 150, 'duration': 120, 'intensity': 0.0, 'description': 'TMZ chemo'}
    ]
