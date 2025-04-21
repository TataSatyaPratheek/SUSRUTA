"""Example script for constructing and analyzing the glioma knowledge graph."""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from susruta.data import ClinicalDataProcessor
from susruta.graph import GliomaKnowledgeGraph
from susruta.utils import MemoryTracker


def create_synthetic_data():
    """Create synthetic data for demonstration."""
    # Create synthetic clinical data
    clinical_data = pd.DataFrame({
        'patient_id': range(1, 31),
        'age': np.random.randint(30, 80, 30),
        'sex': np.random.choice(['M', 'F'], 30),
        'karnofsky_score': np.random.randint(60, 100, 30),
        'grade': np.random.choice(['II', 'III', 'IV'], 30),
        'histology': np.random.choice(['Astrocytoma', 'Oligodendroglioma', 'GBM'], 30),
        'location': np.random.choice(['Frontal', 'Temporal', 'Parietal'], 30),
        'idh_mutation': np.random.choice([0, 1], 30),
        'mgmt_methylation': np.random.choice([0, 1], 30)
    })
    
    # Create synthetic treatment data
    treatments = []
    for patient_id in range(1, 31):
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
            response = np.random.choice(['CR', 'PR', 'SD', 'PD'])
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
    
    treatments_df = pd.DataFrame(treatments)
    
    # Create synthetic imaging features
    imaging_features = {}
    for patient_id in range(1, 31):
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
    
    return clinical_data, treatments_df, imaging_features


def main():
    """Demonstrate knowledge graph construction and analysis."""
    tracker = MemoryTracker()
    tracker.log_memory("Initial")
    
    print("=== Glioma Knowledge Graph Construction Example ===")
    
    # Create synthetic data
    print("\nCreating synthetic data...")
    clinical_data, treatments_df, imaging_features = create_synthetic_data()
    tracker.log_memory("After data creation")
    
    # Process clinical data
    print("\nProcessing clinical data...")
    clinical_processor = ClinicalDataProcessor()
    processed_clinical = clinical_processor.preprocess_clinical_data(clinical_data)
    tracker.log_memory("After clinical data processing")
    
    # Process treatment data
    print("\nProcessing treatment data...")
    processed_treatments = clinical_processor.process_treatment_data(treatments_df)
    tracker.log_memory("After treatment data processing")
    
    # Construct knowledge graph
    print("\nConstructing knowledge graph...")
    kg_builder = GliomaKnowledgeGraph(memory_limit_mb=3000)
    
    # Add clinical data to graph
    print("Adding clinical data to graph...")
    kg_builder.add_clinical_data(clinical_data, batch_size=10)
    tracker.log_memory("After adding clinical data")
    
    # Add imaging features to graph
    print("Adding imaging features to graph...")
    kg_builder.add_imaging_features(imaging_features, max_features_per_node=15)
    tracker.log_memory("After adding imaging features")
    
    # Add treatments to graph
    print("Adding treatments to graph...")
    kg_builder.add_treatments(treatments_df, batch_size=10)
    tracker.log_memory("After adding treatments")
    
    # Add similarity edges
    print("Adding similarity edges...")
    kg_builder.add_similarity_edges(threshold=0.6, max_edges_per_node=3)
    tracker.log_memory("After adding similarity edges")
    
    # Get graph statistics
    print("\nGraph Statistics:")
    stats = kg_builder.get_statistics()
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for subkey, subvalue in value.items():
                print(f"    {subkey}: {subvalue}")
        else:
            print(f"  {key}: {value}")
    
    # Convert to PyTorch Geometric format
    print("\nConverting to PyTorch Geometric format...")
    pyg_data = kg_builder.to_pytorch_geometric()
    tracker.log_memory("After PyG conversion")
    
    print("\nPyTorch Geometric Data Statistics:")
    for node_type in pyg_data.node_types:
        print(f"  {node_type} nodes: {pyg_data[node_type].x.shape[0]}")
    
    for edge_type in pyg_data.edge_types:
        print(f"  {edge_type} edges: {pyg_data[edge_type].edge_index.shape[1]}")
    
    # Plot memory usage
    print("\nPlotting memory usage...")
    fig = tracker.plot_memory_usage()
    fig.savefig("graph_construction_memory.png")
    print("Memory usage plot saved as 'graph_construction_memory.png'")
    
    print("\nKnowledge graph construction example completed successfully!")


if __name__ == "__main__":
    main()