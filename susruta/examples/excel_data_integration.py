"""Example script for integrating Excel data with MRI analysis."""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from susruta.data import EfficientMRIProcessor, ClinicalDataProcessor
from susruta.data import ExcelDataLoader, MultimodalDataIntegrator
from susruta.graph import GliomaKnowledgeGraph
from susruta.utils import MemoryTracker


def create_synthetic_imaging_features():
    """Create synthetic imaging features for demonstration."""
    imaging_features = {}
    for patient_id in range(1, 11):
        imaging_features[patient_id] = {
            't1c': {
                'mean': np.random.uniform(100, 200),
                'std': np.random.uniform(10, 50),
                'max': np.random.uniform(200, 300),
                'volume_voxels': np.random.randint(1000, 5000)
            },
            't2w': {
                'mean': np.random.uniform(150, 250),
                'std': np.random.uniform(20, 60),
                'max': np.random.uniform(250, 350),
                'volume_voxels': np.random.randint(1000, 5000)
            },
            'tumor': {
                'volume_mm3': np.random.uniform(1000, 30000),
                'surface_area': np.random.uniform(500, 5000),
                'elongation': np.random.uniform(0.2, 0.8),
                'roundness': np.random.uniform(0.3, 0.9)
            }
        }
    return imaging_features


def create_synthetic_excel_files(output_dir):
    """Create synthetic Excel files for demonstration."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create scanner data
    scanner_data = pd.DataFrame({
        'PatientID': range(1, 11),
        'Timepoint': [1] * 10,
        'ScannerManufacturer': np.random.choice(['GE', 'Siemens', 'Philips'], 10),
        'ScannerModel': [f"Model{i}" for i in range(1, 11)],
        'FieldStrength': np.random.choice([1.5, 3.0], 10),
        'SequenceType': np.random.choice(['T1', 'T2', 'FLAIR'], 10)
    })
    scanner_data.to_excel(os.path.join(output_dir, 'MR_Scanner_data.xlsx'), index=False)
    
    # Create clinical data
    clinical_data = pd.DataFrame({
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
    clinical_data.to_excel(os.path.join(output_dir, 'MUGliomaPost_ClinicalDataFINAL032025.xlsx'), index=False)
    
    # Create segmentation data
    segmentation_data = pd.DataFrame({
        'PatientID': range(1, 11),
        'Timepoint': [1] * 10,
        'TumorVolume_mm3': np.random.uniform(5000, 30000, 10),
        'EnhancingVolume_mm3': np.random.uniform(1000, 10000, 10),
        'NecrotisCoreVolume_mm3': np.random.uniform(500, 5000, 10),
        'EdemaVolume_mm3': np.random.uniform(10000, 50000, 10)
    })
    segmentation_data.to_excel(os.path.join(output_dir, 'MUGliomaPost_Segmentation_Volumes.xlsx'), index=False)
    
    return {
        'scanner_path': os.path.join(output_dir, 'MR_Scanner_data.xlsx'),
        'clinical_path': os.path.join(output_dir, 'MUGliomaPost_ClinicalDataFINAL032025.xlsx'),
        'segmentation_path': os.path.join(output_dir, 'MUGliomaPost_Segmentation_Volumes.xlsx')
    }


def main():
    """Run the Excel data integration example."""
    # Initialize memory tracker
    tracker = MemoryTracker()
    tracker.log_memory("Initial")
    
    print("\n=== Excel Data Integration Example ===\n")
    
    # Set up data directory
    data_dir = Path("./example_data")
    excel_files = create_synthetic_excel_files(data_dir)
    
    print(f"Created synthetic Excel files in {data_dir}\n")
    
    # Create synthetic imaging features (in lieu of actual MRI processing)
    imaging_features = create_synthetic_imaging_features()
    
    # Initialize integrator
    integrator = MultimodalDataIntegrator(memory_limit_mb=4000)
    tracker.log_memory("After initializing integrator")
    
    # Step 1: Load Excel data
    print("Step 1: Loading Excel data sources...")
    excel_data = integrator.load_all_excel_data(
        scanner_path=excel_files['scanner_path'],
        clinical_path=excel_files['clinical_path'],
        segmentation_path=excel_files['segmentation_path'],
        timepoint=1
    )
    
    print(f"  - Loaded scanner data: {len(excel_data['scanner'])} patients")
    print(f"  - Loaded clinical data: {len(excel_data['clinical'])} patients")
    print(f"  - Loaded segmentation data: {len(excel_data['segmentation'])} patients")
    tracker.log_memory("After loading Excel data")
    
    # Step 2: Load and process basic clinical data
    print("\nStep 2: Processing basic clinical data...")
    clinical_processor = ClinicalDataProcessor()
    basic_clinical = excel_data['clinical'].copy()
    processed_clinical = clinical_processor.preprocess_clinical_data(basic_clinical)
    print(f"  - Processed clinical data shape: {processed_clinical.shape}")
    tracker.log_memory("After processing clinical data")
    
    # Step 3: Integrate all data sources
    print("\nStep 3: Integrating all data sources...")
    integrated_data = integrator.integrate_with_clinical_data(
        processed_clinical, excel_data, timepoint=1
    )
    print(f"  - Integrated data shape: {integrated_data.shape}")
    print(f"  - Feature columns: {len(integrated_data.columns)}")
    
    # Print sample columns from each data source
    clinical_cols = [col for col in integrated_data.columns if not col.startswith(('scanner_', 'seg_'))]
    scanner_cols = [col for col in integrated_data.columns if col.startswith('scanner_')]
    seg_cols = [col for col in integrated_data.columns if col.startswith('seg_')]
    
    print(f"  - Sample clinical columns: {clinical_cols[:5]}")
    print(f"  - Sample scanner columns: {scanner_cols[:5]}")
    print(f"  - Sample segmentation columns: {seg_cols[:5]}")
    
    derived_cols = ['risk_score', 'risk_category', 'age_group']
    print(f"  - Derived feature columns: {[col for col in derived_cols if col in integrated_data.columns]}")
    tracker.log_memory("After integration")
    
    # Step 4: Prepare data for graph construction
    print("\nStep 4: Preparing data for graph construction...")
    graph_data, enhanced_imaging = integrator.prepare_for_graph_construction(
        integrated_data, imaging_features
    )
    print(f"  - Graph ready data shape: {graph_data.shape}")
    print(f"  - Enhanced imaging features for {len(enhanced_imaging)} patients")
    print(f"  - Segmentation features added to imaging data")
    
    # Print sample of enhanced imaging
    patient_id = 1
    print(f"\nSample enhanced imaging for patient {patient_id}:")
    for seq_name, seq_feats in enhanced_imaging[patient_id].items():
        print(f"  - {seq_name}: {len(seq_feats)} features")
        if seq_name == 'segmentation':
            print(f"    - Features: {list(seq_feats.keys())}")
    tracker.log_memory("After graph preparation")
    
    # Step 5: Generate patient summaries
    print("\nStep 5: Generating comprehensive patient summaries...")
    summaries = integrator.generate_patient_summaries(integrated_data)
    print(f"  - Generated summaries for {len(summaries)} patients")
    
    # Print sample summary
    print(f"\nSample summary for patient {patient_id}:")
    for section, content in summaries[patient_id].items():
        print(f"  - {section}: {len(content) if isinstance(content, dict) else content} fields")
        if isinstance(content, dict):
            for key, value in list(content.items())[:3]:  # Show first 3 items
                print(f"    - {key}: {value}")
            if len(content) > 3:
                print(f"    - ... and {len(content) - 3} more fields")
    tracker.log_memory("After summary generation")
    
    # Step 6: Construct knowledge graph
    print("\nStep 6: Constructing knowledge graph...")
    kg_builder = GliomaKnowledgeGraph(memory_limit_mb=3000)
    
    # Add all data to graph
    kg_builder.add_clinical_data(integrated_data)
    kg_builder.add_imaging_features(enhanced_imaging)
    kg_builder.add_similarity_edges(threshold=0.6, max_edges_per_node=3)
    
    # Get graph statistics
    stats = kg_builder.get_statistics()
    print(f"  - Graph nodes: {stats['total_nodes']}")
    print(f"  - Graph edges: {stats['total_edges']}")
    print(f"  - Node types: {stats['node_types']}")
    print(f"  - Edge types: {stats['edge_types']}")
    tracker.log_memory("After graph construction")
    
    # Convert to PyTorch Geometric format
    print("\nStep 7: Converting to PyTorch Geometric format...")
    pyg_data = kg_builder.to_pytorch_geometric()
    print("  - PyTorch Geometric conversion successful")
    
    print("\nNode types in PyG data:")
    for node_type in pyg_data.node_types:
        print(f"  - {node_type}: {pyg_data[node_type].num_nodes} nodes with {pyg_data[node_type].num_node_features} features each")
    
    print("\nEdge types in PyG data:")
    for edge_type in pyg_data.edge_types:
        print(f"  - {edge_type}: {pyg_data[edge_type].num_edges} edges")
    tracker.log_memory("After PyG conversion")
    
    # Plot risk scores vs. segmentation volumes
    print("\nStep 8: Generating visualization of risk scores vs. tumor volumes...")
    plt.figure(figsize=(10, 6))
    
    if 'risk_score' in integrated_data.columns and 'seg_tumorvolume_mm3' in integrated_data.columns:
        risk_scores = integrated_data['risk_score']
        tumor_volumes = integrated_data['seg_tumorvolume_mm3'] / 1000  # Convert to cm³
        
        # Color by grade
        if 'grade' in integrated_data.columns:
            grades = integrated_data['grade']
            grade_colors = {'II': 'green', 'III': 'orange', 'IV': 'red'}
            colors = [grade_colors.get(g, 'blue') for g in grades]
        else:
            colors = 'blue'
        
        plt.scatter(tumor_volumes, risk_scores, c=colors, alpha=0.7, s=100)
        
        if 'idh_mutation' in integrated_data.columns:
            # Mark IDH mutated cases with a different marker
            idh_mutated = integrated_data['idh_mutation'] == 1
            plt.scatter(tumor_volumes[idh_mutated], risk_scores[idh_mutated], 
                      marker='*', s=200, facecolors='none', edgecolors='purple', 
                      linewidth=2, label='IDH mutated')
        
        plt.xlabel('Tumor Volume (cm³)')
        plt.ylabel('Risk Score')
        plt.title('Risk Score vs. Tumor Volume by Grade and IDH Status')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        plot_path = os.path.join(data_dir, 'risk_vs_volume.png')
        plt.savefig(plot_path)
        print(f"  - Plot saved to {plot_path}")
    else:
        print("  - Required columns not found for plotting")
    
    # Plot memory usage
    print("\nPlotting memory usage...")
    fig = tracker.plot_memory_usage()
    memory_plot_path = os.path.join(data_dir, 'excel_integration_memory.png')
    fig.savefig(memory_plot_path)
    print(f"Memory usage plot saved to {memory_plot_path}")
    
    print("\nExcel data integration example completed successfully!")


if __name__ == "__main__":
    main()