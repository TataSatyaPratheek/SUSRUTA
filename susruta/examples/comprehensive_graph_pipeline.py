#!/usr/bin/env python
# examples/comprehensive_graph_pipeline.py
"""
Comprehensive example demonstrating the complete unified graph pipeline:
1. Loading and processing Excel data
2. Building MRI-based graphs
3. Integrating multimodal data sources
4. Creating a unified temporal knowledge graph
5. Converting to PyTorch Geometric format for GNN
6. Visualizing the results
"""

import os
import sys
import argparse
import networkx as nx
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import plotly.io as pio

from susruta.data.excel_loader import ExcelDataLoader
from susruta.data.mri import EfficientMRIProcessor
from susruta.graph.knowledge_graph import GliomaKnowledgeGraph
from susruta.graph_builder.mri_graph import MRIGraphBuilder
from susruta.graph_builder.multimodal_graph import MultimodalGraphIntegrator
from susruta.graph_builder.temporal_graph import TemporalGraphBuilder
from susruta.graph_builder.unified_builder import UnifiedGraphBuilder
from susruta.graph_builder.visualization import GraphVisualizer
from susruta.models.gnn import GliomaGNN
from susruta.treatment.simulator import TreatmentSimulator


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='SUSRUTA comprehensive graph pipeline example')
    
    parser.add_argument('--patient-id', type=int, required=True,
                       help='Patient ID to process')
    parser.add_argument('--mri-dir', type=str, required=True,
                       help='Directory containing MRI data')
    parser.add_argument('--scanner-excel', type=str, required=True,
                       help='Path to MR_Scanner_data.xlsx')
    parser.add_argument('--clinical-excel', type=str, required=True,
                       help='Path to MUGliomaPost_ClinicalDataFINAL032025.xlsx')
    parser.add_argument('--segmentation-excel', type=str, required=True,
                       help='Path to MUGliomaPost_Segmentation_Volumes.xlsx')
    parser.add_argument('--output-dir', type=str, default='./output',
                       help='Directory to save results and visualizations')
    parser.add_argument('--timepoints', type=int, nargs='+', default=None,
                       help='Specific timepoints to process (e.g., 1 2 3)')
    parser.add_argument('--memory-limit', type=int, default=7000,
                       help='Memory limit in MB (default: 7000)')
    
    return parser.parse_args()


def main():
    """Main execution function for the comprehensive pipeline."""
    # Parse arguments
    args = parse_arguments()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Processing patient {args.patient_id}")
    print(f"Memory limit: {args.memory_limit} MB")
    
    # Check if input files exist
    for path in [args.scanner_excel, args.clinical_excel, args.segmentation_excel]:
        if not os.path.exists(path):
            print(f"Error: File not found: {path}")
            return 1
    
    if not os.path.exists(args.mri_dir):
        print(f"Error: MRI directory not found: {args.mri_dir}")
        return 1
    
    # Set up Excel paths dictionary
    excel_paths = {
        'scanner': args.scanner_excel,
        'clinical': args.clinical_excel,
        'segmentation': args.segmentation_excel
    }
    
    # 1. Initialize the unified graph builder
    unified_builder = UnifiedGraphBuilder(memory_limit_mb=args.memory_limit)
    
    try:
        # 2. Build unified graph from all available data
        print(f"Building unified graph for patient {args.patient_id}...")
        unified_graph = unified_builder.build_unified_graph(
            patient_id=args.patient_id,
            mri_dir=args.mri_dir,
            excel_paths=excel_paths,
            timepoints=args.timepoints,
            include_temporal=True
        )
        print(f"Graph built with {unified_graph.number_of_nodes()} nodes and {unified_graph.number_of_edges()} edges")
        
        # Get graph statistics
        node_types = {}
        edge_types = {}
        
        for node, attrs in unified_graph.nodes(data=True):
            node_type = attrs.get('type', 'unknown')
            if node_type not in node_types:
                node_types[node_type] = 0
            node_types[node_type] += 1
        
        for u, v, attrs in unified_graph.edges(data=True):
            edge_type = attrs.get('relation', 'unknown')
            if edge_type not in edge_types:
                edge_types[edge_type] = 0
            edge_types[edge_type] += 1
        
        print("\nGraph composition:")
        print("Node types:")
        for node_type, count in node_types.items():
            print(f"  {node_type}: {count}")
        
        print("\nEdge types:")
        for edge_type, count in edge_types.items():
            print(f"  {edge_type}: {count}")
        
        # 3. Generate a patient summary
        print("\nGenerating patient summary...")
        patient_summary = unified_builder.get_patient_summary(unified_graph, args.patient_id)
        
        # Print summary (formatted)
        print("\nPatient Summary:")
        print(f"Patient ID: {patient_summary['patient_id']}")
        
        # Demographics
        print("\nDemographics:")
        for key, value in patient_summary['demographics'].items():
            print(f"  {key}: {value}")
        
        # Tumor
        print("\nTumor Characteristics:")
        for key, value in patient_summary['tumor'].items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")
        
        # Treatments
        print("\nTreatments:")
        for treatment in patient_summary['treatments']:
            print(f"  {treatment.get('category', 'unknown').title()} - Response: {treatment.get('response', 'unknown')}")
        
        # 4. Convert to PyTorch Geometric format for GNN
        print("\nConverting to PyTorch Geometric format...")
        pyg_data = unified_builder.build_pytorch_geometric(unified_graph)
        
        print("PyTorch Geometric conversion complete")
        print(f"Node types: {pyg_data.node_types}")
        print(f"Edge types: {pyg_data.edge_types}")
        
        # 5. Temporal analysis
        if 'timepoint' in next(iter(unified_graph.nodes(data=True)))[1]:
            print("\nAnalyzing temporal progression...")
            temporal_builder = TemporalGraphBuilder()
            progression_metrics = temporal_builder.compute_progression_metrics(unified_graph, args.patient_id)
            progression_patterns = temporal_builder.identify_progression_patterns(unified_graph, args.patient_id)
            
            # Print progression metrics
            print("\nTumor Progression Metrics:")
            if 'tumor_volumes' in progression_metrics:
                print("  Tumor volumes by timepoint:")
                for tp, volume in progression_metrics['tumor_volumes'].items():
                    print(f"    Timepoint {tp}: {volume:.2f} mm³")
            
            if 'growth_rates' in progression_metrics:
                print("  Growth rates between timepoints:")
                for key, rate in progression_metrics['growth_rates'].items():
                    print(f"    {key}: {rate*100:.2f}%")
            
            print("\nProgression Patterns:")
            for key, pattern in progression_patterns.items():
                print(f"  {key}: {pattern}")
        
        # 6. Treatment simulation
        print("\nSimulating treatment options...")
        
        # Initialize GliomaGNN model
        node_feature_dims = {node_type: data.x.size(1) for node_type, data in pyg_data.items() if hasattr(data, 'x')}
        edge_types = [(str(source), str(relation), str(target)) for source, relation, target in pyg_data.edge_types]
        edge_feature_dims = {edge_type: 1 for edge_type in edge_types}
        
        gnn_model = GliomaGNN(
            node_feature_dims=node_feature_dims,
            edge_feature_dims=edge_feature_dims,
            hidden_channels=32
        )
        
        # Dummy training would be done here in a real scenario
        # For this example, we'll just use the untrained model
        
        # Create treatment simulator
        treatment_simulator = TreatmentSimulator(gnn_model, unified_graph)
        
        # Define treatment options to simulate
        treatment_options = [
            {'category': 'surgery', 'dose': None, 'duration': 1},
            {'category': 'radiation', 'dose': 60.0, 'duration': 30},
            {'category': 'chemotherapy', 'dose': 150.0, 'duration': 180},
            {'category': 'combined', 'dose': 200.0, 'duration': 180}
        ]
        
        # Simulate treatments
        tumor_node = f"tumor_{args.patient_id}"
        patient_node = f"patient_{args.patient_id}"
        
        simulation_results = treatment_simulator.simulate_treatments(
            patient_id=patient_node,
            tumor_id=tumor_node,
            treatment_options=treatment_options,
            data=pyg_data
        )
        
        # Rank treatments
        ranked_treatments = treatment_simulator.rank_treatments(simulation_results)
        
        # Print simulation results
        print("\nTreatment Simulation Results (ranked):")
        for i, option_id in enumerate(ranked_treatments):
            option = simulation_results[option_id]
            print(f"\n{i+1}. {option['config']['category'].title()}")
            print(f"   Response probability: {option['response_prob']*100:.1f}%")
            print(f"   Survival days: {option['survival_days']:.0f}")
            print(f"   Uncertainty: {option['uncertainty']:.3f}")
        
        # 7. Create visualizations
        print("\nGenerating visualizations...")
        visualizer = GraphVisualizer(output_dir=args.output_dir, use_plotly=True)
        
        # Visualization 1: Knowledge Graph
        print("Creating knowledge graph visualization...")
        fig_kg = visualizer.visualize_knowledge_graph(
            graph=unified_graph,
            title=f"Knowledge Graph for Patient {args.patient_id}",
            save_path=f"patient_{args.patient_id}_knowledge_graph.html"
        )
        
        # Visualization 2: Tumor Progression
        if 'timepoint' in next(iter(unified_graph.nodes(data=True)))[1]:
            print("Creating tumor progression visualization...")
            fig_prog = visualizer.visualize_tumor_progression(
                temporal_graph=unified_graph,
                patient_id=args.patient_id,
                save_path=f"patient_{args.patient_id}_tumor_progression.html"
            )
        
        # Visualization 3: Treatment Comparison
        print("Creating treatment comparison visualization...")
        fig_treat = visualizer.visualize_treatment_comparison(
            treatment_simulation=simulation_results,
            patient_id=args.patient_id,
            save_path=f"patient_{args.patient_id}_treatment_comparison.html"
        )
        
        # Visualization 4: Tumor Regions
        print("Creating tumor regions visualization...")
        fig_regions = visualizer.visualize_tumor_regions(
            graph=unified_graph,
            patient_id=args.patient_id,
            save_path=f"patient_{args.patient_id}_tumor_regions.html"
        )
        
        print(f"\nAll visualizations saved to {args.output_dir}")
        
        # Print summary of process
        print("\nSummary of pipeline execution:")
        print(f"Patient ID: {args.patient_id}")
        print(f"Data sources:")
        print(f"  - MRI data: {args.mri_dir}")
        print(f"  - Excel files: {len(excel_paths)}")
        print(f"Graph statistics:")
        print(f"  - Nodes: {unified_graph.number_of_nodes()}")
        print(f"  - Edges: {unified_graph.number_of_edges()}")
        print(f"  - Node types: {len(node_types)}")
        print(f"  - Edge types: {len(edge_types)}")
        print(f"Treatment options evaluated: {len(treatment_options)}")
        print(f"Visualizations created: 4")
        print("\nPipeline completed successfully!")
        
        return 0
        
    except Exception as e:
        print(f"Error in pipeline execution: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())