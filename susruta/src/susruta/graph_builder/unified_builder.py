# susruta/src/susruta/graph_builder/unified_builder.py
"""
Unified graph builder for glioma treatment prediction.

Provides a comprehensive API for building, integrating, and analyzing
knowledge graphs from multiple data sources.
"""

from typing import Dict, List, Tuple, Any, Optional, Union, Set
import os
import gc
import numpy as np
import networkx as nx
import pandas as pd
from pathlib import Path
import torch
from torch_geometric.data import HeteroData

from ..data.mri import EfficientMRIProcessor
from ..data.clinical import ClinicalDataProcessor
from ..data.excel_loader import ExcelDataLoader
from ..data.excel_integration import MultimodalDataIntegrator
from ..graph.knowledge_graph import GliomaKnowledgeGraph
from ..utils.memory import MemoryTracker
from .mri_graph import MRIGraphBuilder
from .multimodal_graph import MultimodalGraphIntegrator
from .temporal_graph import TemporalGraphBuilder


class UnifiedGraphBuilder:
    """Unified API for comprehensive graph construction and analysis."""

    def __init__(self, memory_limit_mb: float = 7000):
        """
        Initialize unified graph builder.

        Args:
            memory_limit_mb: Memory limit in MB
        """
        self.memory_limit_mb = memory_limit_mb
        self.memory_tracker = MemoryTracker(threshold_mb=memory_limit_mb)
        
        # Sub-allocate memory to individual components
        mri_memory = memory_limit_mb * 0.3  # 30% for MRI
        excel_memory = memory_limit_mb * 0.1  # 10% for Excel
        graph_memory = memory_limit_mb * 0.2  # 20% for graph construction
        temporal_memory = memory_limit_mb * 0.1  # 10% for temporal
        # Rest (30%) reserved for unified operations
        
        # Initialize sub-components
        self.mri_builder = MRIGraphBuilder(memory_limit_mb=mri_memory)
        self.graph_integrator = MultimodalGraphIntegrator(memory_limit_mb=graph_memory)
        self.temporal_builder = TemporalGraphBuilder(memory_limit_mb=temporal_memory)
        self.excel_integrator = MultimodalDataIntegrator(memory_limit_mb=excel_memory)
        
        # Initialize cache for processed data
        self._cached_graphs = {}
        self._cached_pytorch_geometric = None
    
    def build_unified_graph(self,
                          patient_id: int,
                          mri_dir: str,
                          excel_paths: Dict[str, str],
                          timepoints: Optional[List[int]] = None,
                          include_temporal: bool = True) -> nx.MultiDiGraph:
        """
        Build comprehensive unified graph from all available data sources.

        Args:
            patient_id: Patient identifier
            mri_dir: Directory containing MRI data
            excel_paths: Dictionary of paths to Excel files
            timepoints: Optional list of timepoints to process
            include_temporal: Whether to include temporal connections

        Returns:
            Unified MultiDiGraph with all integrated data
        """
        self.memory_tracker.log_memory("Starting unified graph construction")
        
        # Determine timepoints to process
        if timepoints is None:
            # Auto-detect available timepoints
            timepoints = self._detect_available_timepoints(patient_id, mri_dir)
            if not timepoints:
                timepoints = [1]  # Default to timepoint 1 if none detected
        
        # Process each timepoint
        timepoint_graphs = {}
        for tp in timepoints:
            self.memory_tracker.log_memory(f"Processing timepoint {tp}")
            
            # Build MRI graph for this timepoint
            try:
                mri_graph = self.mri_builder.build_mri_graph(
                    patient_id=patient_id,
                    data_dir=mri_dir,
                    timepoint=tp
                )
                self.memory_tracker.log_memory(f"Built MRI graph for timepoint {tp}")
            except Exception as e:
                print(f"Warning: Failed to build MRI graph for timepoint {tp}: {e}")
                mri_graph = nx.Graph()  # Empty graph on failure
            
            # Build Excel-based graph
            try:
                excel_graph = self.graph_integrator.from_excel_sources(
                    scanner_path=excel_paths.get('scanner', ''),
                    clinical_path=excel_paths.get('clinical', ''),
                    segmentation_path=excel_paths.get('segmentation', ''),
                    patient_id=patient_id,
                    timepoint=tp
                )
                self.memory_tracker.log_memory(f"Built Excel graph for timepoint {tp}")
            except Exception as e:
                print(f"Warning: Failed to build Excel graph for timepoint {tp}: {e}")
                excel_graph = nx.MultiDiGraph()  # Empty graph on failure
            
            # Integrate graphs for this timepoint
            integrated_graph = self.graph_integrator.integrate_graphs(
                mri_graph=mri_graph,
                clinical_graph=excel_graph,
                patient_id=patient_id
            )
            self.memory_tracker.log_memory(f"Integrated graphs for timepoint {tp}")
            
            # Cache graphs
            timepoint_graphs[tp] = integrated_graph
            self._cached_graphs[f"{patient_id}_{tp}"] = integrated_graph.copy()
            
            # Force garbage collection
            del mri_graph
            del excel_graph
            gc.collect()
        
        # Build temporal graph if multiple timepoints
        if include_temporal and len(timepoints) > 1:
            temporal_graph = self.temporal_builder.build_temporal_graph(
                timepoint_graphs=timepoint_graphs,
                patient_id=patient_id
            )
            self.memory_tracker.log_memory("Built temporal graph")
            
            # Extract progression features
            progression_features = self.temporal_builder.extract_temporal_features(
                temporal_graph=temporal_graph,
                patient_id=patient_id
            )
            print(f"Extracted progression features: {progression_features}")
            
            return temporal_graph
        else:
            # Return the latest timepoint graph
            latest_tp = max(timepoints)
            return timepoint_graphs[latest_tp]
    
    def build_pytorch_geometric(self, 
                              graph: nx.MultiDiGraph,
                              precomputed_features: Optional[Dict[str, Dict[str, List[Any]]]] = None) -> HeteroData:
        """
        Convert unified graph to PyTorch Geometric format.

        Args:
            graph: Unified MultiDiGraph
            precomputed_features: Optional precomputed node features

        Returns:
            PyTorch Geometric HeteroData object
        """
        self.memory_tracker.log_memory("Starting conversion to PyTorch Geometric")
        
        # Create a knowledge graph builder
        kg = GliomaKnowledgeGraph(memory_limit_mb=self.memory_limit_mb * 0.5)
        
        # Set the internal graph to our unified graph
        kg.G = graph
        
        # Convert to PyTorch Geometric
        pyg_data = kg.to_pytorch_geometric(node_features=precomputed_features)
        
        # Cache result
        self._cached_pytorch_geometric = pyg_data
        
        self.memory_tracker.log_memory("Completed conversion to PyTorch Geometric")
        return pyg_data
    
    def _detect_available_timepoints(self, patient_id: int, mri_dir: str) -> List[int]:
        """
        Auto-detect available timepoints for a patient.

        Args:
            patient_id: Patient identifier
            mri_dir: Directory containing MRI data

        Returns:
            List of available timepoints
        """
        timepoints = []
        patient_dir = os.path.join(mri_dir, f'PatientID_{patient_id:04d}')
        
        if not os.path.exists(patient_dir):
            return timepoints
        
        # Look for timepoint directories
        import glob
        timepoint_dirs = glob.glob(os.path.join(patient_dir, 'Timepoint_*'))
        
        for tp_dir in timepoint_dirs:
            try:
                # Extract timepoint number from directory name
                tp_str = os.path.basename(tp_dir).split('_')[1]
                timepoint = int(tp_str)
                timepoints.append(timepoint)
            except (IndexError, ValueError):
                continue
        
        return sorted(timepoints)
    
    def add_custom_features(self, 
                          graph: nx.MultiDiGraph, 
                          features: Dict[str, Dict[str, Any]]) -> nx.MultiDiGraph:
        """
        Add custom features to nodes in the graph.

        Args:
            graph: MultiDiGraph to add features to
            features: Dictionary mapping node IDs to feature dictionaries

        Returns:
            Updated graph
        """
        updated_graph = graph.copy()
        
        # Add features to corresponding nodes
        for node_id, node_features in features.items():
            if node_id in updated_graph:
                # Update node attributes
                for feature_name, feature_value in node_features.items():
                    updated_graph.nodes[node_id][feature_name] = feature_value
        
        return updated_graph
    
    def get_patient_summary(self, 
                          graph: nx.MultiDiGraph, 
                          patient_id: int) -> Dict[str, Any]:
        """
        Generate comprehensive patient summary from the unified graph.

        Args:
            graph: Unified MultiDiGraph
            patient_id: Patient identifier

        Returns:
            Dictionary with patient summary information
        """
        summary = {
            'patient_id': patient_id,
            'demographics': {},
            'tumor': {},
            'treatments': [],
            'sequences': {},
            'temporal': {}
        }
        
        # Get patient node
        patient_node = f"patient_{patient_id}"
        if patient_node in graph:
            patient_attrs = graph.nodes[patient_node]
            # Extract demographics
            for attr in ['age', 'sex', 'karnofsky']:
                if attr in patient_attrs:
                    summary['demographics'][attr] = patient_attrs[attr]
        
        # Get tumor info
        tumor_node = f"tumor_{patient_id}"
        if tumor_node in graph:
            tumor_attrs = graph.nodes[tumor_node]
            # Extract tumor attributes
            for attr in ['grade', 'histology', 'location', 'idh_status', 'mgmt_status', 
                        'volume_mm3', 'volume_voxels', 'surface_area', 'elongation', 'roundness']:
                if attr in tumor_attrs:
                    summary['tumor'][attr] = tumor_attrs[attr]
        
        # Get treatments
        for node, attrs in graph.nodes(data=True):
            if attrs.get('type') == 'treatment':
                # Check if this treatment is for our patient
                is_for_patient = False
                # --- START FIX: Check incoming edges from tumor ---
                for u, v, data in graph.in_edges(node, data=True): # Check incoming edges
                    # Check if the source node 'u' is the tumor node and relation is correct
                    if u == tumor_node and data.get('relation') == 'treated_with':
                        is_for_patient = True
                        break
                # --- END FIX ---

                
                if is_for_patient:
                    treatment = {
                        'id': node,
                        'category': attrs.get('category', 'unknown'),
                        'name': attrs.get('name', attrs.get('category', 'unknown')),
                        'dose': attrs.get('dose', None),
                        'duration': attrs.get('duration', None),
                        'start_day': attrs.get('start_day', None)
                    }
                    
                    # Look for outcomes
                    for _, v in graph.out_edges(node):
                        v_attrs = graph.nodes[v]
                        if v_attrs.get('type') == 'outcome':
                            treatment['response'] = v_attrs.get('response', None)
                            treatment['progression_free_days'] = v_attrs.get('progression_free_days', None)
                            treatment['survival_days'] = v_attrs.get('survival_days', None)
                    
                    summary['treatments'].append(treatment)
        
        # Get MRI sequences
        for node, attrs in graph.nodes(data=True):
            if attrs.get('type') == 'sequence':
                sequence_type = attrs.get('sequence', 'unknown')
                if sequence_type not in summary['sequences']:
                    summary['sequences'][sequence_type] = {}
                
                for key, value in attrs.items():
                    if key not in ['type', 'sequence']:
                        summary['sequences'][sequence_type][key] = value
        
        # Get temporal features if available - check for timepoint attribute
        has_temporal = any('timepoint' in attrs for _, attrs in graph.nodes(data=True))
        
        if has_temporal:
            # Try to extract temporal features
            try:
                temporal_features = self.temporal_builder.extract_temporal_features(
                    temporal_graph=graph,
                    patient_id=patient_id
                )
                summary['temporal'] = temporal_features
            except Exception as e:
                print(f"Warning: Failed to extract temporal features: {e}")
        
        return summary
    
    def get_node_subgraph(self,
                         graph: nx.MultiDiGraph,
                         node_id: str,
                         depth: int = 1) -> nx.MultiDiGraph:
        """
        Extract a subgraph centered around a specific node.

        Args:
            graph: Unified MultiDiGraph
            node_id: Central node identifier
            depth: Number of hops from central node

        Returns:
            Subgraph as MultiDiGraph
        """
        if node_id not in graph:
            return nx.MultiDiGraph()  # Return empty graph if node not found
        
        # Find nodes within 'depth' hops
        nodes_to_keep = {node_id}
        current_nodes = {node_id}
        
        # BFS to find nodes within depth
        for _ in range(depth):
            next_nodes = set()
            for node in current_nodes:
                # Get neighbors (both in and out edges for directed graph)
                neighbors = set(graph.predecessors(node)).union(set(graph.successors(node)))
                next_nodes.update(neighbors)
            
            nodes_to_keep.update(next_nodes)
            current_nodes = next_nodes
        
        # Create subgraph
        subgraph = graph.subgraph(nodes_to_keep).copy()
        
        return subgraph
    
    def clear_cache(self) -> None:
        """Clear all cached graphs and data to free memory."""
        self._cached_graphs.clear()
        self._cached_pytorch_geometric = None
        
        # Force garbage collection
        gc.collect()
        
        self.memory_tracker.log_memory("Cleared cache")