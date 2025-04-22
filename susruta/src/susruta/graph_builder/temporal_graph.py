# susruta/src/susruta/graph_builder/temporal_graph.py
"""
Temporal graph analysis for tracking glioma progression over time.

Builds and analyzes graphs that represent the temporal evolution of tumors
across multiple timepoints.
"""

from typing import Dict, List, Tuple, Any, Optional, Union, Set
import gc
import numpy as np
import networkx as nx
import pandas as pd
from pathlib import Path

from ..utils.memory import MemoryTracker


class TemporalGraphBuilder:
    """Builds and analyzes temporal graphs for glioma progression."""

    def __init__(self, memory_limit_mb: float = 3000):
        """
        Initialize temporal graph builder.

        Args:
            memory_limit_mb: Memory limit in MB
        """
        self.memory_limit_mb = memory_limit_mb
        self.memory_tracker = MemoryTracker(threshold_mb=memory_limit_mb)
    
    def build_temporal_graph(self, 
                           timepoint_graphs: Dict[int, nx.MultiDiGraph],
                           patient_id: int) -> nx.MultiDiGraph:
        """
        Build a unified temporal graph connecting multiple timepoints.

        Args:
            timepoint_graphs: Dictionary mapping timepoints to individual graphs
            patient_id: Patient identifier

        Returns:
            MultiDiGraph with temporal connections
        """
        self.memory_tracker.log_memory("Starting temporal graph construction")
        
        # Create a new graph to hold all timepoints
        temporal_graph = nx.MultiDiGraph()
        
        # Sort timepoints for sequential processing
        timepoints = sorted(timepoint_graphs.keys())
        
        if not timepoints:
            return temporal_graph  # Return empty graph if no timepoints
        
        # Add all nodes and edges from individual timepoint graphs
        for timepoint in timepoints:
            tp_graph = timepoint_graphs[timepoint]
            
            # Add timepoint suffix to node IDs to make them unique
            for node, attrs in tp_graph.nodes(data=True):
                # Skip nodes that don't belong to this patient
                node_patient_id = None
                if 'patient_' in node:
                    try:
                        node_patient_id = int(node.split('_')[1])
                    except (IndexError, ValueError):
                        pass
                elif attrs.get('type') == 'patient':
                    try:
                        node_patient_id = int(attrs.get('id', '0'))
                    except ValueError:
                        pass
                        
                if node_patient_id is not None and node_patient_id != patient_id:
                    continue
                
                # Create new node ID with timepoint
                temporal_node = f"{node}_tp{timepoint}"
                
                # Add timepoint attribute
                node_attrs = attrs.copy()
                node_attrs['timepoint'] = timepoint
                
                # Add node to temporal graph
                temporal_graph.add_node(temporal_node, **node_attrs)
            
            # Add edges within this timepoint
            for u, v, key, attrs in tp_graph.edges(data=True, keys=True):
                # Skip edges for other patients
                u_patient = None
                v_patient = None
                
                if 'patient_' in u:
                    try:
                        u_patient = int(u.split('_')[1])
                    except (IndexError, ValueError):
                        pass
                        
                if 'patient_' in v:
                    try:
                        v_patient = int(v.split('_')[1])
                    except (IndexError, ValueError):
                        pass
                
                if (u_patient is not None and u_patient != patient_id) or \
                   (v_patient is not None and v_patient != patient_id):
                    continue
                
                # Add edge with timepoint suffix
                temporal_u = f"{u}_tp{timepoint}"
                temporal_v = f"{v}_tp{timepoint}"
                
                if temporal_u in temporal_graph and temporal_v in temporal_graph:
                    temporal_graph.add_edge(temporal_u, temporal_v, key, **attrs)
        
        # Add temporal edges connecting nodes across timepoints
        if len(timepoints) > 1:
            # Connect sequential timepoints
            for i in range(len(timepoints) - 1):
                tp1 = timepoints[i]
                tp2 = timepoints[i + 1]
                
                self._connect_timepoints(temporal_graph, tp1, tp2, patient_id)
                
        self.memory_tracker.log_memory("Completed temporal graph construction")
        return temporal_graph
    
    def _connect_timepoints(self, 
                           graph: nx.MultiDiGraph, 
                           timepoint1: int, 
                           timepoint2: int,
                           patient_id: int) -> None:
        """
        Connect corresponding nodes between two timepoints.

        Args:
            graph: Temporal graph to modify
            timepoint1: First timepoint
            timepoint2: Second timepoint
            patient_id: Patient identifier
        """
        # Find nodes for each timepoint
        tp1_nodes = [n for n, attrs in graph.nodes(data=True) 
                    if attrs.get('timepoint') == timepoint1]
        tp2_nodes = [n for n, attrs in graph.nodes(data=True) 
                    if attrs.get('timepoint') == timepoint2]
        
        # Group nodes by their base ID (without timepoint suffix)
        tp1_dict = {}
        for node in tp1_nodes:
            base_id = node.rsplit('_tp', 1)[0]
            tp1_dict[base_id] = node
            
        tp2_dict = {}
        for node in tp2_nodes:
            base_id = node.rsplit('_tp', 1)[0]
            tp2_dict[base_id] = node
        
        # Connect matching nodes
        for base_id, tp1_node in tp1_dict.items():
            if base_id in tp2_dict:
                tp2_node = tp2_dict[base_id]
                
                # Get node types
                tp1_type = graph.nodes[tp1_node].get('type')
                tp2_type = graph.nodes[tp2_node].get('type')
                
                # Only connect same type nodes
                if tp1_type == tp2_type:
                    # Connect based on node type
                    if tp1_type == 'patient':
                        graph.add_edge(tp1_node, tp2_node, relation='same_patient')
                    elif tp1_type == 'tumor':
                        graph.add_edge(tp1_node, tp2_node, relation='progression')
                    elif tp1_type == 'treatment':
                        graph.add_edge(tp1_node, tp2_node, relation='continued_treatment')
                    else:
                        graph.add_edge(tp1_node, tp2_node, relation='temporal_sequence')
    
    def compute_progression_metrics(self, 
                                   temporal_graph: nx.MultiDiGraph,
                                   patient_id: int) -> Dict[str, Any]:
        """
        Compute tumor progression metrics from temporal graph.

        Args:
            temporal_graph: Temporal graph with multiple timepoints
            patient_id: Patient identifier

        Returns:
            Dictionary of progression metrics
        """
        metrics = {}
        
        # Get all timepoints
        timepoints = sorted(list(set(
            attrs.get('timepoint') for _, attrs in temporal_graph.nodes(data=True)
            if 'timepoint' in attrs
        )))
        
        if not timepoints:
            return metrics
        
        # Extract tumor volumes for each timepoint
        tumor_volumes = {}
        for timepoint in timepoints:
            # Find tumor node for this patient and timepoint
            tumor_nodes = [
                n for n, attrs in temporal_graph.nodes(data=True)
                if attrs.get('type') == 'tumor' and
                attrs.get('timepoint') == timepoint and
                f"tumor_{patient_id}" in n
            ]
            
            if tumor_nodes:
                tumor_node = tumor_nodes[0]  # Take the first matching tumor node
                volume = temporal_graph.nodes[tumor_node].get('volume_mm3', 0.0)
                tumor_volumes[timepoint] = float(volume)
        
        # Calculate growth rates
        if len(tumor_volumes) > 1:
            growth_rates = {}
            volume_values = []
            
            for i in range(len(timepoints) - 1):
                tp1 = timepoints[i]
                tp2 = timepoints[i + 1]
                
                if tp1 in tumor_volumes and tp2 in tumor_volumes:
                    vol1 = tumor_volumes[tp1]
                    vol2 = tumor_volumes[tp2]
                    
                    if vol1 > 0:  # Avoid division by zero
                        growth_rate = (vol2 - vol1) / vol1
                        growth_rates[f"{tp1}_to_{tp2}"] = growth_rate
                    else:
                        growth_rates[f"{tp1}_to_{tp2}"] = float('inf') if vol2 > 0 else 0.0
                        
                    volume_values.append(vol1)
            
            # Add the last volume
            if timepoints[-1] in tumor_volumes:
                volume_values.append(tumor_volumes[timepoints[-1]])
            
            # Calculate metrics
            metrics['tumor_volumes'] = tumor_volumes
            metrics['growth_rates'] = growth_rates
            
            if volume_values:
                metrics['initial_volume'] = volume_values[0]
                metrics['final_volume'] = volume_values[-1]
                metrics['max_volume'] = max(volume_values)
                metrics['volume_change'] = volume_values[-1] - volume_values[0]
                metrics['volume_change_percent'] = (
                    (volume_values[-1] - volume_values[0]) / volume_values[0] * 100
                    if volume_values[0] > 0 else float('inf')
                )
                metrics['avg_growth_rate'] = sum(growth_rates.values()) / len(growth_rates) if growth_rates else 0.0
        else:
            # Only one timepoint, can't calculate growth rate
            metrics['tumor_volumes'] = tumor_volumes
            if timepoints and timepoints[0] in tumor_volumes:
                metrics['initial_volume'] = tumor_volumes[timepoints[0]]
        
        return metrics
    
    def identify_progression_patterns(self, 
                                     temporal_graph: nx.MultiDiGraph,
                                     patient_id: int) -> Dict[str, Any]:
        """
        Identify patterns in tumor progression from temporal graph.

        Args:
            temporal_graph: Temporal graph with multiple timepoints
            patient_id: Patient identifier

        Returns:
            Dictionary of identified progression patterns
        """
        patterns = {}
        
        # Get progression metrics
        metrics = self.compute_progression_metrics(temporal_graph, patient_id)
        
        # Check for growth pattern
        if 'growth_rates' in metrics and metrics['growth_rates']:
            growth_rates = list(metrics['growth_rates'].values())
            
            # Determine overall growth pattern
            if all(rate > 0.1 for rate in growth_rates):
                patterns['growth_pattern'] = 'rapid_growth'
            elif all(rate > 0 for rate in growth_rates):
                patterns['growth_pattern'] = 'steady_growth'
            elif all(rate < 0 for rate in growth_rates):
                patterns['growth_pattern'] = 'regression'
            elif any(rate > 0 for rate in growth_rates) and any(rate < 0 for rate in growth_rates):
                patterns['growth_pattern'] = 'variable'
            else:
                patterns['growth_pattern'] = 'stable'
            
            # Check for acceleration
            if len(growth_rates) > 1:
                diffs = [growth_rates[i+1] - growth_rates[i] for i in range(len(growth_rates)-1)]
                if all(diff > 0 for diff in diffs):
                    patterns['acceleration'] = 'increasing'
                elif all(diff < 0 for diff in diffs):
                    patterns['acceleration'] = 'decreasing'
                else:
                    patterns['acceleration'] = 'variable'
        
        # Analyze treatment response
        patterns['treatment_response'] = self._analyze_treatment_response(
            temporal_graph, 
            patient_id,
            metrics.get('tumor_volumes', {})
        )
        
        return patterns
    
    def _analyze_treatment_response(self,
                                   temporal_graph: nx.MultiDiGraph,
                                   patient_id: int,
                                   tumor_volumes: Dict[int, float]) -> str:
        """
        Analyze treatment response based on tumor volume changes.

        Args:
            temporal_graph: Temporal graph with multiple timepoints
            patient_id: Patient identifier
            tumor_volumes: Dictionary of tumor volumes by timepoint

        Returns:
            Treatment response category
        """
        # Get treatment timepoints
        treatment_timepoints = {}
        
        for node, attrs in temporal_graph.nodes(data=True):
            if attrs.get('type') == 'treatment' and f"{patient_id}" in node:
                timepoint = attrs.get('timepoint')
                if timepoint is not None:
                    treatment_category = attrs.get('category', 'unknown')
                    if timepoint not in treatment_timepoints:
                        treatment_timepoints[timepoint] = []
                    treatment_timepoints[timepoint].append(treatment_category)
        
        if not treatment_timepoints or not tumor_volumes:
            return 'unknown'
        
        # Get timepoints
        all_timepoints = sorted(list(set(
            list(tumor_volumes.keys()) + list(treatment_timepoints.keys())
        )))
        
        if len(all_timepoints) < 2:
            return 'unknown'
        
        # Find post-treatment volume changes
        post_treatment_changes = {}
        
        for tp in treatment_timepoints:
            # Find next timepoint with volume data
            next_tps = [t for t in all_timepoints if t > tp and t in tumor_volumes]
            if next_tps:
                next_tp = next_tps[0]
                if tp in tumor_volumes:
                    # Calculate volume change after treatment
                    change = tumor_volumes[next_tp] - tumor_volumes[tp]
                    relative_change = change / tumor_volumes[tp] if tumor_volumes[tp] > 0 else float('inf')
                    post_treatment_changes[tp] = relative_change
        
        if not post_treatment_changes:
            return 'unknown'
        
        # Determine overall response
        avg_change = sum(post_treatment_changes.values()) / len(post_treatment_changes)
        
        if avg_change < -0.3:
            return 'good_response'
        elif avg_change < 0:
            return 'partial_response'
        elif avg_change < 0.1:
            return 'stable_disease'
        else:
            return 'progressive_disease'
    
    def extract_temporal_features(self, 
                                temporal_graph: nx.MultiDiGraph,
                                patient_id: int) -> Dict[str, Any]:
        """
        Extract features from temporal graph for machine learning.

        Args:
            temporal_graph: Temporal graph with multiple timepoints
            patient_id: Patient identifier

        Returns:
            Dictionary of temporal features
        """
        features = {}
        
        # Get progression metrics
        metrics = self.compute_progression_metrics(temporal_graph, patient_id)
        patterns = self.identify_progression_patterns(temporal_graph, patient_id)
        
        # Extract basic features
        features['timepoints'] = len(metrics.get('tumor_volumes', {}))
        features['initial_volume'] = metrics.get('initial_volume', 0.0)
        features['final_volume'] = metrics.get('final_volume', 0.0)
        features['max_volume'] = metrics.get('max_volume', 0.0)
        features['volume_change'] = metrics.get('volume_change', 0.0)
        features['volume_change_percent'] = metrics.get('volume_change_percent', 0.0)
        features['avg_growth_rate'] = metrics.get('avg_growth_rate', 0.0)
        
        # Convert categorical patterns to numerical features
        growth_pattern = patterns.get('growth_pattern', 'unknown')
        if growth_pattern == 'rapid_growth':
            features['growth_pattern_numeric'] = 3.0
        elif growth_pattern == 'steady_growth':
            features['growth_pattern_numeric'] = 2.0
        elif growth_pattern == 'variable':
            features['growth_pattern_numeric'] = 1.0
        elif growth_pattern == 'stable':
            features['growth_pattern_numeric'] = 0.0
        elif growth_pattern == 'regression':
            features['growth_pattern_numeric'] = -1.0
        else:
            features['growth_pattern_numeric'] = 0.0
        
        # Treatment response feature
        treatment_response = patterns.get('treatment_response', 'unknown')
        if treatment_response == 'good_response':
            features['treatment_response_numeric'] = 3.0
        elif treatment_response == 'partial_response':
            features['treatment_response_numeric'] = 2.0
        elif treatment_response == 'stable_disease':
            features['treatment_response_numeric'] = 1.0
        elif treatment_response == 'progressive_disease':
            features['treatment_response_numeric'] = 0.0
        else:
            features['treatment_response_numeric'] = 0.0
        
        # Calculate more advanced features
        
        # Growth stability (variance in growth rates)
        growth_rates = list(metrics.get('growth_rates', {}).values())
        if growth_rates:
            features['growth_variance'] = np.var(growth_rates)
        else:
            features['growth_variance'] = 0.0
        
        # Calculate volume doubling time if growing
        if features['avg_growth_rate'] > 0:
            features['doubling_time'] = 0.693 / features['avg_growth_rate']
        else:
            features['doubling_time'] = float('inf')
        
        return features
    
    def merge_latest_graph(self,
                          temporal_graph: nx.MultiDiGraph,
                          static_graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
        """
        Merge the latest timepoint from temporal graph with a static graph.
        
        Useful for incorporating progression history into current analysis.

        Args:
            temporal_graph: Temporal graph with multiple timepoints
            static_graph: Static graph to merge with

        Returns:
            Merged graph
        """
        # Get the latest timepoint
        timepoints = sorted(list(set(
            attrs.get('timepoint') for _, attrs in temporal_graph.nodes(data=True)
            if 'timepoint' in attrs
        )))
        
        if not timepoints:
            return static_graph.copy()  # No timepoints, return static graph
            
        latest_timepoint = timepoints[-1]
        
        # Create a new graph
        merged_graph = static_graph.copy()
        
        # Add nodes from latest timepoint
        for node, attrs in temporal_graph.nodes(data=True):
            if attrs.get('timepoint') == latest_timepoint:
                # Remove timepoint suffix and attribute
                base_node = node.rsplit('_tp', 1)[0]
                clean_attrs = attrs.copy()
                if 'timepoint' in clean_attrs:
                    del clean_attrs['timepoint']
                
                # Check if node exists in static graph
                if base_node in merged_graph:
                    # Update attributes
                    for key, value in clean_attrs.items():
                        merged_graph.nodes[base_node][key] = value
                else:
                    # Add new node
                    merged_graph.add_node(base_node, **clean_attrs)
        
        # Add edges between latest timepoint nodes
        for u, v, key, attrs in temporal_graph.edges(data=True, keys=True):
            u_attrs = temporal_graph.nodes.get(u, {})
            v_attrs = temporal_graph.nodes.get(v, {})
            
            # Only process edges where both nodes are from latest timepoint
            if u_attrs.get('timepoint') == latest_timepoint and v_attrs.get('timepoint') == latest_timepoint:
                # Remove timepoint suffix
                base_u = u.rsplit('_tp', 1)[0]
                base_v = v.rsplit('_tp', 1)[0]
                
                # Add edge if both nodes exist
                if base_u in merged_graph and base_v in merged_graph:
                    # Check if edge already exists
                    if not merged_graph.has_edge(base_u, base_v, key):
                        merged_graph.add_edge(base_u, base_v, key, **attrs)
        
        return merged_graph