"""
Knowledge graph construction for glioma treatment data.

Builds a heterogeneous graph from clinical, imaging, and treatment data
with memory-efficient implementation.
"""

from typing import Dict, List, Tuple, Any, Optional, Set, Union
import gc
import numpy as np
import pandas as pd
import networkx as nx
from torch_geometric.data import HeteroData
import torch

from ..utils.memory import MemoryTracker


class GliomaKnowledgeGraph:
    """Construct a knowledge graph for glioma treatment data with memory efficiency."""
    
    def __init__(self, memory_limit_mb: float = 3000):
        """
        Initialize knowledge graph builder.
        
        Args:
            memory_limit_mb: Memory limit in MB
        """
        self.memory_limit_mb = memory_limit_mb
        self.memory_tracker = MemoryTracker(threshold_mb=memory_limit_mb)
        self.G = nx.MultiDiGraph()
    
    def add_clinical_data(self, clinical_df: pd.DataFrame, batch_size: int = 20) -> 'GliomaKnowledgeGraph':
        """
        Add clinical data to the graph in batches to control memory usage.
        
        Args:
            clinical_df: DataFrame with clinical information
            batch_size: Number of patients to process in each batch
            
        Returns:
            Self for method chaining
        """
        self.memory_tracker.log_memory("Start adding clinical data")
        
        # Process in batches
        for i in range(0, len(clinical_df), batch_size):
            batch = clinical_df.iloc[i:i+batch_size]
            
            for _, row in batch.iterrows():
                patient_id = f"patient_{row['patient_id']}"
                
                # Add patient node with attributes
                patient_attrs = {
                    'type': 'patient',
                    'age': row.get('age', None),
                    'sex': row.get('sex', None),
                    'karnofsky': row.get('karnofsky_score', None)
                }
                self.G.add_node(patient_id, **patient_attrs)
                
                # Add tumor node
                tumor_id = f"tumor_{row['patient_id']}"
                tumor_attrs = {
                    'type': 'tumor',
                    'grade': row.get('grade', None),
                    'histology': row.get('histology', None),
                    'location': row.get('location', None),
                    'idh_status': row.get('idh_mutation', None),
                    'mgmt_status': row.get('mgmt_methylation', None)
                }
                self.G.add_node(tumor_id, **tumor_attrs)
                
                # Connect patient to tumor
                self.G.add_edge(patient_id, tumor_id, relation='has_tumor')
            
            # Track memory after each batch
            self.memory_tracker.log_memory(f"Added clinical batch {i//batch_size + 1}")
            
            # If approaching memory limit, apply compression
            if self.memory_tracker.current_memory_mb() > self.memory_limit_mb * 0.8:
                self._compress_graph()
        
        return self
    
    def add_imaging_features(self, 
                           imaging_features: Dict[int, Dict[str, Dict[str, float]]], 
                           max_features_per_node: int = 20) -> 'GliomaKnowledgeGraph':
        """
        Add imaging features to the graph efficiently.
        
        Args:
            imaging_features: Dictionary mapping patient IDs to imaging features
            max_features_per_node: Maximum number of features to add per node
            
        Returns:
            Self for method chaining
        """
        self.memory_tracker.log_memory("Start adding imaging features")
        
        # Add feature nodes first (shared across patients)
        all_feature_names: Set[str] = set()
        for patient_id, features in imaging_features.items():
            for sequence, sequence_features in features.items():
                all_feature_names.update([f"{sequence}_{feat}" for feat in sequence_features.keys()])
        
        # If too many features, select most important ones
        if len(all_feature_names) > max_features_per_node:
            # For simplicity, we'll just take the first max_features
            # In practice, you'd use feature importance metrics
            all_feature_names = set(list(all_feature_names)[:max_features_per_node])
        
        # Add feature nodes
        for feature_name in all_feature_names:
            self.G.add_node(f"feature_{feature_name}", type='feature', name=feature_name)
        
        # Process patient imaging features
        batch_size = 10
        patient_ids = list(imaging_features.keys())
        
        for i in range(0, len(patient_ids), batch_size):
            batch_patient_ids = patient_ids[i:i+batch_size]
            
            for patient_id in batch_patient_ids:
                tumor_id = f"tumor_{patient_id}"
                if tumor_id not in self.G:
                    continue  # Skip if tumor node doesn't exist
                
                patient_features = imaging_features[patient_id]
                
                # Connect tumor to features with feature value as edge weight
                for sequence, sequence_features in patient_features.items():
                    for feature_name, feature_value in sequence_features.items():
                        combined_feature = f"{sequence}_{feature_name}"
                        if f"feature_{combined_feature}" in self.G:
                            self.G.add_edge(
                                tumor_id, 
                                f"feature_{combined_feature}", 
                                weight=float(feature_value),
                                relation='has_feature'
                            )
            
            # Track memory
            self.memory_tracker.log_memory(f"Added imaging batch {i//batch_size + 1}")
            
            # Compress if needed
            if self.memory_tracker.current_memory_mb() > self.memory_limit_mb * 0.8:
                self._compress_graph()
        
        return self
    
    def add_treatments(self, treatments_df: pd.DataFrame, batch_size: int = 20) -> 'GliomaKnowledgeGraph':
        """
        Add treatment data to the graph.
        
        Args:
            treatments_df: DataFrame with treatment information
            batch_size: Number of treatments to process in each batch
            
        Returns:
            Self for method chaining
        """
        self.memory_tracker.log_memory("Start adding treatments")
        
        # Process in batches
        for i in range(0, len(treatments_df), batch_size):
            batch = treatments_df.iloc[i:i+batch_size]
            
            for _, row in batch.iterrows():
                patient_id = row['patient_id']
                treatment_id = f"treatment_{row['treatment_id']}"
                tumor_id = f"tumor_{patient_id}"
                
                # Skip if tumor node doesn't exist
                if tumor_id not in self.G:
                    continue
                
                # Add treatment node
                treatment_attrs = {
                    'type': 'treatment',
                    'category': row.get('category', None),  # surgery, radiation, chemo
                    'name': row.get('treatment_name', None),
                    'dose': row.get('dose', None),
                    'duration': row.get('duration_days', None),
                    'start_day': row.get('start_day', None)
                }
                self.G.add_node(treatment_id, **treatment_attrs)
                
                # Connect tumor to treatment
                self.G.add_edge(tumor_id, treatment_id, relation='treated_with')
                
                # Add outcome if available
                if 'response' in row:
                    outcome_id = f"outcome_{patient_id}_{row['treatment_id']}"
                    outcome_attrs = {
                        'type': 'outcome',
                        'response': row.get('response', None),
                        'progression_free_days': row.get('progression_free_days', None),
                        'survival_days': row.get('survival_days', None)
                    }
                    self.G.add_node(outcome_id, **outcome_attrs)
                    
                    # Connect treatment to outcome
                    self.G.add_edge(treatment_id, outcome_id, relation='resulted_in')
            
            # Track memory
            self.memory_tracker.log_memory(f"Added treatment batch {i//batch_size + 1}")
            
            # Compress if needed
            if self.memory_tracker.current_memory_mb() > self.memory_limit_mb * 0.8:
                self._compress_graph()
        
        return self
    
    def add_similarity_edges(self, threshold: float = 0.7, max_edges_per_node: int = 5) -> 'GliomaKnowledgeGraph':
        """
        Add similarity edges between patients/tumors with similar features.
        
        Args:
            threshold: Similarity threshold for edge creation
            max_edges_per_node: Maximum number of similarity edges per node
            
        Returns:
            Self for method chaining
        """
        self.memory_tracker.log_memory("Start adding similarity edges")
        
        # Get all tumor nodes
        tumor_nodes = [node for node, attrs in self.G.nodes(data=True) 
                     if attrs.get('type') == 'tumor']
        
        # Process in small batches to save memory
        batch_size = 10
        
        for i in range(0, len(tumor_nodes), batch_size):
            batch_tumors = tumor_nodes[i:i+batch_size]
            
            for tumor1 in batch_tumors:
                # Get tumor features
                tumor1_features = self._get_tumor_features(tumor1)
                
                # Find similar tumors
                similarities = []
                
                for tumor2 in tumor_nodes:
                    if tumor1 == tumor2:
                        continue
                    
                    tumor2_features = self._get_tumor_features(tumor2)
                    similarity = self._calculate_similarity(tumor1_features, tumor2_features)
                    
                    if similarity > threshold:
                        similarities.append((tumor2, similarity))
                
                # Sort by similarity and take top max_edges_per_node
                similarities.sort(key=lambda x: x[1], reverse=True)
                top_similarities = similarities[:max_edges_per_node]
                
                # Add similarity edges
                for tumor2, similarity in top_similarities:
                    self.G.add_edge(tumor1, tumor2, 
                                   weight=float(similarity),
                                   relation='similar_to')
            
            # Track memory
            self.memory_tracker.log_memory(f"Added similarity batch {i//batch_size + 1}")
            
            # Compress if needed
            if self.memory_tracker.current_memory_mb() > self.memory_limit_mb * 0.8:
                self._compress_graph()
        
        return self
    
    def _get_tumor_features(self, tumor_id: str) -> Dict[str, float]:
        """
        Get all features connected to a tumor node.
        
        Args:
            tumor_id: Tumor node identifier
            
        Returns:
            Dictionary of feature values
        """
        features = {}
        
        # Get feature connections
        for _, feature_id, edge_data in self.G.out_edges(tumor_id, data=True):
            if edge_data.get('relation') == 'has_feature':
                feature_name = self.G.nodes[feature_id].get('name', '')
                feature_value = edge_data.get('weight', 0)
                features[feature_name] = feature_value
        
        # Get tumor attributes
        tumor_attrs = self.G.nodes[tumor_id]
        for key, value in tumor_attrs.items():
            if key != 'type' and key != 'name' and value is not None:
                features[f"tumor_{key}"] = value
        
        return features
    
    def _calculate_similarity(self, features1: Dict[str, float], features2: Dict[str, float]) -> float:
        """
        Calculate cosine similarity between feature dictionaries.
        
        Args:
            features1: First feature dictionary
            features2: Second feature dictionary
            
        Returns:
            Similarity score between 0 and 1
        """
        # Get common features
        common_features = set(features1.keys()) & set(features2.keys())
        
        if not common_features:
            return 0.0
        
        # Calculate cosine similarity
        vec1 = np.array([features1.get(f, 0) for f in common_features])
        vec2 = np.array([features2.get(f, 0) for f in common_features])
        
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(vec1, vec2) / (norm1 * norm2))
    
    def _compress_graph(self) -> 'GliomaKnowledgeGraph':
        """
        Apply compression techniques to reduce graph memory usage.
        
        Returns:
            Self for method chaining
        """
        # Convert node attribute dictionaries to use smaller data types
        for node, attrs in self.G.nodes(data=True):
            for key, value in attrs.items():
                if isinstance(value, float):
                    attrs[key] = np.float32(value)  # Use 32-bit floats instead of 64-bit
                elif isinstance(value, int):
                    attrs[key] = np.int32(value)  # Use 32-bit ints
        
        # Force garbage collection
        gc.collect()
        
        self.memory_tracker.log_memory("Graph compressed")
        return self
    
    def to_pytorch_geometric(self, node_features: Optional[Dict[str, Dict[str, List[Any]]]] = None) -> HeteroData:
        """
        Convert NetworkX graph to PyTorch Geometric format efficiently.
        
        Args:
            node_features: Optional precomputed node features
            
        Returns:
            PyTorch Geometric HeteroData object
        """
        self.memory_tracker.log_memory("Start conversion to PyG")
        
        # If node_features is None, extract from graph
        if node_features is None:
            node_features = {}
            node_types: Set[str] = set()
            
            # First pass: identify node types
            for _, data in self.G.nodes(data=True):
                node_type = data.get('type', 'unknown')
                node_types.add(node_type)
            
            # Initialize feature dictionaries for each node type
            for node_type in node_types:
                node_features[node_type] = {'nodes': [], 'features': []}
            
            # Second pass: collect nodes and features by type
            for node, data in self.G.nodes(data=True):
                node_type = data.get('type', 'unknown')
                node_features[node_type]['nodes'].append(node)
                
                # Extract numerical features
                features = []
                for key, value in data.items():
                    if key != 'type' and key != 'name':
                        if isinstance(value, (int, float)) and value is not None:
                            features.append(float(value))
                        elif value is None:
                            features.append(0.0)  # Replace None with 0
                        else:
                            # For categorical, we'd normally use one-hot, 
                            # but for simplicity we'll just use 0
                            features.append(0.0)
                
                node_features[node_type]['features'].append(features)
        
        # Create heterogeneous PyG data object
        data = HeteroData()
        
        # Add node features
        node_id_maps = {}
        
        for node_type, type_data in node_features.items():
            nodes = type_data['nodes']
            features = type_data['features']
            
            # Create mapping from original node IDs to indices
            node_id_maps[node_type] = {node: idx for idx, node in enumerate(nodes)}
            
            # Ensure all feature vectors have the same length
            if features:
                max_features = max(len(f) for f in features) if features else 0
                padded_features = []
                for f in features:
                    if len(f) < max_features:
                        padded_features.append(f + [0.0] * (max_features - len(f)))
                    else:
                        padded_features.append(f)
                
                # Add node features to PyG data
                data[node_type].x = torch.tensor(padded_features, dtype=torch.float)
            else:
                # If no features, create empty tensor with appropriate shape
                data[node_type].x = torch.zeros((len(nodes), 1), dtype=torch.float)
                
            # Store original node IDs
            data[node_type].original_ids = nodes
        
        # Add edges
        edge_types = {}
        
        # First pass: identify all edge types
        for u, v, edge_data in self.G.edges(data=True):
            u_type = self.G.nodes[u].get('type', 'unknown')
            v_type = self.G.nodes[v].get('type', 'unknown')
            relation = edge_data.get('relation', 'connected_to')
            
            edge_type = (u_type, relation, v_type)
            if edge_type not in edge_types:
                edge_types[edge_type] = {'source': [], 'target': [], 'attr': []}
        
        # Second pass: collect edges by type
        for u, v, edge_data in self.G.edges(data=True):
            u_type = self.G.nodes[u].get('type', 'unknown')
            v_type = self.G.nodes[v].get('type', 'unknown')
            relation = edge_data.get('relation', 'connected_to')
            
            # Skip if node types not in node_id_maps (filtered out)
            if u_type not in node_id_maps or v_type not in node_id_maps:
                continue
                
            # Skip if source or target node not in respective maps
            if u not in node_id_maps[u_type] or v not in node_id_maps[v_type]:
                continue
            
            edge_type = (u_type, relation, v_type)
            source_idx = node_id_maps[u_type][u]
            target_idx = node_id_maps[v_type][v]
            
            edge_types[edge_type]['source'].append(source_idx)
            edge_types[edge_type]['target'].append(target_idx)
            
            # Extract edge attributes
            edge_features = []
            for key, value in edge_data.items():
                if key != 'relation':
                    if isinstance(value, (int, float)) and value is not None:
                        edge_features.append(float(value))
                    else:
                        edge_features.append(0.0)
            
            if not edge_features:
                edge_features = [1.0]  # Default weight 1.0
                
            edge_types[edge_type]['attr'].append(edge_features)
        
        # Add edge indices and attributes to PyG data
        for edge_type, edges in edge_types.items():
            if not edges['source']:
                continue  # Skip if no edges of this type
                
            u_type, relation, v_type = edge_type
            
            # Add edge indices
            data[edge_type].edge_index = torch.tensor(
                [edges['source'], edges['target']], dtype=torch.long)
            
            # Ensure all edge attribute vectors have the same length
            edge_attrs = edges['attr']
            max_attrs = max(len(a) for a in edge_attrs) if edge_attrs else 0
            padded_attrs = []
            for a in edge_attrs:
                if len(a) < max_attrs:
                    padded_attrs.append(a + [0.0] * (max_attrs - len(a)))
                else:
                    padded_attrs.append(a)
            
            # Add edge attributes
            if padded_attrs:
                data[edge_type].edge_attr = torch.tensor(
                    padded_attrs, dtype=torch.float)
        
        self.memory_tracker.log_memory("PyG conversion complete")
        
        return data
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the graph.
        
        Returns:
            Dictionary of graph statistics
        """
        node_types = {}
        edge_types = {}
        
        # Count node types
        for node, attrs in self.G.nodes(data=True):
            node_type = attrs.get('type', 'unknown')
            node_types[node_type] = node_types.get(node_type, 0) + 1
        
        # Count edge types
        for u, v, attrs in self.G.edges(data=True):
            edge_type = attrs.get('relation', 'unknown')
            edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
        
        return {
            'total_nodes': self.G.number_of_nodes(),
            'total_edges': self.G.number_of_edges(),
            'node_types': node_types,
            'edge_types': edge_types,
            'density': nx.density(self.G),
            'memory_usage_mb': self.memory_tracker.current_memory_mb()
        }