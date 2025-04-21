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
                # Ensure patient_id is treated as string consistently
                patient_id_val = row['patient_id']
                patient_id = f"patient_{patient_id_val}"
                tumor_id = f"tumor_{patient_id_val}"

                # Add patient node with attributes
                patient_attrs = {
                    'type': 'patient',
                    'age': row.get('age', None),
                    'sex': row.get('sex', None),
                    'karnofsky': row.get('karnofsky_score', None)
                }
                # Convert attributes to appropriate types (float32, int32) for memory
                for key, value in patient_attrs.items():
                    if pd.isna(value): patient_attrs[key] = None # Handle NaN before conversion
                    elif isinstance(value, float): patient_attrs[key] = np.float32(value)
                    elif isinstance(value, int): patient_attrs[key] = np.int32(value)
                self.G.add_node(patient_id, **patient_attrs)

                # Add tumor node
                tumor_attrs = {
                    'type': 'tumor',
                    'grade': row.get('grade', None),
                    'histology': row.get('histology', None),
                    'location': row.get('location', None),
                    'idh_status': row.get('idh_mutation', None),
                    'mgmt_status': row.get('mgmt_methylation', None)
                }
                # Convert attributes
                for key, value in tumor_attrs.items():
                    if pd.isna(value): tumor_attrs[key] = None
                    elif isinstance(value, float): tumor_attrs[key] = np.float32(value)
                    elif isinstance(value, int): tumor_attrs[key] = np.int32(value)
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
            imaging_features: Dictionary mapping patient IDs (int) to imaging features
            max_features_per_node: Maximum number of features to add per node

        Returns:
            Self for method chaining
        """
        self.memory_tracker.log_memory("Start adding imaging features")

        # Add feature nodes first (shared across patients)
        all_feature_names: Set[str] = set()
        for patient_id_val, features in imaging_features.items():
            for sequence, sequence_features in features.items():
                all_feature_names.update([f"{sequence}_{feat}" for feat in sequence_features.keys()])

        # If too many features, select most important ones
        if len(all_feature_names) > max_features_per_node:
            # For simplicity, we'll just take the first max_features
            # In practice, you'd use feature importance metrics
            print(f"Warning: Limiting imaging features to {max_features_per_node} from {len(all_feature_names)}")
            all_feature_names = set(list(all_feature_names)[:max_features_per_node])

        # Add feature nodes
        for feature_name in all_feature_names:
            # Add feature node with type 'feature' and name attribute
            self.G.add_node(f"feature_{feature_name}", type='feature', name=feature_name)

        # Process patient imaging features
        batch_size = 10
        patient_ids = list(imaging_features.keys())

        for i in range(0, len(patient_ids), batch_size):
            batch_patient_ids = patient_ids[i:i+batch_size]

            for patient_id_val in batch_patient_ids:
                tumor_id = f"tumor_{patient_id_val}" # Use int patient_id
                if tumor_id not in self.G:
                    continue  # Skip if tumor node doesn't exist

                patient_features = imaging_features[patient_id_val]

                # Connect tumor to features with feature value as edge weight
                for sequence, sequence_features in patient_features.items():
                    for feature_name, feature_value in sequence_features.items():
                        combined_feature_name = f"{sequence}_{feature_name}"
                        feature_node_id = f"feature_{combined_feature_name}"
                        # Only add edge if the feature node exists (wasn't filtered out)
                        if feature_node_id in self.G:
                            self.G.add_edge(
                                tumor_id,
                                feature_node_id,
                                weight=np.float32(feature_value), # Use float32
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
                patient_id_val = row['patient_id']
                treatment_id_num = row['treatment_id']
                treatment_id = f"treatment_{treatment_id_num}" # Consistent ID format
                tumor_id = f"tumor_{patient_id_val}"

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
                # Convert attributes
                for key, value in treatment_attrs.items():
                    if pd.isna(value): treatment_attrs[key] = None
                    elif isinstance(value, float): treatment_attrs[key] = np.float32(value)
                    elif isinstance(value, int): treatment_attrs[key] = np.int32(value)
                self.G.add_node(treatment_id, **treatment_attrs)

                # Connect tumor to treatment
                self.G.add_edge(tumor_id, treatment_id, relation='treated_with')

                # Add outcome if available
                if 'response' in row and pd.notna(row['response']):
                    outcome_id = f"outcome_{patient_id_val}_{treatment_id_num}"
                    outcome_attrs = {
                        'type': 'outcome',
                        'response': row.get('response', None),
                        'progression_free_days': row.get('progression_free_days', None),
                        'survival_days': row.get('survival_days', None)
                    }
                    # Convert attributes
                    for key, value in outcome_attrs.items():
                        if pd.isna(value): outcome_attrs[key] = None
                        elif isinstance(value, float): outcome_attrs[key] = np.float32(value)
                        elif isinstance(value, int): outcome_attrs[key] = np.int32(value)
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

        # Precompute features for all tumors to avoid redundant lookups
        all_tumor_features = {tumor_id: self._get_tumor_features(tumor_id) for tumor_id in tumor_nodes}

        # Process in small batches to save memory
        batch_size = 10

        for i in range(0, len(tumor_nodes), batch_size):
            batch_tumors = tumor_nodes[i:i+batch_size]

            for tumor1 in batch_tumors:
                # Get precomputed features
                tumor1_features = all_tumor_features.get(tumor1, {})

                # Find similar tumors
                similarities = []

                for tumor2 in tumor_nodes:
                    if tumor1 == tumor2:
                        continue

                    tumor2_features = all_tumor_features.get(tumor2, {})
                    similarity = self._calculate_similarity(tumor1_features, tumor2_features)

                    if similarity > threshold:
                        similarities.append((tumor2, similarity))

                # Sort by similarity and take top max_edges_per_node
                similarities.sort(key=lambda x: x[1], reverse=True)
                top_similarities = similarities[:max_edges_per_node]

                # Add similarity edges
                for tumor2, similarity in top_similarities:
                    self.G.add_edge(tumor1, tumor2,
                                   weight=np.float32(similarity), # Use float32
                                   relation='similar_to')

            # Track memory
            self.memory_tracker.log_memory(f"Added similarity batch {i//batch_size + 1}")

            # Compress if needed
            if self.memory_tracker.current_memory_mb() > self.memory_limit_mb * 0.8:
                self._compress_graph()

        return self

    def _get_tumor_features(self, tumor_id: str) -> Dict[str, float]:
        """
        Get all numerical features connected to or attributes of a tumor node.

        Args:
            tumor_id: Tumor node identifier

        Returns:
            Dictionary of feature values
        """
        features = {}

        # Get feature connections
        if self.G.has_node(tumor_id):
            for _, feature_id, edge_data in self.G.out_edges(tumor_id, data=True):
                if edge_data.get('relation') == 'has_feature' and self.G.has_node(feature_id):
                    feature_name = self.G.nodes[feature_id].get('name', '')
                    feature_value = edge_data.get('weight', 0)
                    if feature_name and isinstance(feature_value, (int, float, np.number)) and not np.isnan(feature_value): # Ensure feature name is not empty and value is numeric
                        features[feature_name] = float(feature_value)

            # Get tumor attributes
            tumor_attrs = self.G.nodes[tumor_id]
            for key, value in tumor_attrs.items():
                # Only include numerical attributes directly, handle categorical later if needed
                if key != 'type' and key != 'name' and isinstance(value, (int, float, np.number)) and not np.isnan(value):
                    features[f"tumor_{key}"] = float(value)

        return features


    def _calculate_similarity(self, features1: Dict[str, Any], features2: Dict[str, Any]) -> float:
        """
        Calculate cosine similarity between feature dictionaries.

        Args:
            features1: First feature dictionary
            features2: Second feature dictionary

        Returns:
            Similarity score between 0 and 1
        """
        # Get common features that are numeric in BOTH dictionaries
        common_keys = set(features1.keys()) & set(features2.keys())
        numeric_features_in_common = []
        for key in common_keys:
            val1 = features1.get(key)
            val2 = features2.get(key)
            if isinstance(val1, (int, float, np.number)) and not np.isnan(val1) and \
               isinstance(val2, (int, float, np.number)) and not np.isnan(val2):
                numeric_features_in_common.append(key)

        if not numeric_features_in_common:
            return 0.0

        # --- Start Fix: Build vectors based *only* on common numeric features ---
        vec1_list = [float(features1[key]) for key in numeric_features_in_common]
        vec2_list = [float(features2[key]) for key in numeric_features_in_common]
        # --- End Fix ---

        # Calculate cosine similarity
        vec1 = np.array(vec1_list)
        vec2 = np.array(vec2_list)

        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        # Use np.float32 for calculation and result
        similarity = np.dot(vec1, vec2) / (norm1 * norm2)
        return np.float32(similarity)

    def _compress_graph(self) -> 'GliomaKnowledgeGraph':
        """
        Apply compression techniques to reduce graph memory usage.

        Returns:
            Self for method chaining
        """
        # Convert node attribute dictionaries to use smaller data types
        for node, attrs in self.G.nodes(data=True):
            for key, value in list(attrs.items()):  # Create a copy of items to avoid modification during iteration
                if isinstance(value, (float, np.float64)):
                    attrs[key] = np.float32(value)  # Use 32-bit floats
                elif isinstance(value, (int, np.int64)):
                    # Check range before converting to int32
                    if np.iinfo(np.int32).min <= value <= np.iinfo(np.int32).max:
                        attrs[key] = np.int32(value)
                    # else keep as int64 or handle appropriately

        # Convert edge attributes
        for u, v, key, attrs in self.G.edges(data=True, keys=True):
             for attr_key, value in list(attrs.items()):
                 if isinstance(value, (float, np.float64)):
                     attrs[attr_key] = np.float32(value)
                 elif isinstance(value, (int, np.int64)):
                     if np.iinfo(np.int32).min <= value <= np.iinfo(np.int32).max:
                         attrs[attr_key] = np.int32(value)


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

            # First pass: identify node types and collect all attribute keys per type
            node_attr_keys: Dict[str, Set[str]] = {}
            for node, data in self.G.nodes(data=True):
                node_type = data.get('type', 'unknown')
                node_types.add(node_type)
                if node_type not in node_attr_keys:
                    node_attr_keys[node_type] = set()
                # Collect keys of numerical attributes
                for key, value in data.items():
                    if key != 'type' and isinstance(value, (int, float, np.number)) and not np.isnan(value): # Exclude NaN
                         node_attr_keys[node_type].add(key)

            # Sort attribute keys for consistent feature order
            sorted_node_attr_keys = {nt: sorted(list(keys)) for nt, keys in node_attr_keys.items()}

            # Initialize feature dictionaries for each node type
            for node_type in node_types:
                node_features[node_type] = {'nodes': [], 'features': []}

            # Second pass: collect nodes and features by type using sorted keys
            for node, data in self.G.nodes(data=True):
                node_type = data.get('type', 'unknown')
                node_features[node_type]['nodes'].append(node)

                # Extract numerical features based on sorted keys
                features = []
                for key in sorted_node_attr_keys.get(node_type, []):
                    value = data.get(key)
                    # Impute missing/non-numeric/NaN with 0.0
                    if isinstance(value, (int, float, np.number)) and not np.isnan(value):
                        features.append(float(value))
                    else:
                        features.append(0.0)

                # Handle nodes with NO numerical features
                if not features and sorted_node_attr_keys.get(node_type):
                    # If keys existed but all values were missing/non-numeric/NaN
                    features = [0.0] * len(sorted_node_attr_keys[node_type])
                elif not features:
                    # If node type truly has no numerical attributes defined
                    features = [0.0] # Assign a single default feature (dim=1)

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

            # Ensure all feature vectors have the same length for this node type
            if features:
                # Find feature dim based on the first non-empty feature vector if exists
                first_valid_feature = next((f for f in features if f), None)
                feature_dim = len(first_valid_feature) if first_valid_feature else 1 # Default to 1 if all empty or no features
                # Ensure all vectors match this dimension, pad if necessary
                padded_features = []
                for f in features:
                    current_len = len(f)
                    if current_len < feature_dim:
                        padded_features.append(f + [0.0] * (feature_dim - current_len))
                    elif current_len > feature_dim:
                         padded_features.append(f[:feature_dim]) # Truncate if longer
                    else:
                        padded_features.append(f)

                # Add node features to PyG data
                data[node_type].x = torch.tensor(padded_features, dtype=torch.float)
            else:
                # If no features/nodes, create empty tensor with dim 1
                data[node_type].x = torch.empty((len(nodes), 1), dtype=torch.float) # Shape (num_nodes, 1)

            # Store original node IDs
            data[node_type].original_ids = nodes

        # Add edges
        edge_types_collected: Dict[Tuple[str, str, str], Dict[str, List]] = {}

        # First pass: identify all edge types and collect attribute keys
        edge_attr_keys: Dict[Tuple[str, str, str], Set[str]] = {}
        for u, v, edge_data in self.G.edges(data=True):
            u_type = self.G.nodes[u].get('type', 'unknown')
            v_type = self.G.nodes[v].get('type', 'unknown')
            relation = edge_data.get('relation', 'connected_to')
            edge_type = (u_type, relation, v_type)

            if edge_type not in edge_attr_keys:
                edge_attr_keys[edge_type] = set()
            # Collect keys of numerical attributes
            for key, value in edge_data.items():
                if key != 'relation' and isinstance(value, (int, float, np.number)) and not np.isnan(value): # Exclude NaN
                    edge_attr_keys[edge_type].add(key)

        # Sort attribute keys for consistent feature order
        sorted_edge_attr_keys = {et: sorted(list(keys)) for et, keys in edge_attr_keys.items()}


        # Second pass: collect edges by type using sorted keys
        for u, v, edge_data in self.G.edges(data=True):
            u_type = self.G.nodes[u].get('type', 'unknown')
            v_type = self.G.nodes[v].get('type', 'unknown')
            relation = edge_data.get('relation', 'connected_to')
            edge_type = (u_type, relation, v_type)

            # Skip if node types not in node_id_maps (e.g., 'unknown')
            if u_type not in node_id_maps or v_type not in node_id_maps:
                continue

            # Skip if source or target node not in respective maps (shouldn't happen if maps are correct)
            if u not in node_id_maps[u_type] or v not in node_id_maps[v_type]:
                continue

            if edge_type not in edge_types_collected:
                edge_types_collected[edge_type] = {'source': [], 'target': [], 'attr': []}

            source_idx = node_id_maps[u_type][u]
            target_idx = node_id_maps[v_type][v]

            edge_types_collected[edge_type]['source'].append(source_idx)
            edge_types_collected[edge_type]['target'].append(target_idx)

            # Extract edge attributes based on sorted keys
            edge_features = []
            for key in sorted_edge_attr_keys.get(edge_type, []):
                 value = edge_data.get(key)
                 # Impute missing/non-numeric/NaN with 0.0
                 if isinstance(value, (int, float, np.number)) and not np.isnan(value):
                     edge_features.append(float(value))
                 else:
                     edge_features.append(0.0)

            # Handle edges with NO numerical features
            if not edge_features and sorted_edge_attr_keys.get(edge_type):
                 edge_features = [0.0] * len(sorted_edge_attr_keys[edge_type])
            elif not edge_features:
                 edge_features = [1.0]  # Default weight 1.0 if no attributes defined

            edge_types_collected[edge_type]['attr'].append(edge_features)

        # Add edge indices and attributes to PyG data
        for edge_type, edges in edge_types_collected.items():
            if not edges['source']:
                continue  # Skip if no edges of this type

            # Add edge indices
            data[edge_type].edge_index = torch.tensor(
                [edges['source'], edges['target']], dtype=torch.long)

            # Ensure all edge attribute vectors have the same length for this edge type
            edge_attrs = edges['attr']
            if edge_attrs:
                # Find feature dim based on the first non-empty attribute vector if exists
                first_valid_attr = next((a for a in edge_attrs if a), None)
                attr_dim = len(first_valid_attr) if first_valid_attr else 1 # Default to 1
                padded_attrs = []
                for a in edge_attrs:
                    current_len = len(a)
                    if current_len < attr_dim:
                        padded_attrs.append(a + [0.0] * (attr_dim - current_len))
                    elif current_len > attr_dim:
                        padded_attrs.append(a[:attr_dim]) # Truncate
                    else:
                        padded_attrs.append(a)

                # Add edge attributes
                data[edge_type].edge_attr = torch.tensor(
                    padded_attrs, dtype=torch.float)
            else:
                 # If no attributes, create empty tensor with dim 1
                 data[edge_type].edge_attr = torch.empty((len(edges['source']), 1), dtype=torch.float) # Shape (num_edges, 1)


        self.memory_tracker.log_memory("PyG conversion complete")

        # Ensure all node types exist even if empty
        all_node_types_in_graph = set(d.get('type', 'unknown') for _, d in self.G.nodes(data=True))
        for node_type in all_node_types_in_graph:
            if node_type != 'unknown' and node_type not in data.node_types:
                print(f"Warning: Adding empty node type '{node_type}' to PyG data.")
                data[node_type].x = torch.empty((0, 1), dtype=torch.float) # Shape (0, 1)
                data[node_type].original_ids = []
            # --- Start Fix: Ensure x exists even for empty node types ---
            elif node_type != 'unknown' and not hasattr(data[node_type], 'x'):
                 print(f"Warning: Adding empty 'x' tensor for node type '{node_type}' in PyG data.")
                 data[node_type].x = torch.empty((0, 1), dtype=torch.float) # Shape (0, 1)
            # --- End Fix ---

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

        # Count edge types (relations)
        for u, v, attrs in self.G.edges(data=True):
            edge_type = attrs.get('relation', 'unknown')
            edge_types[edge_type] = edge_types.get(edge_type, 0) + 1

        # Calculate density safely
        num_nodes = self.G.number_of_nodes()
        density = nx.density(self.G) if num_nodes > 1 else 0.0

        return {
            'total_nodes': num_nodes,
            'total_edges': self.G.number_of_edges(),
            'node_types': node_types,
            'edge_types': edge_types,
            'density': density,
            'memory_usage_mb': self.memory_tracker.current_memory_mb()
        }
