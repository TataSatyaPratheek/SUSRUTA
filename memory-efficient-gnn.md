import os
import gc
import psutil
import numpy as np
import pandas as pd
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv
from torch_geometric.data import Data, HeteroData
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import from_networkx
import SimpleITK as sitk
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


class MemoryTracker:
    """Utility class to track memory usage during processing."""
    
    def __init__(self, threshold_mb=7000):
        self.process = psutil.Process()
        self.threshold_mb = threshold_mb
        self.log = []
    
    def current_memory_mb(self):
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / (1024 * 1024)
    
    def log_memory(self, operation):
        """Log memory usage for an operation."""
        memory_mb = self.current_memory_mb()
        self.log.append((operation, memory_mb))
        if memory_mb > self.threshold_mb:
            print(f"WARNING: {operation} used {memory_mb:.2f}MB, exceeding threshold of {self.threshold_mb}MB")
        return memory_mb
    
    def plot_memory_usage(self):
        """Plot memory usage over time."""
        operations, memory = zip(*self.log)
        plt.figure(figsize=(10, 6))
        plt.plot(memory)
        plt.xticks(range(len(operations)), operations, rotation=90)
        plt.xlabel('Operation')
        plt.ylabel('Memory Usage (MB)')
        plt.title('Memory Usage During Processing')
        plt.tight_layout()
        return plt


class EfficientMRIProcessor:
    """Memory-efficient MRI processing for large volumes."""
    
    def __init__(self, memory_limit_mb=2000):
        self.memory_limit_mb = memory_limit_mb
        self.memory_tracker = MemoryTracker()
    
    def load_nifti_metadata(self, file_path):
        """Load NIfTI file metadata without loading full volume into memory."""
        reader = sitk.ImageFileReader()
        reader.SetFileName(file_path)
        reader.ReadImageInformation()
        return {
            'size': reader.GetSize(),
            'spacing': reader.GetSpacing(),
            'origin': reader.GetOrigin(),
            'direction': reader.GetDirection()
        }
    
    def compute_bounding_box(self, mask_path):
        """Compute tumor bounding box from mask without loading full volume."""
        mask = sitk.ReadImage(mask_path)
        
        # Convert to binary mask if needed
        if mask.GetPixelID() != sitk.sitkUInt8:
            binary_mask = mask > 0
        else:
            binary_mask = mask
        
        # Use label statistics to get bounding box efficiently
        label_stats = sitk.LabelStatisticsImageFilter()
        label_stats.Execute(binary_mask, binary_mask)
        
        bbox = label_stats.GetBoundingBox(1)  # Label 1 bounding box
        
        # Convert to min/max coordinates format
        bbox_min = [bbox[0], bbox[2], bbox[4]]
        bbox_max = [bbox[1], bbox[3], bbox[5]]
        
        return bbox_min, bbox_max
    
    def extract_roi_features(self, img_path, mask_path):
        """Extract features only from tumor region to save memory."""
        # Track memory
        self.memory_tracker.log_memory("Starting ROI extraction")
        
        # Get bounding box of tumor with margin
        bbox_min, bbox_max = self.compute_bounding_box(mask_path)
        margin = 5  # Add 5 voxel margin around tumor
        
        # Get image metadata
        img_info = self.load_nifti_metadata(img_path)
        size = img_info['size']
        
        # Add margin and clamp to image bounds
        bbox_min = [max(0, x - margin) for x in bbox_min]
        bbox_max = [min(s-1, x + margin) for x, s in zip(bbox_max, size)]
        
        # Calculate ROI size
        roi_size = [bbox_max[i] - bbox_min[i] + 1 for i in range(3)]
        
        # Estimate memory requirements
        voxel_bytes = 4  # 32-bit float
        roi_memory_mb = np.prod(roi_size) * voxel_bytes / (1024 * 1024)
        
        self.memory_tracker.log_memory(f"ROI calculation (size: {roi_size}, memory: {roi_memory_mb:.2f}MB)")
        
        # Load ROI using SimpleITK's efficient region extraction
        img = sitk.ReadImage(img_path)
        mask = sitk.ReadImage(mask_path)
        
        if roi_memory_mb < self.memory_limit_mb * 0.5:
            # Can process ROI at once
            roi_img = sitk.RegionOfInterest(img, roi_size, bbox_min)
            roi_mask = sitk.RegionOfInterest(mask, roi_size, bbox_min)
            
            # Extract features
            features = self._compute_roi_features(roi_img, roi_mask)
            self.memory_tracker.log_memory("Feature extraction complete")
            
            # Clean up
            del roi_img, roi_mask
            gc.collect()
        else:
            # Process in chunks along Z-axis
            chunk_features = []
            chunk_count = max(1, int(np.ceil(roi_memory_mb / (self.memory_limit_mb * 0.3))))
            chunk_size_z = int(np.ceil(roi_size[2] / chunk_count))
            
            for chunk_idx in range(chunk_count):
                z_start = bbox_min[2] + chunk_idx * chunk_size_z
                z_end = min(bbox_min[2] + (chunk_idx + 1) * chunk_size_z - 1, bbox_max[2])
                
                chunk_bbox_min = [bbox_min[0], bbox_min[1], z_start]
                chunk_roi_size = [roi_size[0], roi_size[1], z_end - z_start + 1]
                
                # Extract chunk
                chunk_img = sitk.RegionOfInterest(img, chunk_roi_size, chunk_bbox_min)
                chunk_mask = sitk.RegionOfInterest(mask, chunk_roi_size, chunk_bbox_min)
                
                # Compute features for chunk
                chunk_features.append(self._compute_roi_features(chunk_img, chunk_mask))
                self.memory_tracker.log_memory(f"Chunk {chunk_idx+1}/{chunk_count} processed")
                
                # Explicitly clean up
                del chunk_img, chunk_mask
                gc.collect()
            
            # Combine chunk features
            features = self._combine_chunk_features(chunk_features)
            self.memory_tracker.log_memory("All chunks combined")
        
        # Final cleanup
        del img, mask
        gc.collect()
        
        return features
    
    def _compute_roi_features(self, img, mask):
        """Compute radiomics features from an ROI."""
        # Convert to numpy array for processing - only for masked region
        # This is memory efficient since we're only working with the ROI
        img_array = sitk.GetArrayFromImage(img)
        mask_array = sitk.GetArrayFromImage(mask)
        
        # Calculate first-order statistics on masked region
        masked_img = img_array[mask_array > 0]
        
        if len(masked_img) == 0:
            # No tumor voxels in this region
            return {
                'mean': 0,
                'std': 0,
                'min': 0,
                'max': 0,
                'p25': 0,
                'p50': 0,
                'p75': 0,
                'volume_voxels': 0
            }
        
        features = {
            'mean': np.mean(masked_img),
            'std': np.std(masked_img),
            'min': np.min(masked_img),
            'max': np.max(masked_img),
            'p25': np.percentile(masked_img, 25),
            'p50': np.percentile(masked_img, 50),
            'p75': np.percentile(masked_img, 75),
            'volume_voxels': len(masked_img)
        }
        
        # More advanced features could be added here
        # For simplicity and memory efficiency, we're keeping it basic
        
        return features
    
    def _combine_chunk_features(self, chunk_features):
        """Combine features from multiple chunks."""
        # For statistics like mean, we need a weighted average based on volume
        total_volume = sum(f['volume_voxels'] for f in chunk_features)
        
        if total_volume == 0:
            return chunk_features[0]  # No tumor voxels found
        
        # Weighted combination of statistics
        combined = {}
        
        # Simple statistics (min, max) can be combined directly
        combined['min'] = min(f['min'] for f in chunk_features)
        combined['max'] = max(f['max'] for f in chunk_features)
        combined['volume_voxels'] = total_volume
        
        # Weighted statistics
        for stat in ['mean', 'p25', 'p50', 'p75']:
            combined[stat] = sum(f[stat] * f['volume_voxels'] for f in chunk_features) / total_volume
        
        # For standard deviation, we need to combine variances and then sqrt
        # This is an approximation that works reasonably well for large volumes
        combined['std'] = np.sqrt(
            sum(f['std']**2 * f['volume_voxels'] for f in chunk_features) / total_volume
        )
        
        return combined
    
    def extract_features_for_patient(self, patient_id, data_dir, timepoint=1):
        """Extract features from all MRI sequences for a patient timepoint."""
        timepoint_dir = os.path.join(data_dir, f'PatientID_{patient_id:04d}', f'Timepoint_{timepoint}')
        
        if not os.path.exists(timepoint_dir):
            raise ValueError(f"Data not found for patient {patient_id}, timepoint {timepoint}")
        
        # Get file paths for all sequences
        sequence_files = {}
        for sequence in ['t1c', 't1n', 't2f', 't2w']:
            matches = [f for f in os.listdir(timepoint_dir) if sequence in f]
            if matches:
                sequence_files[sequence] = os.path.join(timepoint_dir, matches[0])
        
        # Get tumor mask path
        mask_path = os.path.join(timepoint_dir, [f for f in os.listdir(timepoint_dir) if 'tumorMask' in f][0])
        
        # Extract features for each sequence
        features = {}
        for sequence, file_path in sequence_files.items():
            print(f"Processing {sequence} for patient {patient_id}, timepoint {timepoint}")
            sequence_features = self.extract_roi_features(file_path, mask_path)
            features[sequence] = sequence_features
        
        # Get tumor volume and shape features from mask
        features['tumor'] = self._extract_tumor_features(mask_path)
        
        # Flatten the feature dictionary for easier use
        flat_features = {}
        for seq, feat_dict in features.items():
            for feat_name, feat_value in feat_dict.items():
                flat_features[f'{seq}_{feat_name}'] = feat_value
        
        return flat_features
    
    def _extract_tumor_features(self, mask_path):
        """Extract shape and volume features from tumor mask."""
        mask = sitk.ReadImage(mask_path)
        
        # Use label shape statistics for efficient feature extraction
        shape_stats = sitk.LabelShapeStatisticsImageFilter()
        shape_stats.Execute(mask)
        
        if not shape_stats.HasLabel(1):
            return {
                'volume_mm3': 0,
                'surface_area': 0,
                'elongation': 0,
                'roundness': 0
            }
        
        # Get physical space measurements
        volume = shape_stats.GetPhysicalSize(1)
        surface = shape_stats.GetPerimeter(1)  # 2D perimeter, approximation
        elongation = shape_stats.GetElongation(1)
        roundness = shape_stats.GetRoundness(1)
        
        features = {
            'volume_mm3': volume,
            'surface_area': surface,
            'elongation': elongation,
            'roundness': roundness
        }
        
        return features


class GliomaKnowledgeGraph:
    """Construct a knowledge graph for glioma treatment data with memory efficiency."""
    
    def __init__(self, memory_limit_mb=3000):
        self.memory_limit_mb = memory_limit_mb
        self.memory_tracker = MemoryTracker(threshold_mb=memory_limit_mb)
        self.G = nx.MultiDiGraph()
    
    def add_clinical_data(self, clinical_df, batch_size=20):
        """Add clinical data to the graph in batches to control memory usage."""
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
    
    def add_imaging_features(self, imaging_features, max_features_per_node=20):
        """Add imaging features to the graph efficiently."""
        self.memory_tracker.log_memory("Start adding imaging features")
        
        # Add feature nodes first (shared across patients)
        all_feature_names = set()
        for patient_id, features in imaging_features.items():
            all_feature_names.update(features.keys())
        
        # If too many features, select most important ones
        if len(all_feature_names) > max_features_per_node:
            # For simplicity, we'll just take the first max_features
            # In practice, you'd use feature importance metrics
            all_feature_names = list(all_feature_names)[:max_features_per_node]
        
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
                for feature_name in all_feature_names:
                    if feature_name in patient_features:
                        feature_value = patient_features[feature_name]
                        feature_id = f"feature_{feature_name}"
                        
                        self.G.add_edge(
                            tumor_id, 
                            feature_id, 
                            weight=float(feature_value),
                            relation='has_feature'
                        )
            
            # Track memory
            self.memory_tracker.log_memory(f"Added imaging batch {i//batch_size + 1}")
            
            # Compress if needed
            if self.memory_tracker.current_memory_mb() > self.memory_limit_mb * 0.8:
                self._compress_graph()
        
        return self
    
    def add_treatments(self, treatments_df, batch_size=20):
        """Add treatment data to the graph."""
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
    
    def add_similarity_edges(self, threshold=0.7, max_edges_per_node=5):
        """Add similarity edges between patients/tumors with similar features."""
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
    
    def _get_tumor_features(self, tumor_id):
        """Get all features connected to a tumor node."""
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
    
    def _calculate_similarity(self, features1, features2):
        """Calculate cosine similarity between feature dictionaries."""
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
        
        return np.dot(vec1, vec2) / (norm1 * norm2)
    
    def _compress_graph(self):
        """Apply compression techniques to reduce graph memory usage."""
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
    
    def to_pytorch_geometric(self, node_features=None):
        """Convert NetworkX graph to PyTorch Geometric format efficiently."""
        self.memory_tracker.log_memory("Start conversion to PyG")
        
        # If node_features is None, extract from graph
        if node_features is None:
            node_features = {}
            node_types = set()
            
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
            max_features = max(len(f) for f in features) if features else 0
            padded_features = []
            for f in features:
                if len(f) < max_features:
                    padded_features.append(f + [0.0] * (max_features - len(f)))
                else:
                    padded_features.append(f)
            
            # Add node features to PyG data
            if padded_features:
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
            data[(u_type, relation, v_type)].edge_index = torch.tensor(
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
                data[(u_type, relation, v_type)].edge_attr = torch.tensor(
                    padded_attrs, dtype=torch.float)
        
        self.memory_tracker.log_memory("PyG conversion complete")
        
        return data


class GliomaGNN(torch.nn.Module):
    """Memory-efficient GNN for glioma treatment outcome prediction."""
    
    def __init__(self, node_feature_dims, edge_feature_dims=None, 
                 hidden_channels=32, dropout=0.3):
        super().__init__()
        
        self.node_feature_dims = node_feature_dims
        self.hidden_channels = hidden_channels
        
        # Node type-specific encoders
        self.node_encoders = nn.ModuleDict()
        for node_type, dim in node_feature_dims.items():
            self.node_encoders[node_type] = nn.Linear(dim, hidden_channels)
        
        # GNN layers - use GAT for better inductive bias
        self.conv1 = nn.ModuleDict()
        self.conv2 = nn.ModuleDict()
        
        # Create convolutions for each edge type
        for edge_type in edge_feature_dims.keys():
            # Format: (src_type, edge_type, dst_type)
            src_type, _, dst_type = edge_type
            
            # First convolution
            self.conv1[str(edge_type)] = GATConv(
                (hidden_channels, hidden_channels), 
                hidden_channels, 
                heads=4,
                dropout=dropout,
                add_self_loops=False
            )
            
            # Second convolution
            self.conv2[str(edge_type)] = GATConv(
                (hidden_channels * 4, hidden_channels * 4),  # 4 attention heads
                hidden_channels,
                heads=1,
                dropout=dropout,
                add_self_loops=False
            )
        
        # Prediction heads for different tasks
        self.response_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
            nn.Sigmoid()
        )
        
        self.survival_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
            nn.ReLU()  # Survival time is non-negative
        )
        
        # Uncertainty estimation head
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
            nn.Softplus()  # For positive uncertainty values
        )
    
    def forward(self, x_dict, edge_indices_dict):
        """
        Forward pass for memory-efficient heterogeneous GNN.
        
        Args:
            x_dict: Dictionary of node features by type
            edge_indices_dict: Dictionary of edge indices by type
            
        Returns:
            Dictionary of predictions
        """
        # Encode node features by type
        h_dict = {}
        for node_type, x in x_dict.items():
            if node_type in self.node_encoders:
                h_dict[node_type] = self.node_encoders[node_type](x)
            else:
                # Handle unexpected node types
                dim = x.size(1)
                self.node_encoders[node_type] = nn.Linear(dim, self.hidden_channels).to(x.device)
                h_dict[node_type] = self.node_encoders[node_type](x)
        
        # First message-passing layer with residual connection
        h_dict_1 = {node_type: h for node_type, h in h_dict.items()}
        
        for edge_type, edge_index in edge_indices_dict.items():
            src_type, _, dst_type = edge_type
            
            # Skip if edge type not in convolution dictionary
            if str(edge_type) not in self.conv1:
                continue
                
            # Apply convolution
            h_dict_1[dst_type] = h_dict_1.get(dst_type, 0) + self.conv1[str(edge_type)](
                (h_dict[src_type], h_dict[dst_type]), 
                edge_index
            )
        
        # Apply activation and dropout
        h_dict_1 = {node_type: F.relu(h) for node_type, h in h_dict_1.items()}
        h_dict_1 = {node_type: F.dropout(h, p=0.3, training=self.training) 
                   for node_type, h in h_dict_1.items()}
        
        # Second message-passing layer
        h_dict_2 = {node_type: h for node_type, h in h_dict_1.items()}
        
        for edge_type, edge_index in edge_indices_dict.items():
            src_type, _, dst_type = edge_type
            
            # Skip if edge type not in convolution dictionary
            if str(edge_type) not in self.conv2:
                continue
                
            # Apply convolution
            h_dict_2[dst_type] = h_dict_2.get(dst_type, 0) + self.conv2[str(edge_type)](
                (h_dict_1[src_type], h_dict_1[dst_type]), 
                edge_index
            )
        
        # Apply final activation
        h_dict_2 = {node_type: F.relu(h) for node_type, h in h_dict_2.items()}
        
        # Create predictions
        predictions = {}
        
        # Predict treatment outcomes if treatment nodes exist
        if 'treatment' in h_dict_2:
            treatment_embeddings = h_dict_2['treatment']
            
            # Response prediction (binary classification)
            response_pred = self.response_head(treatment_embeddings)
            predictions['response'] = response_pred
            
            # Survival prediction (regression)
            survival_pred = self.survival_head(treatment_embeddings)
            predictions['survival'] = survival_pred
            
            # Uncertainty estimation
            uncertainty = self.uncertainty_head(treatment_embeddings)
            predictions['uncertainty'] = uncertainty
        
        return predictions, h_dict_2


class TreatmentSimulator:
    """Simulate counterfactual treatment outcomes based on graph structure."""
    
    def __init__(self, model, graph, device='cpu'):
        self.model = model
        self.graph = graph
        self.device = device
        self.model.eval()  # Set model to evaluation mode
    
    def simulate_treatments(self, patient_id, tumor_id, treatment_options, data):
        """
        Simulate outcomes for different treatment options.
        
        Args:
            patient_id: Patient node ID
            tumor_id: Tumor node ID
            treatment_options: List of treatment configurations to simulate
            data: PyTorch Geometric data object
            
        Returns:
            Dictionary of predicted outcomes for each treatment option
        """
        results = {}
        
        # Find node indices
        if hasattr(data['patient'], 'original_ids'):
            patient_idx = data['patient'].original_ids.index(patient_id)
            tumor_idx = data['tumor'].original_ids.index(tumor_id)
        else:
            # If original_ids not available, assume indices match
            patient_idx = int(patient_id.split('_')[1])
            tumor_idx = int(tumor_id.split('_')[1])
        
        # Get patient and tumor embeddings
        with torch.no_grad():
            # Forward pass to get node embeddings
            _, node_embeddings = self.model(
                {k: v.x.to(self.device) for k, v in data.items()},
                {k: v.edge_index.to(self.device) for k, v in data.edge_types().items()}
            )
        
        # Get tumor embedding
        tumor_embedding = node_embeddings['tumor'][tumor_idx]
        
        # Simulate each treatment
        for treatment_idx, treatment_config in enumerate(treatment_options):
            # Create synthetic treatment node with configuration
            treatment_features = self._encode_treatment(treatment_config)
            
            # Add this treatment to the graph
            treatment_id = f"treatment_sim_{treatment_idx}"
            
            # Connect tumor to treatment
            # This is a simplified version; in practice, you'd modify the PyG data object
            # Here we're creating a synthetic treatment embedding
            
            # Average treatment embeddings of similar type
            similar_treatments = self._find_similar_treatments(treatment_config, data)
            if similar_treatments:
                # Use average embedding of similar treatments
                treatment_embedding = torch.stack(similar_treatments).mean(dim=0)
            else:
                # Create new embedding from treatment encoder
                treatment_embedding = self.model.node_encoders['treatment'](
                    torch.tensor(treatment_features, dtype=torch.float).to(self.device)
                )
            
            # Predict outcomes using the model's prediction heads
            response_pred = self.model.response_head(treatment_embedding)
            survival_pred = self.model.survival_head(treatment_embedding)
            uncertainty = self.model.uncertainty_head(treatment_embedding)
            
            # Store results
            results[f"option_{treatment_idx}"] = {
                'config': treatment_config,
                'response_prob': response_pred.item(),
                'survival_days': survival_pred.item(),
                'uncertainty': uncertainty.item()
            }
        
        return results
    
    def _encode_treatment(self, treatment_config):
        """Encode treatment configuration into features."""
        # This is a simplified encoding, adapt to your feature space
        features = []
        
        # Encode treatment category (one-hot)
        categories = ['surgery', 'radiation', 'chemotherapy', 'combined']
        category_vec = [0] * len(categories)
        if treatment_config['category'] in categories:
            category_vec[categories.index(treatment_config['category'])] = 1
        features.extend(category_vec)
        
        # Encode numerical features
        for key in ['dose', 'duration', 'intensity']:
            if key in treatment_config:
                features.append(float(treatment_config[key]))
            else:
                features.append(0.0)
        
        return features
    
    def _find_similar_treatments(self, treatment_config, data):
        """Find embeddings of similar treatments in the graph."""
        similar_treatments = []
        
        # Get treatment category
        category = treatment_config['category']
        
        # This is a simplified version; in practice, you'd query the graph structure
        # For demonstration, we'll just look for treatments of the same category
        # that are close in feature space
        
        # This implementation depends on your data structure
        # Here's a placeholder:
        if 'treatment' in data:
            if hasattr(data['treatment'], 'original_ids'):
                for idx, treatment_id in enumerate(data['treatment'].original_ids):
                    if treatment_id in self.graph.nodes():
                        treatment_attrs = self.graph.nodes[treatment_id]
                        if treatment_attrs.get('category') == category:
                            # Get the embedding from node_embeddings
                            with torch.no_grad():
                                # Forward pass to get node embeddings
                                _, node_embeddings = self.model(
                                    {k: v.x.to(self.device) for k, v in data.items()},
                                    {k: v.edge_index.to(self.device) for k, v in data.edge_types().items()}
                                )
                            
                            similar_treatments.append(node_embeddings['treatment'][idx])
        
        return similar_treatments


class ExplainableGliomaTreatment:
    """Explainable treatment recommendations with confidence estimates."""
    
    def __init__(self, model, graph_data, original_graph=None, device='cpu'):
        self.model = model
        self.data = graph_data
        self.original_graph = original_graph  # NetworkX graph for attribute lookup
        self.device = device
        self.model.eval()  # Set to evaluation mode
    
    def explain_treatment_prediction(self, treatment_idx, k=5):
        """
        Explain treatment outcome prediction.
        
        Args:
            treatment_idx: Index of treatment node to explain
            k: Number of top features to include in explanation
            
        Returns:
            Dictionary of explanations and visualizations
        """
        # Get treatment node features
        x = self.data['treatment'].x[treatment_idx].clone().to(self.device)
        x.requires_grad_(True)
        
        # Forward pass with gradients
        self.model.zero_grad()
        output_dict, _ = self.model(
            {k: v.x.to(self.device) if k != 'treatment' else torch.stack([x])
             for k, v in self.data.items()},
            {k: v.edge_index.to(self.device) for k, v in self.data.edge_types().items()}
        )
        
        # Extract predictions
        response_pred = output_dict['response'][0]
        survival_pred = output_dict['survival'][0]
        uncertainty = output_dict['uncertainty'][0]
        
        # Compute gradients for response prediction
        response_pred.backward(retain_graph=True)
        response_gradients = x.grad.clone()
        
        # Zero gradients and compute for survival prediction
        self.model.zero_grad()
        x.grad = None
        survival_pred.backward()
        survival_gradients = x.grad.clone()
        
        # Feature importance scores (absolute gradient values)
        response_importance = torch.abs(response_gradients).cpu().numpy()
        survival_importance = torch.abs(survival_gradients).cpu().numpy()
        
        # Get original treatment node ID
        treatment_id = self.data['treatment'].original_ids[treatment_idx] if hasattr(
            self.data['treatment'], 'original_ids') else f"treatment_{treatment_idx}"
        
        # Get treatment attributes and connected patient/tumor
        treatment_attrs = {}
        patient_id = None
        tumor_id = None
        
        if self.original_graph is not None and treatment_id in self.original_graph:
            treatment_attrs = dict(self.original_graph.nodes[treatment_id])
            
            # Find connected tumor
            for src, dst in self.original_graph.in_edges(treatment_id):
                if 'tumor' in src:
                    tumor_id = src
                    break
            
            # Find connected patient
            if tumor_id:
                for src, dst in self.original_graph.in_edges(tumor_id):
                    if 'patient' in src:
                        patient_id = src
                        break
        
        # Create feature explanation
        feature_names = ['category_surgery', 'category_radiation', 'category_chemo', 
                        'category_combined', 'dose', 'duration', 'intensity']
        
        # Truncate to actual feature length
        feature_names = feature_names[:len(response_importance)]
        
        # Sort features by importance
        response_indices = np.argsort(-response_importance)[:k]
        survival_indices = np.argsort(-survival_importance)[:k]
        
        response_top_features = [(feature_names[i], float(response_importance[i])) 
                               for i in response_indices]
        survival_top_features = [(feature_names[i], float(survival_importance[i])) 
                                for i in survival_indices]
        
        # Create explanation dictionary
        explanation = {
            'treatment_id': treatment_id,
            'treatment_attributes': treatment_attrs,
            'patient_id': patient_id,
            'tumor_id': tumor_id,
            'predictions': {
                'response_probability': float(response_pred.item()),
                'survival_days': float(survival_pred.item()),
                'uncertainty': float(uncertainty.item())
            },
            'feature_importance': {
                'response': response_top_features,
                'survival': survival_top_features
            }
        }
        
        return explanation


def load_synthetic_data():
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
        patient_features['t1c_mean'] = np.random.uniform(100, 200)
        patient_features['t1c_std'] = np.random.uniform(10, 50)
        patient_features['t1c_max'] = patient_features['t1c_mean'] + 2*patient_features['t1c_std']
        
        # T2 features
        patient_features['t2w_mean'] = np.random.uniform(150, 250)
        patient_features['t2w_std'] = np.random.uniform(20, 60)
        patient_features['t2w_max'] = patient_features['t2w_mean'] + 2*patient_features['t2w_std']
        
        # FLAIR features
        patient_features['t2f_mean'] = np.random.uniform(120, 220)
        patient_features['t2f_std'] = np.random.uniform(15, 55)
        patient_features['t2f_max'] = patient_features['t2f_mean'] + 2*patient_features['t2f_std']
        
        # Tumor features
        patient_features['tumor_volume_mm3'] = np.random.uniform(1000, 30000)
        patient_features['tumor_surface_area'] = np.random.uniform(500, 5000)
        patient_features['tumor_elongation'] = np.random.uniform(0.2, 0.8)
        patient_features['tumor_roundness'] = np.random.uniform(0.3, 0.9)
        
        imaging_features[patient_id] = patient_features
    
    return clinical_data, treatments_df, imaging_features


def main():
    """Run the full pipeline with memory-efficient implementation."""
    print("Starting Glioma Treatment Prediction System")
    memory_tracker = MemoryTracker()
    
    # Load data (synthetic for demo)
    print("\nLoading data...")
    clinical_data, treatments_df, imaging_features = load_synthetic_data()
    memory_tracker.log_memory("Data loaded")
    
    # Build knowledge graph
    print("\nBuilding knowledge graph...")
    kg_builder = GliomaKnowledgeGraph(memory_limit_mb=3000)
    kg_builder.add_clinical_data(clinical_data)
    kg_builder.add_imaging_features(imaging_features)
    kg_builder.add_treatments(treatments_df)
    kg_builder.add_similarity_edges()
    
    G = kg_builder.G
    memory_tracker.log_memory("Knowledge graph built")
    
    print(f"Graph statistics: {len(G.nodes())} nodes, {len(G.edges())} edges")
    
    # Convert to PyTorch Geometric format
    print("\nConverting to PyTorch Geometric format...")
    pyg_data = kg_builder.to_pytorch_geometric()
    memory_tracker.log_memory("Converted to PyG")
    
    # Define model
    print("\nInitializing GNN model...")
    node_feature_dims = {node_type: data.x.size(1) for node_type, data in pyg_data.items()}
    
    # Get edge types
    edge_feature_dims = {}
    for edge_type in pyg_data.edge_types():
        # Extract node types from edge type
        src_type, edge_name, dst_type = edge_type
        edge_feature_dims[edge_type] = 1  # Simplified for demo
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GliomaGNN(node_feature_dims, edge_feature_dims, hidden_channels=16)
    model = model.to(device)
    memory_tracker.log_memory("Model initialized")
    
    # Train model (simplified for demo)
    print("\nTraining model (simplified demo)...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Move data to device
    for key in pyg_data.keys():
        pyg_data[key].x = pyg_data[key].x.to(device)
    
    for edge_type in pyg_data.edge_types():
        pyg_data[edge_type].edge_index = pyg_data[edge_type].edge_index.to(device)
    
    model.train()
    for epoch in range(5):  # Just a few epochs for demo
        optimizer.zero_grad()
        
        # Forward pass
        predictions, _ = model(
            {k: v.x for k, v in pyg_data.items()},
            {k: v.edge_index for k, v in pyg_data.edge_types().items()}
        )
        
        # Create dummy targets for demo (replace with real targets in production)
        if 'treatment' in pyg_data:
            num_treatments = pyg_data['treatment'].x.size(0)
            response_target = torch.rand(num_treatments, 1).to(device)
            survival_target = torch.rand(num_treatments, 1).to(device) * 1000
            
            # Compute loss (simplified)
            response_loss = F.binary_cross_entropy(predictions['response'], response_target)
            survival_loss = F.mse_loss(predictions['survival'], survival_target)
            
            # Combined loss
            loss = response_loss + 0.01 * survival_loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            print(f"Epoch {epoch+1}/5, Loss: {loss.item():.4f}")
    
    memory_tracker.log_memory("Model trained")
    
    # Simulate treatments
    print("\nSimulating treatments...")
    treatment_simulator = TreatmentSimulator(model, G, device)
    
    # Define treatment options to simulate
    treatment_options = [
        {'category': 'surgery', 'intensity': 0.8, 'duration': 1},
        {'category': 'radiation', 'dose': 60.0, 'duration': 30},
        {'category': 'chemotherapy', 'dose': 150, 'duration': 120}
    ]
    
    # Simulate for a patient
    patient_id = 'patient_1'
    tumor_id = 'tumor_1'
    
    simulation_results = treatment_simulator.simulate_treatments(
        patient_id, tumor_id, treatment_options, pyg_data
    )
    
    memory_tracker.log_memory("Treatments simulated")
    
    # Generate explanations
    print("\nGenerating explanations...")
    explainer = ExplainableGliomaTreatment(model, pyg_data, G, device)
    
    # Explain treatment for a specific index
    treatment_idx = 0  # First treatment
    explanation = explainer.explain_treatment_prediction(treatment_idx)
    
    memory_tracker.log_memory("Explanations generated")
    
    # Print results
    print("\nSimulated Treatment Results:")
    for option, results in simulation_results.items():
        print(f"\n{option.upper()}:")
        print(f"  Treatment: {results['config']['category']}")
        print(f"  Response Probability: {results['response_prob']:.3f}")
        print(f"  Predicted Survival: {results['survival_days']:.1f} days")
        print(f"  Uncertainty: {results['uncertainty']:.3f}")
    
    print("\nExplanation for Treatment Prediction:")
    print(f"Treatment ID: {explanation['treatment_id']}")
    print(f"Predicted Response: {explanation['predictions']['response_probability']:.3f}")
    print(f"Predicted Survival: {explanation['predictions']['survival_days']:.1f} days")
    print("\nFeature Importance for Response:")
    for feature, importance in explanation['feature_importance']['response']:
        print(f"  {feature}: {importance:.4f}")
    
    # Plot memory usage
    print("\nPlotting memory usage...")
    plt.figure(figsize=(12, 6))
    operations, memory = zip(*memory_tracker.log)
    plt.plot(memory)
    plt.xticks(range(len(operations)), operations, rotation=90)
    plt.xlabel('Operation')
    plt.ylabel('Memory Usage (MB)')
    plt.title('Memory Usage During Processing')
    plt.tight_layout()
    plt.savefig('memory_usage.png')
    print("Memory usage plot saved as 'memory_usage.png'")
    
    print("\nPipeline completed successfully!")


if __name__ == "__main__":
    main()