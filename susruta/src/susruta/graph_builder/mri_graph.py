# susruta/src/susruta/graph_builder/mri_graph.py
"""
MRI-specific graph construction for glioma data analysis.

Builds graph representations from MRI brain scans, including tumor segmentation,
tissue boundaries, and connectivity patterns.
"""

from typing import Dict, List, Tuple, Any, Optional, Union, Set
import os
import gc
import numpy as np
import SimpleITK as sitk
import networkx as nx
import nibabel as nib
from pathlib import Path
import glob # Import glob

from ..data.mri import EfficientMRIProcessor
from ..utils.memory import MemoryTracker


class MRIGraphBuilder:
    """Builds graph representations from MRI scan data with memory efficiency."""

    def __init__(self, memory_limit_mb: float = 2000):
        """
        Initialize MRI graph builder.

        Args:
            memory_limit_mb: Memory limit in MB
        """
        self.memory_limit_mb = memory_limit_mb
        self.memory_tracker = MemoryTracker(threshold_mb=memory_limit_mb)
        self.mri_processor = EfficientMRIProcessor(memory_limit_mb=memory_limit_mb * 0.7)

    def build_mri_graph(self,
                       patient_id: int,
                       data_dir: str, # Expects the specific timepoint directory
                       timepoint: int = 1,
                       sequences: Optional[List[str]] = None) -> nx.Graph:
        """
        Build graph representation from MRI scans for a patient timepoint.

        Args:
            patient_id: Patient identifier
            data_dir: Specific directory for the patient and timepoint
                      (e.g., /path/to/base/PatientID_0003/Timepoint_1)
            timepoint: Timepoint number (used for logging/errors)
            sequences: Optional list of sequences to process (e.g., ['t1c', 't2w'])

        Returns:
            NetworkX graph representing MRI data for the timepoint
        """
        self.memory_tracker.log_memory("Starting MRI graph construction")

        # Create a new graph
        G = nx.Graph()

        # --- Use data_dir directly as it's the timepoint dir ---
        timepoint_dir = Path(data_dir)

        if not timepoint_dir.is_dir():
            raise ValueError(f"Data directory not found: {data_dir}")
        # --- End Change ---

        # Define sequences if not provided
        if sequences is None:
            sequences = ['t1c', 't1n', 't2f', 't2w']

        # Find mask file within the timepoint_dir
        mask_pattern = str(timepoint_dir / '*tumorMask*.nii*')
        mask_files = glob.glob(mask_pattern)

        if not mask_files:
            raise ValueError(f"No tumor mask found for patient {patient_id}, timepoint {timepoint} in {timepoint_dir}")

        mask_path = mask_files[0]

        # Extract features for all available sequences
        try:
            # --- Pass data_dir (timepoint_dir) correctly ---
            features = self.mri_processor.extract_features_for_patient(
                patient_id=patient_id,
                data_dir=str(timepoint_dir), # Pass the specific timepoint directory
                timepoint=timepoint,
                sequences=sequences
            )
            # --- End Change ---
            self.memory_tracker.log_memory("Extracted MRI features")
        except Exception as e:
            raise ValueError(f"Failed to extract MRI features: {e}")

        # Add patient node
        patient_node = f"patient_{patient_id}"
        G.add_node(patient_node, type='patient', timepoint=timepoint)

        # Add tumor node and connect to patient
        tumor_node = f"tumor_{patient_id}"
        tumor_features = features.get('tumor', {})
        G.add_node(tumor_node, type='tumor', **tumor_features)
        G.add_edge(patient_node, tumor_node, relation='has_tumor')

        # Add sequence nodes and connect to tumor
        for sequence, seq_features in features.items():
            if sequence == 'tumor':
                continue  # Skip tumor features as they're already added

            sequence_node = f"sequence_{sequence}_{patient_id}"
            G.add_node(sequence_node, type='sequence', sequence=sequence, **seq_features)
            G.add_edge(tumor_node, sequence_node, relation='has_sequence')

        # Extract tumor regions and build region graph
        try:
            region_subgraph = self._build_tumor_region_graph(mask_path, patient_id)
            G = nx.compose(G, region_subgraph)
            # Connect tumor node to region nodes
            for node in region_subgraph.nodes():
                if node.startswith('region_') and region_subgraph.nodes[node].get('type') == 'region':
                    G.add_edge(tumor_node, node, relation='has_region')
            self.memory_tracker.log_memory("Built tumor region graph")
        except Exception as e:
            print(f"Warning: Could not build tumor region graph: {e}")

        self.memory_tracker.log_memory("Completed MRI graph construction")
        return G

    def _build_tumor_region_graph(self, mask_path: str, patient_id: int) -> nx.Graph:
        """
        Build graph representation of tumor regions from mask.

        Args:
            mask_path: Path to tumor mask NIfTI file
            patient_id: Patient identifier

        Returns:
            NetworkX graph of tumor regions
        """
        region_graph = nx.Graph()

        try:
            # Load mask as nibabel for more flexible processing
            mask_nib = nib.load(mask_path)
            # Get mask data with memory efficiency
            mask_data = np.asarray(mask_nib.dataobj, dtype=np.int8)

            # Get tumor shape properties
            mask_sitk = sitk.ReadImage(mask_path)
            shape_stats = sitk.LabelShapeStatisticsImageFilter()
            shape_stats.Execute(mask_sitk)

            # Get bounding box safely
            label_to_analyze = 1
            if not shape_stats.HasLabel(label_to_analyze):
                 existing_labels = shape_stats.GetLabels()
                 valid_labels = [l for l in existing_labels if l > 0]
                 if valid_labels:
                     label_to_analyze = valid_labels[0]
                 else:
                     print(f"Warning: No valid labels found in mask {mask_path} for region analysis.")
                     return region_graph # Return empty if no valid labels

            if shape_stats.HasLabel(label_to_analyze):
                bbox = shape_stats.GetBoundingBox(label_to_analyze)
                # Convert to min/max coordinates format
                bbox_min = [bbox[0], bbox[1], bbox[2]]
                bbox_max = [bbox[0] + bbox[3] - 1, bbox[1] + bbox[4] - 1, bbox[2] + bbox[5] - 1]

                # Process only within bounding box to save memory
                # Ensure indices are integers and within bounds
                z_min, z_max = max(0, int(bbox_min[2])), min(mask_data.shape[0], int(bbox_max[2]) + 1)
                y_min, y_max = max(0, int(bbox_min[1])), min(mask_data.shape[1], int(bbox_max[1]) + 1)
                x_min, x_max = max(0, int(bbox_min[0])), min(mask_data.shape[2], int(bbox_max[0]) + 1)

                # Check if ROI is valid
                if z_min >= z_max or y_min >= y_max or x_min >= x_max:
                     print(f"Warning: Invalid ROI calculated for region analysis in {mask_path}. Skipping.")
                     return region_graph

                roi = mask_data[z_min:z_max, y_min:y_max, x_min:x_max]

                # Use connected component analysis to find distinct regions
                from scipy import ndimage
                labeled_array, num_regions = ndimage.label(roi > 0)

                # If too many regions, limit to manage memory
                max_regions = 10
                if num_regions > max_regions:
                    print(f"Warning: Found {num_regions} regions. Limiting to {max_regions} largest.")

                    # Find volumes of each region
                    region_sizes = np.zeros(num_regions + 1, dtype=int)
                    for i in range(1, num_regions + 1):
                        region_sizes[i] = np.sum(labeled_array == i)

                    # Keep only the largest regions
                    largest_regions = np.argsort(-region_sizes)[1:max_regions+1]  # Skip background (0)

                else:
                    largest_regions = range(1, num_regions + 1)

                # Add region nodes
                for i in largest_regions:
                    region_mask = (labeled_array == i)
                    if np.sum(region_mask) == 0:
                        continue  # Skip empty regions

                    # Calculate region properties
                    region_volume = float(np.sum(region_mask))
                    region_center = ndimage.center_of_mass(region_mask)

                    # Add region node
                    region_node = f"region_{i}_{patient_id}"
                    region_graph.add_node(
                        region_node,
                        type='region',
                        volume_voxels=region_volume,
                        # Adjust center coordinates relative to the full image
                        center_z=float(region_center[0] + z_min),
                        center_y=float(region_center[1] + y_min),
                        center_x=float(region_center[2] + x_min)
                    )

                # Connect adjacent regions
                for i in largest_regions:
                    region_i_mask = (labeled_array == i)
                    if np.sum(region_i_mask) == 0:
                        continue

                    for j in largest_regions:
                        if i >= j:  # Avoid duplicates and self-connections
                            continue

                        region_j_mask = (labeled_array == j)
                        if np.sum(region_j_mask) == 0:
                            continue

                        # Check for adjacency - dilate region i and check overlap with j
                        dilated_i = ndimage.binary_dilation(region_i_mask)
                        if np.any(dilated_i & region_j_mask):
                            region_i_node = f"region_{i}_{patient_id}"
                            region_j_node = f"region_{j}_{patient_id}"
                            region_graph.add_edge(region_i_node, region_j_node, relation='adjacent_to')

            # Clean up to save memory
            del mask_data, mask_nib, mask_sitk
            gc.collect()

        except Exception as e:
            print(f"Error in tumor region graph construction: {e}")
            # Return empty graph in case of error
            return nx.Graph()

        return region_graph

    def extract_connectivity_features(self,
                                    patient_id: int,
                                    data_dir: str,
                                    timepoint: int = 1) -> Dict[str, Any]:
        """
        Extract brain connectivity features from DTI or functional MRI data if available.

        Args:
            patient_id: Patient identifier
            data_dir: Base directory for data
            timepoint: Timepoint number

        Returns:
            Dictionary of connectivity features
        """
        connectivity_features = {}
        timepoint_dir = os.path.join(data_dir, f'PatientID_{patient_id:04d}', f'Timepoint_{timepoint}')

        # Check for DTI or functional MRI data
        dti_files = glob.glob(os.path.join(timepoint_dir, '*dti*.nii*'))
        fmri_files = glob.glob(os.path.join(timepoint_dir, '*fmri*.nii*')) + \
                     glob.glob(os.path.join(timepoint_dir, '*bold*.nii*'))

        if dti_files:
            # Process DTI data if available
            try:
                dti_features = self._process_dti_data(dti_files[0])
                connectivity_features.update(dti_features)
            except Exception as e:
                print(f"Warning: Failed to process DTI data: {e}")

        if fmri_files:
            # Process functional MRI data if available
            try:
                fmri_features = self._process_fmri_data(fmri_files[0])
                connectivity_features.update(fmri_features)
            except Exception as e:
                print(f"Warning: Failed to process fMRI data: {e}")

        return connectivity_features

    def _process_dti_data(self, dti_file: str) -> Dict[str, Any]:
        """
        Process DTI data to extract connectivity features.

        Args:
            dti_file: Path to DTI data file

        Returns:
            Dictionary of DTI-derived features
        """
        # Simplified DTI processing - in a real scenario, this would use proper DTI analysis
        features = {}

        try:
            # For demo purposes - just extract some basic properties
            # In a real implementation, this would use DTI-specific libraries
            img = nib.load(dti_file)
            data = np.asarray(img.dataobj)

            # Extract simple metrics
            features['dti_mean_value'] = float(np.mean(data))
            features['dti_volume'] = float(np.prod(img.header.get_zooms()[:3]) * np.sum(data > 0))

            # Clean up
            del data, img
            gc.collect()

        except Exception as e:
            print(f"Error in DTI processing: {e}")

        return features

    def _process_fmri_data(self, fmri_file: str) -> Dict[str, Any]:
        """
        Process functional MRI data to extract connectivity features.

        Args:
            fmri_file: Path to functional MRI data file

        Returns:
            Dictionary of fMRI-derived features
        """
        # Simplified fMRI processing - in a real scenario, this would use proper fMRI analysis
        features = {}

        try:
            # For demo purposes - just extract some basic properties
            img = nib.load(fmri_file)
            data = np.asarray(img.dataobj)

            # Extract simple metrics
            features['fmri_mean_value'] = float(np.mean(data))
            features['fmri_std_value'] = float(np.std(data))

            # Clean up
            del data, img
            gc.collect()

        except Exception as e:
            print(f"Error in fMRI processing: {e}")

        return features

    def build_structural_connectivity_graph(self,
                                          patient_id: int,
                                          data_dir: str,
                                          timepoint: int = 1) -> nx.Graph:
        """
        Build a brain structural connectivity graph from DTI data if available.

        Args:
            patient_id: Patient identifier
            data_dir: Base directory for data
            timepoint: Timepoint number

        Returns:
            NetworkX graph of structural connectivity
        """
        G = nx.Graph()
        timepoint_dir = os.path.join(data_dir, f'PatientID_{patient_id:04d}', f'Timepoint_{timepoint}')

        # Check for DTI data
        dti_files = glob.glob(os.path.join(timepoint_dir, '*dti*.nii*'))

        if not dti_files:
            return G  # Return empty graph if no DTI data

        try:
            # This is a placeholder for actual DTI tractography analysis
            # In a real implementation, this would extract actual structural connections

            # Add regions of interest (ROIs) as nodes
            num_rois = 5  # Simplified example with 5 ROIs
            for i in range(num_rois):
                roi_node = f"roi_{i}_{patient_id}"
                G.add_node(roi_node, type='roi', region_id=i)

            # Add edges representing connections between ROIs
            # In reality, these would be determined from DTI tractography
            for i in range(num_rois):
                for j in range(i+1, num_rois):
                    # Randomly determine if ROIs are connected (for demo only)
                    if np.random.random() < 0.7:  # 70% chance of connection
                        roi_i_node = f"roi_{i}_{patient_id}"
                        roi_j_node = f"roi_{j}_{patient_id}"
                        # Connection strength would be determined from actual data
                        G.add_edge(roi_i_node, roi_j_node,
                                 relation='connected_to',
                                 strength=float(np.random.random()))

        except Exception as e:
            print(f"Error in structural connectivity graph construction: {e}")
            return nx.Graph()  # Return empty graph on error

        return G
