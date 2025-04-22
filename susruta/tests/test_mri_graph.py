# susruta/tests/graph_builder/test_mri_graph.py
"""
Tests for MRI graph builder component.
"""

import os
import unittest
import networkx as nx
import numpy as np
import SimpleITK as sitk
import nibabel as nib
from pathlib import Path
import tempfile
import shutil

from susruta.graph_builder.mri_graph import MRIGraphBuilder
from susruta.utils.memory import MemoryTracker


class TestMRIGraphBuilder(unittest.TestCase):
    """Test cases for MRIGraphBuilder."""

    def setUp(self):
        """Set up test environment with mock MRI data."""
        self.test_dir = tempfile.mkdtemp()
        self.memory_tracker = MemoryTracker(threshold_mb=500)
        self.mri_builder = MRIGraphBuilder(memory_limit_mb=500)
        
        # Create test data structure
        self.patient_id = 1001
        self.timepoint = 1
        self.patient_dir = os.path.join(self.test_dir, f'PatientID_{self.patient_id:04d}')
        self.timepoint_dir = os.path.join(self.patient_dir, f'Timepoint_{self.timepoint}')
        os.makedirs(self.timepoint_dir, exist_ok=True)
        
        # Create mock MRI and mask data
        self.create_mock_mri_data()
    
    def tearDown(self):
        """Clean up temporary test data."""
        shutil.rmtree(self.test_dir)
    
    def create_mock_mri_data(self):
        """Create mock MRI and mask data for testing."""
        # Create small test volumes to save memory and time
        size = (20, 20, 20)
        
        # Create T1c MRI data
        t1c_data = np.zeros(size, dtype=np.float32)
        # Add a simulated brain
        center = (10, 10, 10)
        radius = 8
        for i in range(size[0]):
            for j in range(size[1]):
                for k in range(size[2]):
                    dist = np.sqrt((i-center[0])**2 + (j-center[1])**2 + (k-center[2])**2)
                    if dist < radius:
                        t1c_data[i, j, k] = 100 * (1 - dist/radius)
        
        # Add a simulated tumor
        tumor_center = (14, 12, 10)
        tumor_radius = 3
        for i in range(size[0]):
            for j in range(size[1]):
                for k in range(size[2]):
                    dist = np.sqrt((i-tumor_center[0])**2 + (j-tumor_center[1])**2 + (k-tumor_center[2])**2)
                    if dist < tumor_radius:
                        t1c_data[i, j, k] = 200 * (1 - dist/tumor_radius)
        
        # Create tumor mask
        mask_data = np.zeros(size, dtype=np.uint8)
        for i in range(size[0]):
            for j in range(size[1]):
                for k in range(size[2]):
                    dist = np.sqrt((i-tumor_center[0])**2 + (j-tumor_center[1])**2 + (k-tumor_center[2])**2)
                    if dist < tumor_radius:
                        mask_data[i, j, k] = 1
        
        # Save as NIFTI files
        t1c_img = nib.Nifti1Image(t1c_data, np.eye(4))
        mask_img = nib.Nifti1Image(mask_data, np.eye(4))
        
        t1c_path = os.path.join(self.timepoint_dir, 't1c.nii.gz')
        mask_path = os.path.join(self.timepoint_dir, 'tumorMask.nii.gz')
        
        nib.save(t1c_img, t1c_path)
        nib.save(mask_img, mask_path)
        
        # Create a simple T2w image (copy of T1c for simplicity)
        t2w_path = os.path.join(self.timepoint_dir, 't2w.nii.gz')
        nib.save(t1c_img, t2w_path)
    
    def test_mri_graph_builder_init(self):
        """Test initialization of MRIGraphBuilder."""
        self.assertIsNotNone(self.mri_builder)
        self.assertEqual(self.mri_builder.memory_limit_mb, 500)
    
    def test_compute_bounding_box(self):
        """Test bounding box computation from mask."""
        mask_path = os.path.join(self.timepoint_dir, 'tumorMask.nii.gz')
        
        # --- START FIX ---
        # Call the method on the mri_processor instance within the builder
        bbox_min, bbox_max = self.mri_builder.mri_processor.compute_bounding_box(mask_path)
        # --- END FIX ---

        
        # Verify bbox is not empty and has correct structure
        self.assertIsNotNone(bbox_min)
        self.assertIsNotNone(bbox_max)
        self.assertEqual(len(bbox_min), 3)
        self.assertEqual(len(bbox_max), 3)
        
        # Verify that min coordinates are less than max
        for i in range(3):
            self.assertLessEqual(bbox_min[i], bbox_max[i])
    
    def test_build_mri_graph(self):
        """Test building a graph from MRI data."""
        # Build graph from test data
        graph = self.mri_builder.build_mri_graph(
            patient_id=self.patient_id,
            data_dir=self.test_dir,
            timepoint=self.timepoint,
            sequences=['t1c', 't2w']
        )
        
        # Verify graph structure
        self.assertIsInstance(graph, nx.Graph)
        self.assertGreater(graph.number_of_nodes(), 0)
        
        # Verify specific node types
        node_types = set()
        for _, attrs in graph.nodes(data=True):
            if 'type' in attrs:
                node_types.add(attrs['type'])
        
        # Should have at least patient, tumor, and sequence nodes
        self.assertIn('patient', node_types)
        self.assertIn('tumor', node_types)
        self.assertIn('sequence', node_types)
        
        # Verify patient node exists
        patient_node = f"patient_{self.patient_id}"
        self.assertIn(patient_node, graph)
        
        # Verify tumor node exists and is connected to patient
        tumor_node = f"tumor_{self.patient_id}"
        self.assertIn(tumor_node, graph)
        self.assertTrue(graph.has_edge(patient_node, tumor_node))
        
        # Verify sequence nodes exist and are connected to tumor
        sequence_nodes = [node for node, attrs in graph.nodes(data=True) 
                         if attrs.get('type') == 'sequence']
        self.assertGreaterEqual(len(sequence_nodes), 1)
        
        for seq_node in sequence_nodes:
            self.assertTrue(graph.has_edge(tumor_node, seq_node))


if __name__ == '__main__':
    unittest.main()