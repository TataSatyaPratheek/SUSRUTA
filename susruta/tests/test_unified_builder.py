# susruta/tests/graph_builder/test_unified_builder.py
"""
Tests for unified graph builder component.
"""

import os
import unittest
import networkx as nx
import numpy as np
import pandas as pd
import tempfile
import shutil
import torch
from torch_geometric.data import HeteroData

from susruta.graph_builder.unified_builder import UnifiedGraphBuilder
from susruta.utils.memory import MemoryTracker


class TestUnifiedGraphBuilder(unittest.TestCase):
    """Test cases for UnifiedGraphBuilder."""

    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.memory_tracker = MemoryTracker(threshold_mb=1000)
        self.unified_builder = UnifiedGraphBuilder(memory_limit_mb=1000)
        
        # Create mock graphs and data
        self.patient_id = 1001
        self.create_mock_graphs()
        self.create_mock_excel_files()
    
    def tearDown(self):
        """Clean up temporary test data."""
        shutil.rmtree(self.test_dir)
    
    def create_mock_graphs(self):
        """Create mock graph data for testing."""
        # Create a simple graph
        self.graph = nx.MultiDiGraph()
        
        # Add patient and tumor nodes
        patient_node = f"patient_{self.patient_id}"
        tumor_node = f"tumor_{self.patient_id}"
        
        self.graph.add_node(patient_node, type='patient', age=65, sex='male')
        self.graph.add_node(tumor_node, type='tumor', volume_mm3=15000.0, grade=4)
        self.graph.add_edge(patient_node, tumor_node, relation='has_tumor')
        
        # Add treatment nodes
        for i, category in enumerate(['surgery', 'radiation']):
            treatment_node = f"treatment_{i+1}_{self.patient_id}"
            self.graph.add_node(treatment_node, type='treatment', category=category)
            self.graph.add_edge(tumor_node, treatment_node, relation='treated_with')
        
        # Add sequence nodes
        for seq in ['t1c', 't2w']:
            seq_node = f"sequence_{seq}_{self.patient_id}"
            self.graph.add_node(seq_node, type='sequence', sequence=seq)
            self.graph.add_edge(tumor_node, seq_node, relation='has_sequence')
    
    def create_mock_excel_files(self):
        """Create mock Excel files for testing."""
        # Create scanner data
        scanner_df = pd.DataFrame([
            {
                'PatientID': self.patient_id,
                'Timepoint': 1,
                'ScannerManufacturer': 'Siemens',
                'ScannerModel': 'Prisma',
                'FieldStrength': 3.0,
                'SequenceType': 'T1'
            }
        ])
        
        # Create clinical data
        clinical_df = pd.DataFrame([
            {
                'patient_id': self.patient_id,
                'age': 65,
                'sex': 'male',
                'karnofsky_score': 80,
                'grade': 4,
                'histology': 'GBM'
            }
        ])
        
        # Create segmentation data
        segmentation_df = pd.DataFrame([
            {
                'PatientID': self.patient_id,
                'Timepoint': 1,
                'TumorVolume_mm3': 15200.0,
                'EnhancingVolume_mm3': 9800.0
            }
        ])
        
        # Save to Excel files
        scanner_path = os.path.join(self.test_dir, 'MR_Scanner_data.xlsx')
        clinical_path = os.path.join(self.test_dir, 'MUGliomaPost_ClinicalDataFINAL032025.xlsx')
        segmentation_path = os.path.join(self.test_dir, 'MUGliomaPost_Segmentation_Volumes.xlsx')
        
        scanner_df.to_excel(scanner_path, index=False)
        clinical_df.to_excel(clinical_path, index=False)
        segmentation_df.to_excel(segmentation_path, index=False)
        
        self.excel_paths = {
            'scanner': scanner_path,
            'clinical': clinical_path,
            'segmentation': segmentation_path
        }
    
    def create_mock_mri_dir(self):
        """Create mock MRI directory structure (empty)."""
        patient_dir = os.path.join(self.test_dir, f'PatientID_{self.patient_id:04d}')
        timepoint_dir = os.path.join(patient_dir, 'Timepoint_1')
        os.makedirs(timepoint_dir, exist_ok=True)
        
        return self.test_dir
    
    def test_unified_builder_init(self):
        """Test initialization of UnifiedGraphBuilder."""
        self.assertIsNotNone(self.unified_builder)
        self.assertEqual(self.unified_builder.memory_limit_mb, 1000)
        
        # Verify sub-components are initialized
        self.assertIsNotNone(self.unified_builder.mri_builder)
        self.assertIsNotNone(self.unified_builder.graph_integrator)
        self.assertIsNotNone(self.unified_builder.temporal_builder)
        self.assertIsNotNone(self.unified_builder.excel_integrator)
    
    def test_build_pytorch_geometric(self):
        """Test conversion to PyTorch Geometric format."""
        # Convert to PyTorch Geometric
        pyg_data = self.unified_builder.build_pytorch_geometric(self.graph)
        
        # Verify PyG data structure
        self.assertIsInstance(pyg_data, HeteroData)
        
        # Verify node types
        expected_node_types = ['patient', 'tumor', 'treatment', 'sequence']
        for node_type in expected_node_types:
            self.assertIn(node_type, pyg_data.node_types)
            
            # Verify node features exist
            self.assertTrue(hasattr(pyg_data[node_type], 'x'))
            self.assertIsInstance(pyg_data[node_type].x, torch.Tensor)
            self.assertEqual(pyg_data[node_type].x.dim(), 2)  # Should be 2D tensor
        
        # Verify edge connections
        self.assertGreater(len(pyg_data.edge_types), 0)
        
        # Check at least one edge type
        has_edge = False
        for edge_type in pyg_data.edge_types:
            if hasattr(pyg_data[edge_type], 'edge_index'):
                has_edge = True
                self.assertIsInstance(pyg_data[edge_type].edge_index, torch.Tensor)
                self.assertEqual(pyg_data[edge_type].edge_index.dim(), 2)
                self.assertEqual(pyg_data[edge_type].edge_index.size(0), 2)  # Source, target format
        
        self.assertTrue(has_edge, "No edges found in PyG data")
    
    def test_add_custom_features(self):
        """Test adding custom features to a graph."""
        # Define custom features
        custom_features = {
            f"patient_{self.patient_id}": {
                'risk_score': 0.85,
                'survival_prediction': 365
            },
            f"tumor_{self.patient_id}": {
                'growth_rate': 0.15,
                'heterogeneity_score': 0.7
            }
        }
        
        # Add custom features
        updated_graph = self.unified_builder.add_custom_features(self.graph, custom_features)
        
        # Verify features were added
        patient_node = f"patient_{self.patient_id}"
        tumor_node = f"tumor_{self.patient_id}"
        
        self.assertIn('risk_score', updated_graph.nodes[patient_node])
        self.assertEqual(updated_graph.nodes[patient_node]['risk_score'], 0.85)
        
        self.assertIn('growth_rate', updated_graph.nodes[tumor_node])
        self.assertEqual(updated_graph.nodes[tumor_node]['growth_rate'], 0.15)
    
    def test_get_patient_summary(self):
        """Test generating patient summary from graph."""
        # Generate patient summary
        summary = self.unified_builder.get_patient_summary(self.graph, self.patient_id)
        
        # Verify summary structure
        self.assertIsInstance(summary, dict)
        self.assertEqual(summary['patient_id'], self.patient_id)
        
        # Verify sections exist
        sections = ['demographics', 'tumor', 'treatments', 'sequences']
        for section in sections:
            self.assertIn(section, summary)
        
        # Verify demographics
        self.assertEqual(summary['demographics'].get('age'), 65)
        self.assertEqual(summary['demographics'].get('sex'), 'male')
        
        # Verify tumor info
        self.assertEqual(summary['tumor'].get('grade'), 4)
        self.assertEqual(summary['tumor'].get('volume_mm3'), 15000.0)
        
        # Verify treatments
        self.assertIsInstance(summary['treatments'], list)
        self.assertEqual(len(summary['treatments']), 2)
        
        # Check treatment categories
        treatment_categories = [t.get('category') for t in summary['treatments']]
        self.assertIn('surgery', treatment_categories)
        self.assertIn('radiation', treatment_categories)
    
    def test_get_node_subgraph(self):
        """Test extracting a node-centered subgraph."""
        # Extract subgraph around tumor node
        tumor_node = f"tumor_{self.patient_id}"
        
        # Get subgraph of depth 1
        subgraph = self.unified_builder.get_node_subgraph(
            graph=self.graph,
            node_id=tumor_node,
            depth=1
        )
        
        # Verify subgraph structure
        self.assertIsInstance(subgraph, nx.MultiDiGraph)
        self.assertGreater(subgraph.number_of_nodes(), 1)  # Should include more than just tumor node
        self.assertIn(tumor_node, subgraph)
        
        # Verify expected connections
        patient_node = f"patient_{self.patient_id}"
        self.assertIn(patient_node, subgraph)
        
        # Verify treatment nodes are included
        has_treatment = False
        for node, attrs in subgraph.nodes(data=True):
            if attrs.get('type') == 'treatment':
                has_treatment = True
                break
        
        self.assertTrue(has_treatment, "No treatment nodes found in subgraph")
        
        # Get deeper subgraph (depth 2)
        deep_subgraph = self.unified_builder.get_node_subgraph(
            graph=self.graph,
            node_id=tumor_node,
            depth=2
        )
        
        # Deeper graph should have at least as many nodes
        self.assertGreaterEqual(deep_subgraph.number_of_nodes(), subgraph.number_of_nodes())
    
    def test_clear_cache(self):
        """Test clearing the cache."""
        # First, populate the cache
        self.unified_builder._cached_graphs = {'test': nx.Graph()}
        self.unified_builder._cached_pytorch_geometric = HeteroData()
        
        # Clear cache
        self.unified_builder.clear_cache()
        
        # Verify cache is empty
        self.assertEqual(len(self.unified_builder._cached_graphs), 0)
        self.assertIsNone(self.unified_builder._cached_pytorch_geometric)


if __name__ == '__main__':
    unittest.main()