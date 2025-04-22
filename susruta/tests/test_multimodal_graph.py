# susruta/tests/graph_builder/test_multimodal_graph.py
"""
Tests for multimodal graph integration component.
"""

import os
import unittest
import networkx as nx
import numpy as np
import pandas as pd
import tempfile
import shutil

from susruta.graph_builder.multimodal_graph import MultimodalGraphIntegrator
from susruta.graph.knowledge_graph import GliomaKnowledgeGraph
from susruta.utils.memory import MemoryTracker


class TestMultimodalGraphIntegrator(unittest.TestCase):
    """Test cases for MultimodalGraphIntegrator."""

    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.memory_tracker = MemoryTracker(threshold_mb=500)
        self.graph_integrator = MultimodalGraphIntegrator(memory_limit_mb=500)
        
        # Create mock graph data
        self.create_mock_graphs()
        
        # Create mock Excel files
        self.create_mock_excel_files()
    
    def tearDown(self):
        """Clean up temporary test data."""
        shutil.rmtree(self.test_dir)
    
    def create_mock_graphs(self):
        """Create mock graph data for testing."""
        # Create MRI graph
        self.mri_graph = nx.Graph()
        
        # Add patient and tumor nodes
        self.patient_id = 1001
        patient_node = f"patient_{self.patient_id}"
        tumor_node = f"tumor_{self.patient_id}"
        
        self.mri_graph.add_node(patient_node, type='patient', timepoint=1)
        self.mri_graph.add_node(tumor_node, type='tumor', volume_mm3=15000.0, volume_voxels=1500.0)
        self.mri_graph.add_edge(patient_node, tumor_node, relation='has_tumor')
        
        # Add sequence nodes
        for seq in ['t1c', 't2w']:
            seq_node = f"sequence_{seq}_{self.patient_id}"
            self.mri_graph.add_node(seq_node, type='sequence', sequence=seq, mean=125.0, std=35.0)
            self.mri_graph.add_edge(tumor_node, seq_node, relation='has_sequence')
        
        # Add region nodes
        for i in range(1, 4):  # 3 regions
            region_node = f"region_{i}_{self.patient_id}"
            self.mri_graph.add_node(
                region_node, 
                type='region',
                volume_voxels=float(500 * i),
                center_x=float(10 + i),
                center_y=float(12),
                center_z=float(8)
            )
            self.mri_graph.add_edge(tumor_node, region_node, relation='has_region')
        
        # Connect adjacent regions
        self.mri_graph.add_edge(f"region_1_{self.patient_id}", f"region_2_{self.patient_id}", relation='adjacent_to')
        self.mri_graph.add_edge(f"region_2_{self.patient_id}", f"region_3_{self.patient_id}", relation='adjacent_to')
        
        # Create clinical graph
        kg_builder = GliomaKnowledgeGraph(memory_limit_mb=500)
        
        # Add minimal clinical data
        clinical_data = pd.DataFrame([{
            'patient_id': self.patient_id,
            'age': 65,
            'sex': 'male',
            'karnofsky_score': 80,
            'grade': 4,
            'histology': 'GBM',
            'location': 'temporal',
            'idh_mutation': 0,
            'mgmt_methylation': 1
        }])
        
        kg_builder.add_clinical_data(clinical_data)
        
        # Add treatments
        treatments = pd.DataFrame([
            {
                'patient_id': self.patient_id,
                'treatment_id': 1,
                'category': 'surgery',
                'dose': None,
                'duration_days': 1,
                'start_day': 0,
                'response': 'complete'
            },
            {
                'patient_id': self.patient_id,
                'treatment_id': 2,
                'category': 'radiation',
                'dose': 60.0,
                'duration_days': 30,
                'start_day': 14,
                'response': 'partial'
            }
        ])
        
        kg_builder.add_treatments(treatments)
        
        self.clinical_graph = kg_builder.G
    
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
            },
            {
                'PatientID': self.patient_id,
                'Timepoint': 1,
                'ScannerManufacturer': 'Siemens',
                'ScannerModel': 'Prisma',
                'FieldStrength': 3.0,
                'SequenceType': 'T2'
            }
        ])
        
        # Create segmentation data
        segmentation_df = pd.DataFrame([
            {
                'PatientID': self.patient_id,
                'Timepoint': 1,
                'TumorVolume_mm3': 15200.0,
                'EnhancingVolume_mm3': 9800.0,
                'NecrotisCoreVolume_mm3': 2100.0,
                'EdemaVolume_mm3': 25000.0
            }
        ])
        
        # Save to Excel files
        scanner_path = os.path.join(self.test_dir, 'MR_Scanner_data.xlsx')
        segmentation_path = os.path.join(self.test_dir, 'MUGliomaPost_Segmentation_Volumes.xlsx')
        
        scanner_df.to_excel(scanner_path, index=False)
        segmentation_df.to_excel(segmentation_path, index=False)
        
        self.scanner_path = scanner_path
        self.segmentation_path = segmentation_path
        
        # Create dictionary for convenience
        self.excel_data = {
            'scanner': scanner_df,
            'segmentation': segmentation_df
        }
    
    def test_graph_integrator_init(self):
        """Test initialization of MultimodalGraphIntegrator."""
        self.assertIsNotNone(self.graph_integrator)
        self.assertEqual(self.graph_integrator.memory_limit_mb, 500)
    
    def test_integrate_graphs(self):
        """Test integrating MRI and clinical graphs."""
        # Integrate graphs
        integrated_graph = self.graph_integrator.integrate_graphs(
            mri_graph=self.mri_graph,
            clinical_graph=self.clinical_graph,
            patient_id=self.patient_id
        )
        
        # Verify graph structure
        self.assertIsInstance(integrated_graph, nx.MultiDiGraph)
        self.assertGreater(integrated_graph.number_of_nodes(), 0)
        
        # Count nodes by type
        node_counts = {}
        for _, attrs in integrated_graph.nodes(data=True):
            node_type = attrs.get('type', 'unknown')
            if node_type not in node_counts:
                node_counts[node_type] = 0
            node_counts[node_type] += 1
        
        # Verify we have all expected node types
        expected_types = ['patient', 'tumor', 'treatment', 'sequence', 'region']
        for node_type in expected_types:
            self.assertIn(node_type, node_counts, f"Missing node type: {node_type}")
            self.assertGreater(node_counts[node_type], 0, f"No nodes of type: {node_type}")
        
        # Verify patient node exists and has clinical attributes
        patient_node = f"patient_{self.patient_id}"
        self.assertIn(patient_node, integrated_graph)
        patient_attrs = integrated_graph.nodes[patient_node]
        self.assertEqual(patient_attrs.get('type'), 'patient')
        self.assertEqual(patient_attrs.get('age'), 65)
        self.assertEqual(patient_attrs.get('sex'), 'male')
        
        # Verify tumor node exists and has attributes from both graphs
        tumor_node = f"tumor_{self.patient_id}"
        self.assertIn(tumor_node, integrated_graph)
        tumor_attrs = integrated_graph.nodes[tumor_node]
        self.assertEqual(tumor_attrs.get('type'), 'tumor')
        self.assertIn('volume_mm3', tumor_attrs)
        
        # Verify treatment nodes exist
        treatment_nodes = [node for node, attrs in integrated_graph.nodes(data=True) 
                           if attrs.get('type') == 'treatment']
        self.assertGreaterEqual(len(treatment_nodes), 2)
    
    def test_integrate_excel_data(self):
        """Test integrating Excel data into graph."""
        # Start with a simple graph
        base_graph = nx.MultiDiGraph()
        patient_node = f"patient_{self.patient_id}"
        tumor_node = f"tumor_{self.patient_id}"
        
        base_graph.add_node(patient_node, type='patient')
        base_graph.add_node(tumor_node, type='tumor')
        base_graph.add_edge(patient_node, tumor_node, relation='has_tumor')
        
        # Integrate Excel data
        updated_graph = self.graph_integrator._integrate_excel_data(
            graph=base_graph,
            excel_data=self.excel_data,
            patient_id=self.patient_id
        )
        
        # Verify scanner node was added
        scanner_node = f"scanner_{self.patient_id}"
        self.assertIn(scanner_node, updated_graph)
        
        # Verify tumor node was updated with segmentation data
        tumor_attrs = updated_graph.nodes[tumor_node]
        # --- START FIX ---
        # Use cleaned attribute names (lowercase, no _mm3)
        # Correct 'necroticcorevolume' to 'necrotiscorevolume'
        segmentation_attrs = ['tumorvolume', 'enhancingvolume', 'necrotiscorevolume', 'edemavolume']
        # --- END FIX ---

        # --- START DIAGNOSTIC PRINT ---
        print(f"\nDEBUG: Actual tumor attributes for node {tumor_node}: {tumor_attrs}\n")
        # --- END DIAGNOSTIC PRINT ---

        # Check for at least one segmentation attribute (case insensitive)
        found_attr = False
        for attr in tumor_attrs:
            # Ensure attr is a string before calling lower()
            if isinstance(attr, str) and any(seg_attr.lower() == attr.lower() for seg_attr in segmentation_attrs):
                found_attr = True
                break

        self.assertTrue(found_attr, "No segmentation attributes found in tumor node")

    
    def test_from_excel_sources(self):
        """Test building graph directly from Excel files."""
        # Test with minimal clinical data
        clinical_df = pd.DataFrame([{
            'patient_id': self.patient_id,
            'age': 65,
            'sex': 'male'
        }])
        
        # Save clinical data to Excel
        clinical_path = os.path.join(self.test_dir, 'MUGliomaPost_ClinicalDataFINAL032025.xlsx')
        clinical_df.to_excel(clinical_path, index=False)
        
        # Build graph from Excel sources
        graph = self.graph_integrator.from_excel_sources(
            scanner_path=self.scanner_path,
            clinical_path=clinical_path,
            segmentation_path=self.segmentation_path,
            patient_id=self.patient_id
        )
        
        # Verify graph structure
        self.assertIsInstance(graph, nx.MultiDiGraph)
        self.assertGreater(graph.number_of_nodes(), 0)
        
        # Verify patient node exists
        patient_node = f"patient_{self.patient_id}"
        self.assertIn(patient_node, graph)
        
        # Verify scanner data was incorporated
        scanner_related = False
        for node, attrs in graph.nodes(data=True):
            if any('scanner' in str(attr).lower() for attr in attrs.values()):
                scanner_related = True
                break
        
        self.assertTrue(scanner_related, "No scanner-related data found in graph")


if __name__ == '__main__':
    unittest.main()