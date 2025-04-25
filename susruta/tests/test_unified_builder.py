# susruta/tests/test_unified_builder.py # Corrected path reference
"""
Tests for unified graph builder component.
"""

import os
import unittest
from unittest.mock import patch, MagicMock # Import patch
import networkx as nx
import numpy as np
import pandas as pd
import tempfile
import shutil
import torch
from torch_geometric.data import HeteroData

# Add project root to path to allow importing susruta
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[1]
# Ensure correct relative path if src is not directly under project_root
susruta_src_path = project_root # Assuming src is directly under project_root
if not (susruta_src_path / 'susruta').exists():
     susruta_src_path = project_root / 'src' # Try src subdirectory

if str(susruta_src_path) not in sys.path:
     sys.path.insert(0, str(susruta_src_path))

try:
    from susruta.graph_builder.unified_builder import UnifiedGraphBuilder
    from susruta.utils.memory import MemoryTracker
    # Import sub-builders for mocking
    from susruta.graph_builder.mri_graph import MRIGraphBuilder
    from susruta.graph_builder.multimodal_graph import MultimodalGraphIntegrator
    from susruta.graph_builder.temporal_graph import TemporalGraphBuilder
    from susruta.data.excel_integration import MultimodalDataIntegrator as ExcelIntegrator # Alias to avoid name clash
except ImportError as e:
    print(f"Error importing susruta modules in test_unified_builder: {e}")
    # Allow tests to be skipped if imports fail
    UnifiedGraphBuilder = None
    MemoryTracker = None
    MRIGraphBuilder = None
    MultimodalGraphIntegrator = None
    TemporalGraphBuilder = None
    ExcelIntegrator = None


@unittest.skipIf(UnifiedGraphBuilder is None, "Skipping Unified Builder tests due to import error")
class TestUnifiedGraphBuilder(unittest.TestCase):
    """Test cases for UnifiedGraphBuilder."""

    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.memory_tracker = MemoryTracker(threshold_mb=1000)
        self.unified_builder = UnifiedGraphBuilder(memory_limit_mb=1000)

        # Create mock graphs and data using Patient 3
        self.patient_id = 3 # Use patient 3
        self.create_mock_graphs()
        self.create_mock_excel_files()
        self.mri_dir = self.create_mock_mri_dir() # Store the MRI base dir

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
                'PatientID': self.patient_id, # Use patient 3
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
                'patient_id': self.patient_id, # Use patient 3
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
                'PatientID': self.patient_id, # Use patient 3
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
        """Create mock MRI directory structure (with dummy files)."""
        patient_dir = os.path.join(self.test_dir, f'PatientID_{self.patient_id:04d}')
        # Create timepoints 1 and 2 for patient 3
        for tp in [1, 2]:
            timepoint_dir = os.path.join(patient_dir, f'Timepoint_{tp}')
            os.makedirs(timepoint_dir, exist_ok=True)
            # Create dummy files (just need them to exist for glob)
            Path(os.path.join(timepoint_dir, f'P{self.patient_id}_T{tp}_t1c.nii.gz')).touch()
            Path(os.path.join(timepoint_dir, f'P{self.patient_id}_T{tp}_tumorMask.nii.gz')).touch()

        return self.test_dir # Return the base directory containing PatientID_XXXX

    def test_unified_builder_init(self):
        """Test initialization of UnifiedGraphBuilder."""
        self.assertIsNotNone(self.unified_builder)
        self.assertEqual(self.unified_builder.memory_limit_mb, 1000)

        # Verify sub-components are initialized
        self.assertIsNotNone(self.unified_builder.mri_builder)
        self.assertIsNotNone(self.unified_builder.graph_integrator)
        self.assertIsNotNone(self.unified_builder.temporal_builder)
        self.assertIsNotNone(self.unified_builder.excel_integrator)

    # --- Unskipped and Mocked test_build_unified_graph ---
    @patch.object(MRIGraphBuilder, 'build_mri_graph')
    @patch.object(MultimodalGraphIntegrator, 'from_excel_sources')
    @patch.object(MultimodalGraphIntegrator, 'integrate_graphs')
    @patch.object(TemporalGraphBuilder, 'build_temporal_graph')
    @patch.object(TemporalGraphBuilder, 'extract_temporal_features')
    def test_build_unified_graph(self, mock_temporal_features, mock_temporal_graph,
                                mock_integrate_graphs, mock_excel_graph, mock_mri_graph):
        """Test building comprehensive unified graph with mocking."""

        # --- Configure Mocks ---
        # Mock MRI graph builder
        mock_mri_graph_tp1 = nx.Graph()
        mock_mri_graph_tp1.add_node(f"patient_{self.patient_id}", type='patient')
        mock_mri_graph_tp1.add_node(f"tumor_{self.patient_id}", type='tumor', mri_vol=100)
        mock_mri_graph_tp1.add_edge(f"patient_{self.patient_id}", f"tumor_{self.patient_id}")

        mock_mri_graph_tp2 = nx.Graph()
        mock_mri_graph_tp2.add_node(f"patient_{self.patient_id}", type='patient')
        mock_mri_graph_tp2.add_node(f"tumor_{self.patient_id}", type='tumor', mri_vol=120)
        mock_mri_graph_tp2.add_edge(f"patient_{self.patient_id}", f"tumor_{self.patient_id}")

        mock_mri_graph.side_effect = lambda patient_id, data_dir, timepoint, **kwargs: \
            mock_mri_graph_tp1 if timepoint == 1 else mock_mri_graph_tp2

        # Mock Excel graph builder
        mock_excel_graph_tp1 = nx.MultiDiGraph()
        mock_excel_graph_tp1.add_node(f"patient_{self.patient_id}", type='patient', age=65)
        mock_excel_graph_tp1.add_node(f"tumor_{self.patient_id}", type='tumor', grade=4)
        mock_excel_graph_tp1.add_edge(f"patient_{self.patient_id}", f"tumor_{self.patient_id}", relation='has_tumor')

        mock_excel_graph_tp2 = nx.MultiDiGraph() # Assume same excel data for TP2 for simplicity
        mock_excel_graph_tp2.add_node(f"patient_{self.patient_id}", type='patient', age=65)
        mock_excel_graph_tp2.add_node(f"tumor_{self.patient_id}", type='tumor', grade=4)
        mock_excel_graph_tp2.add_edge(f"patient_{self.patient_id}", f"tumor_{self.patient_id}", relation='has_tumor')

        mock_excel_graph.side_effect = lambda scanner_path, clinical_path, segmentation_path, patient_id, timepoint: \
            mock_excel_graph_tp1 if timepoint == 1 else mock_excel_graph_tp2

        # Mock integrate_graphs
        def mock_integrate_side_effect(mri_graph, clinical_graph, patient_id):
            # Simple merge for testing
            G = nx.MultiDiGraph()
            G.add_nodes_from(clinical_graph.nodes(data=True))
            G.add_edges_from(clinical_graph.edges(data=True, keys=True))
            for node, attrs in mri_graph.nodes(data=True):
                if node not in G: G.add_node(node, **attrs)
                else: G.nodes[node].update(attrs)
            return G
        mock_integrate_graphs.side_effect = mock_integrate_side_effect

        # Mock temporal graph builder
        mock_temporal_graph_result = nx.MultiDiGraph()
        mock_temporal_graph_result.add_node(f"patient_{self.patient_id}_tp1", type='patient', timepoint=1, age=65)
        mock_temporal_graph_result.add_node(f"tumor_{self.patient_id}_tp1", type='tumor', timepoint=1, mri_vol=100, grade=4)
        mock_temporal_graph_result.add_node(f"patient_{self.patient_id}_tp2", type='patient', timepoint=2, age=65)
        mock_temporal_graph_result.add_node(f"tumor_{self.patient_id}_tp2", type='tumor', timepoint=2, mri_vol=120, grade=4)
        mock_temporal_graph_result.add_edge(f"patient_{self.patient_id}_tp1", f"patient_{self.patient_id}_tp2", relation='same_patient')
        mock_temporal_graph_result.add_edge(f"tumor_{self.patient_id}_tp1", f"tumor_{self.patient_id}_tp2", relation='progression')
        mock_temporal_graph.return_value = mock_temporal_graph_result

        # Mock temporal feature extraction
        mock_temporal_features.return_value = {'growth_rate': 0.2, 'volume_change': 20}

        # --- Execute the method ---
        unified_graph = self.unified_builder.build_unified_graph(
            patient_id=self.patient_id,
            mri_dir=self.mri_dir,
            excel_paths=self.excel_paths,
            timepoints=[1, 2], # Use timepoints created in mock MRI dir
            include_temporal=True
        )

        # --- Assertions ---
        self.assertIsInstance(unified_graph, nx.MultiDiGraph)
        # Check that mocks were called
        self.assertEqual(mock_mri_graph.call_count, 2) # Called for TP1 and TP2
        self.assertEqual(mock_excel_graph.call_count, 2)
        self.assertEqual(mock_integrate_graphs.call_count, 2)
        mock_temporal_graph.assert_called_once()
        mock_temporal_features.assert_called_once()

        # Check the structure of the returned temporal graph
        self.assertIn(f"patient_{self.patient_id}_tp1", unified_graph)
        self.assertIn(f"tumor_{self.patient_id}_tp2", unified_graph)
        self.assertTrue(unified_graph.has_edge(f"tumor_{self.patient_id}_tp1", f"tumor_{self.patient_id}_tp2"))
        self.assertEqual(unified_graph.nodes[f"tumor_{self.patient_id}_tp1"]['mri_vol'], 100)
        self.assertEqual(unified_graph.nodes[f"tumor_{self.patient_id}_tp2"]['mri_vol'], 120)
        self.assertEqual(unified_graph.nodes[f"patient_{self.patient_id}_tp1"]['age'], 65)

    # --- End test_build_unified_graph ---


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
            # Check feature dimension is > 0
            self.assertGreater(pyg_data[node_type].x.shape[1], 0)

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
