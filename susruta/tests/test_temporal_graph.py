# susruta/tests/test_temporal_graph.py # Corrected path reference
"""
Tests for temporal graph builder component.
"""

import unittest
import networkx as nx
import numpy as np

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
    from susruta.graph_builder.temporal_graph import TemporalGraphBuilder
    from susruta.utils.memory import MemoryTracker
except ImportError as e:
    print(f"Error importing susruta modules in test_temporal_graph: {e}")
    # Allow tests to be skipped if imports fail
    TemporalGraphBuilder = None
    MemoryTracker = None


@unittest.skipIf(TemporalGraphBuilder is None, "Skipping Temporal Graph tests due to import error")
class TestTemporalGraphBuilder(unittest.TestCase):
    """Test cases for TemporalGraphBuilder."""

    def setUp(self):
        """Set up test environment."""
        self.memory_tracker = MemoryTracker(threshold_mb=500)
        self.temporal_builder = TemporalGraphBuilder(memory_limit_mb=500)

        # Create mock timepoint graphs using Patient 3
        self.patient_id = 3 # Use patient 3
        # Use valid timepoints for patient 3 (1, 2, 5)
        self.timepoints_used = [1, 2, 5]
        self.create_mock_timepoint_graphs()

    def create_mock_timepoint_graphs(self):
        """Create mock timepoint graphs for testing."""
        self.timepoint_graphs = {}

        # Create graphs for specified timepoints
        for tp_idx, tp in enumerate(self.timepoints_used):
            graph = nx.MultiDiGraph()

            # Add patient and tumor nodes
            patient_node = f"patient_{self.patient_id}"
            tumor_node = f"tumor_{self.patient_id}"

            # Simulate growing tumor (use tp_idx for progression)
            tumor_volume = 10000.0 * (1.0 + 0.2 * tp_idx)

            graph.add_node(patient_node, type='patient')
            graph.add_node(tumor_node, type='tumor', volume_mm3=tumor_volume)
            graph.add_edge(patient_node, tumor_node, relation='has_tumor')

            # Add treatment at timepoint 2
            if tp == 2:
                treatment_node = f"treatment_1_{self.patient_id}" # Use a consistent treatment ID base
                graph.add_node(treatment_node, type='treatment', category='radiation',
                              dose=60.0, duration=30)
                graph.add_edge(tumor_node, treatment_node, relation='treated_with')

            self.timepoint_graphs[tp] = graph

    def test_temporal_graph_builder_init(self):
        """Test initialization of TemporalGraphBuilder."""
        self.assertIsNotNone(self.temporal_builder)
        self.assertEqual(self.temporal_builder.memory_limit_mb, 500)

    def test_build_temporal_graph(self):
        """Test building a temporal graph from timepoint graphs."""
        # Build temporal graph
        temporal_graph = self.temporal_builder.build_temporal_graph(
            timepoint_graphs=self.timepoint_graphs,
            patient_id=self.patient_id
        )

        # Verify graph structure
        self.assertIsInstance(temporal_graph, nx.MultiDiGraph)
        self.assertGreater(temporal_graph.number_of_nodes(), 0)

        # Verify timepoint suffixes in node IDs
        for node in temporal_graph.nodes():
            self.assertTrue(any(node.endswith(f"_tp{tp}") for tp in self.timepoints_used),
                          f"Node {node} doesn't have a valid timepoint suffix")

        # Verify temporal connections exist between same node types
        temporal_edges = 0
        sorted_tps = sorted(self.timepoints_used)
        for i in range(len(sorted_tps) - 1):
            tp1 = sorted_tps[i]
            tp2 = sorted_tps[i+1]
            # Check connection between patient nodes
            node1 = f"patient_{self.patient_id}_tp{tp1}"
            node2 = f"patient_{self.patient_id}_tp{tp2}"
            if temporal_graph.has_node(node1) and temporal_graph.has_node(node2):
                 self.assertTrue(temporal_graph.has_edge(node1, node2))
                 temporal_edges += 1
            # Check connection between tumor nodes
            node1 = f"tumor_{self.patient_id}_tp{tp1}"
            node2 = f"tumor_{self.patient_id}_tp{tp2}"
            if temporal_graph.has_node(node1) and temporal_graph.has_node(node2):
                 self.assertTrue(temporal_graph.has_edge(node1, node2))
                 temporal_edges += 1


        # Verify we have temporal edges (at least patient and tumor connections)
        self.assertGreaterEqual(temporal_edges, 2 * (len(self.timepoints_used) - 1), "Missing temporal edges")

    def test_compute_progression_metrics(self):
        """Test computing tumor progression metrics."""
        # Build temporal graph
        temporal_graph = self.temporal_builder.build_temporal_graph(
            timepoint_graphs=self.timepoint_graphs,
            patient_id=self.patient_id
        )

        # Compute progression metrics
        metrics = self.temporal_builder.compute_progression_metrics(
            temporal_graph=temporal_graph,
            patient_id=self.patient_id
        )

        # Verify metrics structure
        self.assertIsInstance(metrics, dict)

        # Verify tumor volumes are stored
        self.assertIn('tumor_volumes', metrics)
        self.assertEqual(len(metrics['tumor_volumes']), len(self.timepoints_used))
        for tp in self.timepoints_used:
             self.assertIn(tp, metrics['tumor_volumes'])

        # Verify growth rates
        self.assertIn('growth_rates', metrics)
        self.assertEqual(len(metrics['growth_rates']), len(self.timepoints_used) - 1)

        # Verify volume change metrics
        self.assertIn('initial_volume', metrics)
        self.assertIn('final_volume', metrics)
        self.assertIn('volume_change', metrics)
        self.assertGreater(metrics['volume_change'], 0)  # Should be growing in mock data

    def test_identify_progression_patterns(self):
        """Test identifying progression patterns."""
        # Build temporal graph
        temporal_graph = self.temporal_builder.build_temporal_graph(
            timepoint_graphs=self.timepoint_graphs,
            patient_id=self.patient_id
        )

        # Identify progression patterns
        patterns = self.temporal_builder.identify_progression_patterns(
            temporal_graph=temporal_graph,
            patient_id=self.patient_id
        )

        # Verify patterns structure
        self.assertIsInstance(patterns, dict)

        # Verify growth pattern is identified
        self.assertIn('growth_pattern', patterns)
        self.assertIn(patterns['growth_pattern'],
                     ['rapid_growth', 'steady_growth', 'stable', 'regression', 'variable', 'unknown']) # Added unknown

        # Verify treatment response pattern
        self.assertIn('treatment_response', patterns)
        self.assertIn(patterns['treatment_response'],
                     ['good_response', 'partial_response', 'stable_disease', 'progressive_disease', 'unknown'])


    def test_extract_temporal_features(self):
        """Test extracting features from temporal graph."""
        # Build temporal graph
        temporal_graph = self.temporal_builder.build_temporal_graph(
            timepoint_graphs=self.timepoint_graphs,
            patient_id=self.patient_id
        )

        # Extract temporal features
        features = self.temporal_builder.extract_temporal_features(
            temporal_graph=temporal_graph,
            patient_id=self.patient_id
        )

        # Verify features structure
        self.assertIsInstance(features, dict)

        # Verify essential features exist
        essential_features = ['timepoints', 'initial_volume', 'final_volume',
                             'volume_change', 'growth_pattern_numeric',
                             'treatment_response_numeric']

        for feature in essential_features:
            self.assertIn(feature, features, f"Missing feature: {feature}")
            self.assertIsInstance(features[feature], (int, float, np.number))

    def test_merge_latest_graph(self):
        """Test merging latest timepoint from temporal graph with static graph."""
        # Build temporal graph
        temporal_graph = self.temporal_builder.build_temporal_graph(
            timepoint_graphs=self.timepoint_graphs,
            patient_id=self.patient_id
        )

        # Create a static graph
        static_graph = nx.MultiDiGraph()
        static_graph.add_node(f"patient_{self.patient_id}", type='patient', additional_attr='test')

        # Merge latest timepoint
        merged_graph = self.temporal_builder.merge_latest_graph(
            temporal_graph=temporal_graph,
            static_graph=static_graph
        )

        # Verify merged graph structure
        self.assertIsInstance(merged_graph, nx.MultiDiGraph)

        # Verify patient node exists with additional attribute
        patient_node = f"patient_{self.patient_id}"
        self.assertIn(patient_node, merged_graph)
        self.assertEqual(merged_graph.nodes[patient_node].get('additional_attr'), 'test')

        # Verify tumor node exists with latest volume
        tumor_node = f"tumor_{self.patient_id}"
        self.assertIn(tumor_node, merged_graph)
        self.assertIn('volume_mm3', merged_graph.nodes[tumor_node])
        # Check volume matches the last timepoint's volume
        latest_tp = max(self.timepoints_used)
        latest_tp_idx = self.timepoints_used.index(latest_tp)
        expected_latest_volume = 10000.0 * (1.0 + 0.2 * latest_tp_idx)
        self.assertAlmostEqual(merged_graph.nodes[tumor_node]['volume_mm3'], expected_latest_volume)


        # Verify that we don't have timepoint suffixes in the merged graph
        for node in merged_graph.nodes():
            self.assertFalse('_tp' in node, f"Node {node} still has timepoint suffix")


if __name__ == '__main__':
    unittest.main()
