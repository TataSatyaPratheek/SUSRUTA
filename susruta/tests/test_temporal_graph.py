# susruta/tests/graph_builder/test_temporal_graph.py
"""
Tests for temporal graph builder component.
"""

import unittest
import networkx as nx
import numpy as np

from susruta.graph_builder.temporal_graph import TemporalGraphBuilder
from susruta.utils.memory import MemoryTracker


class TestTemporalGraphBuilder(unittest.TestCase):
    """Test cases for TemporalGraphBuilder."""

    def setUp(self):
        """Set up test environment."""
        self.memory_tracker = MemoryTracker(threshold_mb=500)
        self.temporal_builder = TemporalGraphBuilder(memory_limit_mb=500)
        
        # Create mock timepoint graphs
        self.patient_id = 1001
        self.create_mock_timepoint_graphs()
    
    def create_mock_timepoint_graphs(self):
        """Create mock timepoint graphs for testing."""
        self.timepoint_graphs = {}
        
        # Create graphs for 3 timepoints
        for tp in range(1, 4):
            graph = nx.MultiDiGraph()
            
            # Add patient and tumor nodes
            patient_node = f"patient_{self.patient_id}"
            tumor_node = f"tumor_{self.patient_id}"
            
            # Simulate growing tumor
            tumor_volume = 10000.0 * (1.0 + 0.2 * (tp - 1))
            
            graph.add_node(patient_node, type='patient')
            graph.add_node(tumor_node, type='tumor', volume_mm3=tumor_volume)
            graph.add_edge(patient_node, tumor_node, relation='has_tumor')
            
            # Add treatment at timepoint 2
            if tp == 2:
                treatment_node = f"treatment_{tp}_{self.patient_id}"
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
            self.assertTrue(node.endswith(f"_tp1") or 
                          node.endswith(f"_tp2") or 
                          node.endswith(f"_tp3"),
                          f"Node {node} doesn't have timepoint suffix")
        
        # Verify temporal connections exist between same node types
        temporal_edges = 0
        for u, v, attrs in temporal_graph.edges(data=True):
            relation = attrs.get('relation', '')
            if 'progression' in relation or 'same_patient' in relation or 'temporal_sequence' in relation:
                temporal_edges += 1
                
                # Extract timepoints from node IDs
                u_tp = int(u.split('_tp')[-1])
                v_tp = int(v.split('_tp')[-1])
                
                # Verify timepoints are sequential
                self.assertEqual(v_tp, u_tp + 1, f"Non-sequential timepoints in edge {u} -> {v}")
        
        # Verify we have temporal edges
        self.assertGreater(temporal_edges, 0, "No temporal edges found")
    
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
        self.assertEqual(len(metrics['tumor_volumes']), 3)
        
        # Verify growth rates
        self.assertIn('growth_rates', metrics)
        self.assertEqual(len(metrics['growth_rates']), 2)  # 3 timepoints -> 2 transitions
        
        # Verify volume change metrics
        self.assertIn('initial_volume', metrics)
        self.assertIn('final_volume', metrics)
        self.assertIn('volume_change', metrics)
        self.assertGreater(metrics['volume_change'], 0)  # Should be growing
    
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
                     ['rapid_growth', 'steady_growth', 'stable', 'regression', 'variable'])
    
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
        
        # Verify that we don't have timepoint suffixes in the merged graph
        for node in merged_graph.nodes():
            self.assertFalse('_tp' in node, f"Node {node} still has timepoint suffix")


if __name__ == '__main__':
    unittest.main()