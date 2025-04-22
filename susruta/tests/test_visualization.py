# susruta/tests/graph_builder/test_visualization.py
"""
Tests for graph visualization component.
"""

import os
import unittest
import networkx as nx
import numpy as np
import tempfile
import shutil
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from susruta.graph_builder.visualization import GraphVisualizer


class TestGraphVisualizer(unittest.TestCase):
    """Test cases for GraphVisualizer."""

    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.output_dir = os.path.join(self.test_dir, 'output')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create visualizer instances for both backends
        self.plotly_visualizer = GraphVisualizer(output_dir=self.output_dir, use_plotly=True)
        self.mpl_visualizer = GraphVisualizer(output_dir=self.output_dir, use_plotly=False)
        
        # Create mock graph data
        self.patient_id = 1001
        self.create_mock_graph()
        self.create_mock_temporal_graph()
        self.create_mock_treatment_simulation()
    
    def tearDown(self):
        """Clean up temporary test data."""
        plt.close('all')  # Close any open Matplotlib figures
        shutil.rmtree(self.test_dir)
    
    def create_mock_graph(self):
        """Create mock graph for testing."""
        # Create a sample knowledge graph
        self.graph = nx.MultiDiGraph()
        
        # Add patient and tumor nodes
        patient_node = f"patient_{self.patient_id}"
        tumor_node = f"tumor_{self.patient_id}"
        
        self.graph.add_node(patient_node, type='patient', age=65, sex='male')
        self.graph.add_node(tumor_node, type='tumor', volume_mm3=15000.0, grade=4)
        self.graph.add_edge(patient_node, tumor_node, relation='has_tumor')
        
        # Add treatment nodes
        for i, category in enumerate(['surgery', 'radiation', 'chemotherapy']):
            treatment_node = f"treatment_{i+1}_{self.patient_id}"
            self.graph.add_node(treatment_node, type='treatment', category=category)
            self.graph.add_edge(tumor_node, treatment_node, relation='treated_with')
            
            # Add outcome node for treatment
            outcome_node = f"outcome_{i+1}_{self.patient_id}"
            self.graph.add_node(outcome_node, type='outcome', 
                               response='partial' if i < 2 else 'complete')
            self.graph.add_edge(treatment_node, outcome_node, relation='resulted_in')
        
        # Add sequence nodes
        for seq in ['t1c', 't2w']:
            seq_node = f"sequence_{seq}_{self.patient_id}"
            self.graph.add_node(seq_node, type='sequence', sequence=seq)
            self.graph.add_edge(tumor_node, seq_node, relation='has_sequence')
        
        # Add region nodes
        for i in range(3):
            region_node = f"region_{i+1}_{self.patient_id}"
            self.graph.add_node(region_node, type='region', 
                               volume_voxels=float(500*(i+1)),
                               center_x=float(10+i), center_y=float(12), center_z=float(8))
            self.graph.add_edge(tumor_node, region_node, relation='has_region')
        
        # Connect adjacent regions
        self.graph.add_edge(f"region_1_{self.patient_id}", f"region_2_{self.patient_id}", 
                           relation='adjacent_to')
        self.graph.add_edge(f"region_2_{self.patient_id}", f"region_3_{self.patient_id}", 
                           relation='adjacent_to')
    
    def create_mock_temporal_graph(self):
        """Create mock temporal graph for testing."""
        # Create a temporal graph with 3 timepoints
        self.temporal_graph = nx.MultiDiGraph()
        
        # Add nodes for each timepoint
        for tp in range(1, 4):
            # Progressively growing tumor
            volume = 10000.0 * (1.0 + 0.2 * (tp - 1))
            
            # Add patient node for this timepoint
            patient_node = f"patient_{self.patient_id}_tp{tp}"
            tumor_node = f"tumor_{self.patient_id}_tp{tp}"
            
            self.temporal_graph.add_node(patient_node, type='patient', timepoint=tp)
            self.temporal_graph.add_node(tumor_node, type='tumor', timepoint=tp, volume_mm3=volume)
            self.temporal_graph.add_edge(patient_node, tumor_node, relation='has_tumor')
            
            # Add treatment at timepoint 2
            if tp == 2:
                treatment_node = f"treatment_1_{self.patient_id}_tp{tp}"
                self.temporal_graph.add_node(treatment_node, type='treatment', 
                                           timepoint=tp, category='radiation')
                self.temporal_graph.add_edge(tumor_node, treatment_node, relation='treated_with')
        
        # Add temporal connections between patient nodes
        for tp in range(1, 3):
            self.temporal_graph.add_edge(
                f"patient_{self.patient_id}_tp{tp}",
                f"patient_{self.patient_id}_tp{tp+1}",
                relation='same_patient'
            )
            
            # Add temporal connections between tumor nodes
            self.temporal_graph.add_edge(
                f"tumor_{self.patient_id}_tp{tp}",
                f"tumor_{self.patient_id}_tp{tp+1}",
                relation='progression'
            )
    
    def create_mock_treatment_simulation(self):
        """Create mock treatment simulation results for testing."""
        self.treatment_simulation = {
            'option_0': {
                'config': {
                    'category': 'surgery',
                    'dose': None,
                    'duration': 1
                },
                'response_prob': 0.7,
                'survival_days': 450,
                'uncertainty': 0.1
            },
            'option_1': {
                'config': {
                    'category': 'radiation',
                    'dose': 60.0,
                    'duration': 30
                },
                'response_prob': 0.65,
                'survival_days': 425,
                'uncertainty': 0.15
            },
            'option_2': {
                'config': {
                    'category': 'chemotherapy',
                    'dose': 150.0,
                    'duration': 180
                },
                'response_prob': 0.8,
                'survival_days': 520,
                'uncertainty': 0.2
            }
        }
    
    def test_visualize_knowledge_graph_plotly(self):
        """Test knowledge graph visualization with Plotly."""
        # Visualize knowledge graph
        fig = self.plotly_visualizer.visualize_knowledge_graph(
            graph=self.graph,
            title="Test Knowledge Graph",
            save_path="test_knowledge_graph.html"
        )
        
        # Verify output
        self.assertIsInstance(fig, go.Figure)
        self.assertGreater(len(fig.data), 0)
        
        # Verify file was created
        output_path = os.path.join(self.output_dir, "test_knowledge_graph.html")
        self.assertTrue(os.path.exists(output_path))
    
    def test_visualize_knowledge_graph_matplotlib(self):
        """Test knowledge graph visualization with Matplotlib."""
        # Visualize knowledge graph
        fig = self.mpl_visualizer.visualize_knowledge_graph(
            graph=self.graph,
            title="Test Knowledge Graph",
            save_path="test_knowledge_graph.png"
        )
        
        # Verify output
        self.assertIsInstance(fig, plt.Figure)
        
        # Verify file was created
        output_path = os.path.join(self.output_dir, "test_knowledge_graph.png")
        self.assertTrue(os.path.exists(output_path))
    
    def test_visualize_tumor_progression_plotly(self):
        """Test tumor progression visualization with Plotly."""
        # Visualize tumor progression
        fig = self.plotly_visualizer.visualize_tumor_progression(
            temporal_graph=self.temporal_graph,
            patient_id=self.patient_id,
            save_path="test_progression.html"
        )
        
        # Verify output
        self.assertIsInstance(fig, go.Figure)
        self.assertGreater(len(fig.data), 0)
        
        # Verify file was created
        output_path = os.path.join(self.output_dir, "test_progression.html")
        self.assertTrue(os.path.exists(output_path))
    
    def test_visualize_tumor_progression_matplotlib(self):
        """Test tumor progression visualization with Matplotlib."""
        # Visualize tumor progression
        fig = self.mpl_visualizer.visualize_tumor_progression(
            temporal_graph=self.temporal_graph,
            patient_id=self.patient_id,
            save_path="test_progression.png"
        )
        
        # Verify output
        self.assertIsInstance(fig, plt.Figure)
        
        # Verify file was created
        output_path = os.path.join(self.output_dir, "test_progression.png")
        self.assertTrue(os.path.exists(output_path))
    
    def test_visualize_tumor_regions(self):
        """Test tumor region visualization."""
        # This test only applies to Plotly
        fig = self.plotly_visualizer.visualize_tumor_regions(
            graph=self.graph,
            patient_id=self.patient_id,
            save_path="test_regions.html"
        )
        
        # Verify output
        self.assertIsInstance(fig, go.Figure)
        
        # Verify file was created
        output_path = os.path.join(self.output_dir, "test_regions.html")
        self.assertTrue(os.path.exists(output_path))
    
    def test_visualize_treatment_comparison_plotly(self):
        """Test treatment comparison visualization with Plotly."""
        # Visualize treatment comparison
        fig = self.plotly_visualizer.visualize_treatment_comparison(
            treatment_simulation=self.treatment_simulation,
            patient_id=self.patient_id,
            save_path="test_treatment_comparison.html"
        )
        
        # Verify output
        self.assertIsInstance(fig, go.Figure)
        
        # Should have 2 subplots (response and survival)
        self.assertGreaterEqual(len(fig.data), 2)
        
        # Verify file was created
        output_path = os.path.join(self.output_dir, "test_treatment_comparison.html")
        self.assertTrue(os.path.exists(output_path))
    
    def test_visualize_treatment_comparison_matplotlib(self):
        """Test treatment comparison visualization with Matplotlib."""
        # Visualize treatment comparison
        fig = self.mpl_visualizer.visualize_treatment_comparison(
            treatment_simulation=self.treatment_simulation,
            patient_id=self.patient_id,
            save_path="test_treatment_comparison.png"
        )
        
        # Verify output
        self.assertIsInstance(fig, plt.Figure)
        
        # Verify file was created
        output_path = os.path.join(self.output_dir, "test_treatment_comparison.png")
        self.assertTrue(os.path.exists(output_path))


if __name__ == '__main__':
    unittest.main()