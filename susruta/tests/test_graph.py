"""Tests for the knowledge graph module."""

import os
import pytest
import numpy as np
import pandas as pd
import networkx as nx
import torch
from torch_geometric.data import HeteroData

from susruta.graph import GliomaKnowledgeGraph


class TestGliomaKnowledgeGraph:
    """Test suite for the GliomaKnowledgeGraph class."""
    
    def test_initialization(self):
        """Test initialization of GliomaKnowledgeGraph."""
        kg = GliomaKnowledgeGraph(memory_limit_mb=3000)
        
        assert kg.memory_limit_mb == 3000
        assert hasattr(kg, 'memory_tracker')
        assert hasattr(kg, 'G')
        assert isinstance(kg.G, nx.MultiDiGraph)
    
    def test_add_clinical_data(self, synthetic_clinical_data):
        """Test adding clinical data to the graph."""
        kg = GliomaKnowledgeGraph()
        
        # Add clinical data
        kg.add_clinical_data(synthetic_clinical_data, batch_size=5)
        
        # Check that patient nodes were added
        patient_nodes = [node for node, attrs in kg.G.nodes(data=True) 
                        if attrs.get('type') == 'patient']
        assert len(patient_nodes) == len(synthetic_clinical_data)
        
        # Check that tumor nodes were added
        tumor_nodes = [node for node, attrs in kg.G.nodes(data=True) 
                      if attrs.get('type') == 'tumor']
        assert len(tumor_nodes) == len(synthetic_clinical_data)
        
        # Check patient-tumor connections
        for patient_id in synthetic_clinical_data['patient_id']:
            patient_node = f"patient_{patient_id}"
            tumor_node = f"tumor_{patient_id}"
            
            assert patient_node in kg.G
            assert tumor_node in kg.G
            assert kg.G.has_edge(patient_node, tumor_node)
            assert kg.G[patient_node][tumor_node][0]['relation'] == 'has_tumor'
    
    def test_add_imaging_features(self, synthetic_clinical_data, synthetic_imaging_features):
        """Test adding imaging features to the graph."""
        kg = GliomaKnowledgeGraph()
        
        # First add clinical data to create tumor nodes
        kg.add_clinical_data(synthetic_clinical_data)
        
        # Add imaging features
        kg.add_imaging_features(synthetic_imaging_features, max_features_per_node=10)
        
        # Check that feature nodes were added
        feature_nodes = [node for node, attrs in kg.G.nodes(data=True) 
                        if attrs.get('type') == 'feature']
        assert len(feature_nodes) > 0
        
        # Check tumor-feature connections for the first patient
        patient_id = synthetic_clinical_data['patient_id'].iloc[0]
        tumor_node = f"tumor_{patient_id}"
        
        feature_edges = [(u, v) for u, v, attrs in kg.G.out_edges(tumor_node, data=True) 
                        if attrs.get('relation') == 'has_feature']
        assert len(feature_edges) > 0
    
    def test_add_treatments(self, synthetic_clinical_data, synthetic_treatment_data):
        """Test adding treatments to the graph."""
        kg = GliomaKnowledgeGraph()
        
        # First add clinical data to create tumor nodes
        kg.add_clinical_data(synthetic_clinical_data)
        
        # Add treatments
        kg.add_treatments(synthetic_treatment_data, batch_size=5)
        
        # Check that treatment nodes were added
        treatment_nodes = [node for node, attrs in kg.G.nodes(data=True) 
                          if attrs.get('type') == 'treatment']
        assert len(treatment_nodes) == len(synthetic_treatment_data)
        
        # Check tumor-treatment and treatment-outcome connections
        for _, row in synthetic_treatment_data.iterrows():
            patient_id = row['patient_id']
            treatment_id = row['treatment_id']
            
            tumor_node = f"tumor_{patient_id}"
            treatment_node = f"treatment_{treatment_id}"
            outcome_node = f"outcome_{patient_id}_{treatment_id}"
            
            assert treatment_node in kg.G
            assert kg.G.has_edge(tumor_node, treatment_node)
            assert kg.G[tumor_node][treatment_node][0]['relation'] == 'treated_with'
            
            if 'response' in row:
                assert outcome_node in kg.G
                assert kg.G.has_edge(treatment_node, outcome_node)
                assert kg.G[treatment_node][outcome_node][0]['relation'] == 'resulted_in'
    
    def test_add_similarity_edges(self, synthetic_clinical_data, synthetic_imaging_features):
        """Test adding similarity edges between tumors."""
        kg = GliomaKnowledgeGraph()
        
        # Add clinical data and imaging features
        kg.add_clinical_data(synthetic_clinical_data)
        kg.add_imaging_features(synthetic_imaging_features)
        
        # Add similarity edges
        kg.add_similarity_edges(threshold=0.0, max_edges_per_node=3)
        
        # Check that similarity edges were added
        similarity_edges = [(u, v) for u, v, attrs in kg.G.edges(data=True) 
                           if attrs.get('relation') == 'similar_to']
        
        # At least some tumors should have similarity edges
        assert len(similarity_edges) > 0
        
        # Each tumor should have at most max_edges_per_node similarity edges
        for tumor_node in [f"tumor_{id}" for id in synthetic_clinical_data['patient_id']]:
            outgoing_similarity = [(u, v) for u, v, attrs in kg.G.out_edges(tumor_node, data=True) 
                                 if attrs.get('relation') == 'similar_to']
            assert len(outgoing_similarity) <= 3
    
    def test_get_tumor_features(self, knowledge_graph):
        """Test extracting features for a tumor node."""
        kg = GliomaKnowledgeGraph()
        kg.G = knowledge_graph  # Use the fixture graph
        
        # Get a tumor node
        tumor_nodes = [node for node, attrs in kg.G.nodes(data=True) 
                      if attrs.get('type') == 'tumor']
        
        if tumor_nodes:
            tumor_node = tumor_nodes[0]
            features = kg._get_tumor_features(tumor_node)
            
            # Should return a dictionary with features
            assert isinstance(features, dict)
            assert len(features) > 0
            
            # Features should include tumor attributes
            for key in kg.G.nodes[tumor_node]:
                if key != 'type' and key != 'name':
                    assert f"tumor_{key}" in features
    
    def test_calculate_similarity(self):
        """Test calculating similarity between feature dictionaries."""
        kg = GliomaKnowledgeGraph()
        
        # Test with identical features
        features1 = {'a': 1.0, 'b': 2.0, 'c': 3.0}
        features2 = {'a': 1.0, 'b': 2.0, 'c': 3.0}
        similarity = kg._calculate_similarity(features1, features2)
        assert similarity == 1.0
        
        # Test with orthogonal features
        features1 = {'a': 1.0, 'b': 0.0, 'c': 0.0}
        features2 = {'a': 0.0, 'b': 1.0, 'c': 0.0}
        similarity = kg._calculate_similarity(features1, features2)
        assert similarity == 0.0
        
        # Test with partial overlap
        features1 = {'a': 1.0, 'b': 1.0, 'c': 0.0}
        features2 = {'a': 1.0, 'b': 0.0, 'c': 1.0}
        similarity = kg._calculate_similarity(features1, features2)
        assert 0.0 < similarity < 1.0
        
        # Test with no common features
        features1 = {'a': 1.0, 'b': 2.0}
        features2 = {'c': 3.0, 'd': 4.0}
        similarity = kg._calculate_similarity(features1, features2)
        assert similarity == 0.0
    
    def test_compress_graph(self, knowledge_graph):
        """Test graph compression function."""
        kg = GliomaKnowledgeGraph()
        kg.G = knowledge_graph.copy()  # Use a copy of the fixture graph
        
        # Add some float64 values to test compression
        for node in list(kg.G.nodes()):
            kg.G.nodes[node]['test_float'] = np.float64(1.234)
            kg.G.nodes[node]['test_int'] = np.int64(42)
        
        # Compress the graph
        kg._compress_graph()
        
        # Check that values were converted to smaller types
        for node in kg.G.nodes():
            if 'test_float' in kg.G.nodes[node]:
                assert isinstance(kg.G.nodes[node]['test_float'], np.float32)
            if 'test_int' in kg.G.nodes[node]:
                assert isinstance(kg.G.nodes[node]['test_int'], np.int32)
    
    def test_to_pytorch_geometric(self, knowledge_graph):
        """Test conversion to PyTorch Geometric format."""
        kg = GliomaKnowledgeGraph()
        kg.G = knowledge_graph  # Use the fixture graph
        
        # Convert to PyG
        pyg_data = kg.to_pytorch_geometric()
        
        # Check that the result is a HeteroData object
        assert isinstance(pyg_data, HeteroData)
        
        # Check that node types from the graph are present in PyG data
        node_types = set()
        for _, attrs in kg.G.nodes(data=True):
            if 'type' in attrs:
                node_types.add(attrs['type'])
        
        for node_type in node_types:
            assert node_type in pyg_data.node_types
        
        # Check that edge types from the graph are present in PyG data
        edge_types = set()
        for u, v, attrs in kg.G.edges(data=True):
            if 'relation' in attrs:
                u_type = kg.G.nodes[u].get('type', 'unknown')
                v_type = kg.G.nodes[v].get('type', 'unknown')
                edge_types.add((u_type, attrs['relation'], v_type))
        
        for edge_type in edge_types:
            # Some edge types might be filtered out during conversion
            # (e.g., if source or target nodes were filtered)
            if edge_type[0] in pyg_data.node_types and edge_type[2] in pyg_data.node_types:
                assert edge_type in pyg_data.edge_types or edge_type not in edge_types
    
    def test_get_statistics(self, knowledge_graph):
        """Test getting graph statistics."""
        kg = GliomaKnowledgeGraph()
        kg.G = knowledge_graph  # Use the fixture graph
        
        stats = kg.get_statistics()
        
        # Check that the statistics dictionary contains expected keys
        assert 'total_nodes' in stats
        assert 'total_edges' in stats
        assert 'node_types' in stats
        assert 'edge_types' in stats
        assert 'density' in stats
        assert 'memory_usage_mb' in stats
        
        # Check that the statistics are consistent with the graph
        assert stats['total_nodes'] == kg.G.number_of_nodes()
        assert stats['total_edges'] == kg.G.number_of_edges()
        
        # Check that node types are counted correctly
        for node_type, count in stats['node_types'].items():
            nodes_of_type = [n for n, attrs in kg.G.nodes(data=True) 
                           if attrs.get('type') == node_type]
            assert count == len(nodes_of_type)
        
        # Check that edge types are counted correctly
        for edge_type, count in stats['edge_types'].items():
            edges_of_type = [(u, v) for u, v, attrs in kg.G.edges(data=True) 
                           if attrs.get('relation') == edge_type]
            assert count == len(edges_of_type)