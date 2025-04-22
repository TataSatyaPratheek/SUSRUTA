# susruta/src/susruta/graph_builder/multimodal_graph.py
"""
Multimodal graph integration for glioma data analysis.

Combines graphs from MRI, clinical, and Excel data sources into a unified knowledge graph.
"""

from typing import Dict, List, Tuple, Any, Optional, Union, Set
import gc
import numpy as np
import networkx as nx
import pandas as pd

from ..data.clinical import ClinicalDataProcessor
from ..data.excel_loader import ExcelDataLoader
from ..data.excel_integration import MultimodalDataIntegrator
from ..graph.knowledge_graph import GliomaKnowledgeGraph
from ..utils.memory import MemoryTracker


class MultimodalGraphIntegrator:
    """Integrates graphs from multiple data modalities."""

    def __init__(self, memory_limit_mb: float = 3000):
        """
        Initialize multimodal graph integrator.

        Args:
            memory_limit_mb: Memory limit in MB
        """
        self.memory_limit_mb = memory_limit_mb
        self.memory_tracker = MemoryTracker(threshold_mb=memory_limit_mb)
        
        # Initialize data processors with memory allocations
        self.excel_loader = ExcelDataLoader(memory_limit_mb=memory_limit_mb * 0.2)
        self.clinical_processor = ClinicalDataProcessor()
        self.excel_integrator = MultimodalDataIntegrator(memory_limit_mb=memory_limit_mb * 0.3)
        
    def integrate_graphs(self,
                       mri_graph: nx.Graph,
                       clinical_graph: nx.MultiDiGraph,
                       excel_data: Optional[Dict[str, pd.DataFrame]] = None,
                       patient_id: Optional[int] = None) -> nx.MultiDiGraph:
        """
        Integrate graphs from multiple sources into a unified knowledge graph.

        Args:
            mri_graph: Graph built from MRI data
            clinical_graph: Graph built from clinical data
            excel_data: Optional dictionary of DataFrames from Excel sources
            patient_id: Optional patient ID to filter data

        Returns:
            Integrated MultiDiGraph
        """
        self.memory_tracker.log_memory("Starting graph integration")
        
        # Create a new MultiDiGraph to hold the integrated data
        # Use MultiDiGraph to match the GliomaKnowledgeGraph structure
        integrated_graph = nx.MultiDiGraph()
        
        # First, add all nodes and edges from the clinical graph
        integrated_graph.add_nodes_from(clinical_graph.nodes(data=True))
        for u, v, key, attr in clinical_graph.edges(data=True, keys=True):
            integrated_graph.add_edge(u, v, key, **attr)
        
        self.memory_tracker.log_memory("Added clinical graph nodes/edges")
        
        # Then add nodes and edges from the MRI graph
        # Convert to MultiDiGraph format
        for node, attrs in mri_graph.nodes(data=True):
            if node not in integrated_graph:
                integrated_graph.add_node(node, **attrs)
            else:
                # Merge attributes for existing nodes
                for key, value in attrs.items():
                    if key not in integrated_graph.nodes[node]:
                        integrated_graph.nodes[node][key] = value
        
        # Add edges from MRI graph with unique keys
        edge_index = clinical_graph.number_of_edges()
        for u, v, attr in mri_graph.edges(data=True):
            if not integrated_graph.has_edge(u, v):
                integrated_graph.add_edge(u, v, edge_index, **attr)
                edge_index += 1
            else:
                # Check if this exact relation already exists
                relation_exists = False
                for _, _, e_attr in integrated_graph.edges(data=True):
                    if e_attr.get('relation') == attr.get('relation'):
                        relation_exists = True
                        break
                if not relation_exists:
                    integrated_graph.add_edge(u, v, edge_index, **attr)
                    edge_index += 1
                    
        self.memory_tracker.log_memory("Added MRI graph nodes/edges")
        
        # Process Excel data if provided
        if excel_data is not None:
            integrated_graph = self._integrate_excel_data(integrated_graph, excel_data, patient_id)
            self.memory_tracker.log_memory("Integrated Excel data")
        
        # Connect nodes between modalities
        integrated_graph = self._connect_multimodal_nodes(integrated_graph, patient_id)
        self.memory_tracker.log_memory("Connected multimodal nodes")
        
        # Apply graph optimization techniques
        integrated_graph = self._optimize_graph(integrated_graph)
        self.memory_tracker.log_memory("Optimized graph")
        
        return integrated_graph
    
    def _integrate_excel_data(self,
                             graph: nx.MultiDiGraph,
                             excel_data: Dict[str, pd.DataFrame],
                             patient_id: Optional[int] = None) -> nx.MultiDiGraph:
        """
        Integrate Excel data into the graph.

        Args:
            graph: Existing graph to integrate with
            excel_data: Dictionary of Excel DataFrames
            patient_id: Optional patient ID to filter data

        Returns:
            Updated graph with Excel data integrated
        """
        # Filter Excel data for specific patient if requested
        if patient_id is not None:
            filtered_excel_data = {}
            for key, df in excel_data.items():
                # Skip empty DataFrames
                if df.empty:
                    filtered_excel_data[key] = df
                    continue
                
                # Check for patient ID column in each DataFrame
                id_cols = [col for col in df.columns if 'patient' in col.lower() or 'id' in col.lower()]
                if id_cols:
                    # Use the first matching column
                    id_col = id_cols[0]
                    # Convert patient_id to string for reliable matching
                    filtered_df = df[df[id_col].astype(str) == str(patient_id)]
                    filtered_excel_data[key] = filtered_df
                else:
                    # Keep original if no patient ID column found
                    filtered_excel_data[key] = df
            excel_data = filtered_excel_data
        
        # Process each Excel source
        scanner_data = excel_data.get('scanner', pd.DataFrame())
        segmentation_data = excel_data.get('segmentation', pd.DataFrame())
        
        # Add scanner nodes if data exists
        if not scanner_data.empty:
            for _, row in scanner_data.iterrows():
                patient_id_val = str(row.get('PatientID', ''))
                scanner_id = f"scanner_{patient_id_val}"
                
                # Create scanner attributes
                scanner_attrs = {
                    'type': 'scanner',
                    'manufacturer': row.get('ScannerManufacturer', None),
                    'model': row.get('ScannerModel', None),
                    'field_strength': row.get('FieldStrength', None),
                }
                # Filter out None values
                scanner_attrs = {k: v for k, v in scanner_attrs.items() if v is not None}
                
                # Add scanner node
                graph.add_node(scanner_id, **scanner_attrs)
                
                # Connect scanner to patient
                patient_node_id = f"patient_{patient_id_val}"
                if patient_node_id in graph:
                    graph.add_edge(scanner_id, patient_node_id, relation='scanned')
        
        # Add segmentation data if exists
        if not segmentation_data.empty:
            for _, row in segmentation_data.iterrows():
                # --- START FIX: Ensure consistent patient ID string format ---
                try:
                    # Convert to float first to handle potential '.0', then to int, then to str
                    patient_id_float = float(row.get('PatientID', ''))
                    patient_id_val = str(int(patient_id_float))
                except (ValueError, TypeError):
                    # Fallback if conversion fails (e.g., non-numeric ID)
                    patient_id_val = str(row.get('PatientID', ''))
                # --- END FIX ---

                tumor_node_id = f"tumor_{patient_id_val}"

                if tumor_node_id in graph:
                    for col in segmentation_data.columns:
                        if 'Volume' in col or 'Ratio' in col:
                            value = row.get(col)
                            if pd.notna(value):
                                attr_name = col.replace('_mm3', '').lower()
                                graph.nodes[tumor_node_id][attr_name] = float(value)
                else:
                    # --- START DIAGNOSTIC PRINTS ---
                    print(f"DEBUG: Tumor node '{tumor_node_id}' not found in graph.")
                    # --- END DIAGNOSTIC PRINTS ---
        else:
            # --- START DIAGNOSTIC PRINTS ---
            print(f"\nDEBUG: Segmentation data is empty after filtering for patient {patient_id}.")
            # --- END DIAGNOSTIC PRINTS ---

        return graph

    
    def _connect_multimodal_nodes(self, 
                                 graph: nx.MultiDiGraph,
                                 patient_id: Optional[int] = None) -> nx.MultiDiGraph:
        """
        Create connections between nodes from different modalities.

        Args:
            graph: Graph with nodes from multiple modalities
            patient_id: Optional patient ID to filter connections

        Returns:
            Graph with inter-modality connections
        """
        # Get lists of different node types
        patient_nodes = [n for n, attrs in graph.nodes(data=True) 
                       if attrs.get('type') == 'patient']
        tumor_nodes = [n for n, attrs in graph.nodes(data=True) 
                      if attrs.get('type') == 'tumor']
        region_nodes = [n for n, attrs in graph.nodes(data=True) 
                       if attrs.get('type') == 'region']
        sequence_nodes = [n for n, attrs in graph.nodes(data=True) 
                         if attrs.get('type') == 'sequence']
        treatment_nodes = [n for n, attrs in graph.nodes(data=True) 
                          if attrs.get('type') == 'treatment']
        
        # Filter by patient_id if specified
        if patient_id is not None:
            patient_id_str = str(patient_id)
            patient_nodes = [n for n in patient_nodes if patient_id_str in n]
            tumor_nodes = [n for n in tumor_nodes if patient_id_str in n]
            region_nodes = [n for n in region_nodes if patient_id_str in n]
            sequence_nodes = [n for n in sequence_nodes if patient_id_str in n]
            treatment_nodes = [n for n in treatment_nodes if patient_id_str in n]
        
        # Connect regions to sequences
        for region_node in region_nodes:
            patient_id_in_region = region_node.split('_')[-1]
            for sequence_node in sequence_nodes:
                if patient_id_in_region in sequence_node:
                    # Add edge from region to sequence
                    graph.add_edge(region_node, sequence_node, relation='visible_in')
        
        # Connect treatments to MRI sequences (to represent treatment effects)
        for treatment_node in treatment_nodes:
            # Extract patient ID from connected tumor
            tumor_found = False
            for u, v in graph.out_edges(treatment_node):
                if 'tumor_' in v:
                    tumor_found = True
                    patient_id_in_tumor = v.split('_')[1]
                    # Find corresponding sequences
                    for sequence_node in sequence_nodes:
                        if patient_id_in_tumor in sequence_node:
                            # Add edge from treatment to sequence
                            graph.add_edge(treatment_node, sequence_node, relation='affects')
            
            # If no tumor connection found, try to find patient through other means
            if not tumor_found:
                # This would require more complex traversal in a real implementation
                pass
        
        return graph
    
    def _optimize_graph(self, graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
        """
        Apply optimization techniques to the graph to improve memory efficiency.

        Args:
            graph: Graph to optimize

        Returns:
            Optimized graph
        """
        # 1. Convert attribute types to more memory-efficient formats
        for node, attrs in graph.nodes(data=True):
            for key, value in list(attrs.items()):
                if isinstance(value, (float, np.float64)):
                    attrs[key] = np.float32(value)
                elif isinstance(value, (int, np.int64)):
                    # Only convert if within range
                    if np.iinfo(np.int32).min <= value <= np.iinfo(np.int32).max:
                        attrs[key] = np.int32(value)
        
        # 2. Check edge attributes similarly
        for u, v, key, attrs in graph.edges(data=True, keys=True):
            for attr_key, value in list(attrs.items()):
                if isinstance(value, (float, np.float64)):
                    attrs[attr_key] = np.float32(value)
                elif isinstance(value, (int, np.int64)):
                    if np.iinfo(np.int32).min <= value <= np.iinfo(np.int32).max:
                        attrs[attr_key] = np.int32(value)
        
        # 3. Remove redundant edges
        redundant_edges = []
        seen_edges = set()
        
        for u, v, key, attrs in graph.edges(data=True, keys=True):
            edge_type = attrs.get('relation', 'default')
            edge_signature = (u, v, edge_type)
            
            if edge_signature in seen_edges:
                redundant_edges.append((u, v, key))
            else:
                seen_edges.add(edge_signature)
        
        # Remove identified redundant edges
        for u, v, key in redundant_edges:
            graph.remove_edge(u, v, key)
        
        # Force garbage collection
        gc.collect()
        
        return graph
    
    def from_excel_sources(self,
                          scanner_path: str,
                          clinical_path: str,
                          segmentation_path: str,
                          patient_id: Optional[int] = None,
                          timepoint: int = 1) -> nx.MultiDiGraph:
        """
        Build graph directly from Excel data sources.

        Args:
            scanner_path: Path to scanner metadata Excel file
            clinical_path: Path to clinical data Excel file
            segmentation_path: Path to segmentation volumes Excel file
            patient_id: Optional patient ID to filter data
            timepoint: Timepoint to filter for

        Returns:
            MultiDiGraph built from Excel data
        """
        self.memory_tracker.log_memory("Starting Excel-to-graph conversion")
        
        # Load all Excel data
        excel_data = self.excel_integrator.load_all_excel_data(
            scanner_path=scanner_path,
            clinical_path=clinical_path,
            segmentation_path=segmentation_path,
            timepoint=timepoint,
            force_reload=True
        )
        
        # Filter for specific patient if requested
        if patient_id is not None:
            for key, df in excel_data.items():
                if df.empty:
                    continue
                    
                # Find the patient ID column
                id_cols = [col for col in df.columns if 'patient' in col.lower() or 'id' in col.lower()]
                if id_cols:
                    id_col = id_cols[0]
                    # Ensure both are strings for comparison
                    excel_data[key] = df[df[id_col].astype(str) == str(patient_id)]
        
        # Process clinical data
        clinical_df = excel_data.get('clinical', pd.DataFrame())
        if clinical_df.empty:
            # Create minimal dataframe with patient ID if it was filtered out
            if patient_id is not None:
                clinical_df = pd.DataFrame({'patient_id': [patient_id]})
        
        # Preprocess and integrate with other Excel sources
        integrated_df = self.excel_integrator.integrate_with_clinical_data(
            clinical_df=clinical_df,
            excel_data_sources=excel_data,
            timepoint=timepoint
        )
        
        # Create a knowledge graph from the integrated data
        kg_builder = GliomaKnowledgeGraph(memory_limit_mb=self.memory_limit_mb * 0.5)
        kg_builder.add_clinical_data(integrated_df)
        
        # --- START FIX: Integrate scanner/segmentation data into the graph ---
        # Use the graph built so far and the original excel_data dictionary
        graph_with_excel = self._integrate_excel_data(
            graph=kg_builder.G,
            excel_data=excel_data, # Pass the originally loaded dict
            patient_id=patient_id
        )
        kg_builder.G = graph_with_excel # Update the graph in the builder
        # --- END FIX ---

        
        # Add treatments if data available
        # This requires a DataFrame with treatment info, not directly available from Excel
        # For demo purposes, we'll create a simple treatment dataset if needed
        if patient_id is not None:
            self._add_demo_treatments(kg_builder, patient_id)
        
        self.memory_tracker.log_memory("Completed Excel-to-graph conversion")
        return kg_builder.G
    
    def _add_demo_treatments(self, kg_builder: GliomaKnowledgeGraph, patient_id: int) -> None:
        """
        Add demo treatment data for testing purposes.

        Args:
            kg_builder: Knowledge graph builder
            patient_id: Patient ID to add treatments for
        """
        # Create a simple treatments dataframe
        treatments_df = pd.DataFrame([
            {
                'patient_id': patient_id,
                'treatment_id': 1,
                'category': 'surgery',
                'treatment_name': 'Gross Total Resection',
                'dose': None,
                'duration_days': 1,
                'start_day': 0,
                'response': 'complete'
            },
            {
                'patient_id': patient_id,
                'treatment_id': 2,
                'category': 'radiation',
                'treatment_name': 'External Beam Radiation',
                'dose': 60.0,
                'duration_days': 42,
                'start_day': 14,
                'response': 'partial'
            },
            {
                'patient_id': patient_id,
                'treatment_id': 3,
                'category': 'chemotherapy',
                'treatment_name': 'Temozolomide',
                'dose': 150.0,
                'duration_days': 180,
                'start_day': 14,
                'response': 'partial'
            }
        ])
        
        # Add to the knowledge graph
        kg_builder.add_treatments(treatments_df)