# susruta/src/susruta/graph_builder/visualization.py
"""
Visualization tools for glioma knowledge graphs.

Provides comprehensive 2D and 3D visualization tools for graph-based
representations of glioma data, including brain visualizations.
"""

from typing import Dict, List, Tuple, Any, Optional, Union, Set
import os
import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import nibabel as nib
from pathlib import Path


class GraphVisualizer:
    """Visualization tools for glioma knowledge graphs."""

    def __init__(self, 
                output_dir: Optional[str] = None, 
                use_plotly: bool = True):
        """
        Initialize graph visualizer.

        Args:
            output_dir: Optional directory to save visualizations
            use_plotly: Whether to use Plotly (True) or Matplotlib (False)
        """
        self.output_dir = output_dir
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        self.use_plotly = use_plotly
    
    def visualize_knowledge_graph(self,
                                graph: nx.MultiDiGraph,
                                title: str = "Glioma Knowledge Graph",
                                node_types_to_include: Optional[List[str]] = None,
                                color_by: str = 'type',
                                save_path: Optional[str] = None) -> Union[plt.Figure, go.Figure]:
        """
        Visualize knowledge graph structure.

        Args:
            graph: Knowledge graph to visualize
            title: Plot title
            node_types_to_include: Optional list of node types to include
            color_by: Node attribute to use for coloring
            save_path: Optional path to save visualization

        Returns:
            Matplotlib or Plotly figure
        """
        # Filter nodes by type if specified
        if node_types_to_include:
            nodes_to_keep = [
                node for node, attrs in graph.nodes(data=True)
                if attrs.get('type') in node_types_to_include
            ]
            subgraph = graph.subgraph(nodes_to_keep)
        else:
            subgraph = graph
        
        if self.use_plotly:
            return self._plotly_knowledge_graph(subgraph, title, color_by, save_path)
        else:
            return self._matplotlib_knowledge_graph(subgraph, title, color_by, save_path)
    
    def _plotly_knowledge_graph(self,
                               graph: nx.MultiDiGraph,
                               title: str,
                               color_by: str,
                               save_path: Optional[str] = None) -> go.Figure:
        """
        Visualize knowledge graph using Plotly.

        Args:
            graph: Knowledge graph to visualize
            title: Plot title
            color_by: Node attribute to use for coloring
            save_path: Optional path to save visualization

        Returns:
            Plotly figure
        """
        # Use networkx spring layout
        pos = nx.spring_layout(graph, seed=42)
        
        # Get node attributes for coloring
        node_types = {}
        for node in graph.nodes():
            attrs = graph.nodes[node]
            if color_by in attrs:
                node_types[node] = attrs[color_by]
            else:
                node_types[node] = 'unknown'
        
        unique_types = list(set(node_types.values()))
        colormap = px.colors.qualitative.Plotly
        color_dict = {t: colormap[i % len(colormap)] for i, t in enumerate(unique_types)}
        
        # Create node traces by type
        node_traces = []
        for node_type in unique_types:
            nodes = [node for node, nt in node_types.items() if nt == node_type]
            if not nodes:
                continue
                
            x_vals = [pos[node][0] for node in nodes]
            y_vals = [pos[node][1] for node in nodes]
            
            # Create node labels
            node_labels = []
            node_sizes = []
            for node in nodes:
                # Use node base name (strip type prefix)
                if '_' in node:
                    parts = node.split('_', 1)
                    if len(parts) > 1:
                        label = parts[0] + '<br>' + parts[1]
                    else:
                        label = node
                else:
                    label = node
                
                # Add key attributes to label
                attrs = graph.nodes[node]
                attr_str = '<br>'.join([f"{k}: {v}" for k, v in attrs.items() 
                                    if k not in ['type', 'id'] and str(v) != 'nan'])
                if attr_str:
                    label += f"<br>{attr_str}"
                
                node_labels.append(label)
                
                # Node size based on connections
                size = 10 + 2 * (graph.degree(node))
                node_sizes.append(size)
            
            node_trace = go.Scatter(
                x=x_vals,
                y=y_vals,
                mode='markers',
                marker=dict(
                    size=node_sizes,
                    color=color_dict[node_type],
                    line=dict(width=1, color='#000000')
                ),
                text=node_labels,
                hoverinfo='text',
                name=node_type
            )
            node_traces.append(node_trace)
        
        # Create edge traces
        edge_traces = []
        for u, v, attrs in graph.edges(data=True):
            relation = attrs.get('relation', 'default')
            
            # Create a unique trace for each relation type if not already created
            trace_exists = False
            for trace in edge_traces:
                if trace.name == relation:
                    trace_exists = True
                    break
            
            if not trace_exists:
                edge_x = []
                edge_y = []
                edge_text = []
                
                # Find all edges with this relation
                for src, dst, e_attrs in graph.edges(data=True):
                    if e_attrs.get('relation', 'default') == relation and src in pos and dst in pos:
                        x0, y0 = pos[src]
                        x1, y1 = pos[dst]
                        edge_x.extend([x0, x1, None])
                        edge_y.extend([y0, y1, None])
                        
                        # Create edge label
                        src_label = src.split('_')[0] if '_' in src else src
                        dst_label = dst.split('_')[0] if '_' in dst else dst
                        label = f"{src_label} → {dst_label}<br>{relation}"
                        edge_text.append(label)
                
                edge_trace = go.Scatter(
                    x=edge_x,
                    y=edge_y,
                    mode='lines',
                    hoverinfo='text',
                    text=edge_text,
                    line=dict(width=1),
                    opacity=0.7,
                    name=relation
                )
                edge_traces.append(edge_trace)
        
        # Create figure
        fig = go.Figure(
            data=edge_traces + node_traces,
            layout=go.Layout(
                title=title,
                showlegend=True,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                annotations=[dict(
                    text="Glioma Knowledge Graph",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002)],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=800,
                legend=dict(x=1.05, y=0.5)
            )
        )
        
        # Save figure if path provided
        if save_path:
            output_path = save_path if os.path.dirname(save_path) else os.path.join(self.output_dir or '.', save_path)
            fig.write_html(output_path)
        
        return fig
    
    def _matplotlib_knowledge_graph(self,
                                   graph: nx.MultiDiGraph,
                                   title: str,
                                   color_by: str,
                                   save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize knowledge graph using Matplotlib.

        Args:
            graph: Knowledge graph to visualize
            title: Plot title
            color_by: Node attribute to use for coloring
            save_path: Optional path to save visualization

        Returns:
            Matplotlib figure
        """
        # Get node attributes for coloring
        node_types = {}
        for node in graph.nodes():
            attrs = graph.nodes[node]
            if color_by in attrs:
                node_types[node] = attrs[color_by]
            else:
                node_types[node] = 'unknown'
        
        unique_types = list(set(node_types.values()))
        colormap = plt.cm.tab10
        color_dict = {t: colormap(i % 10) for i, t in enumerate(unique_types)}
        
        # Create figure
        plt.figure(figsize=(12, 10))
        
        # Create position layout
        pos = nx.spring_layout(graph, seed=42)
        
        # Draw nodes by type
        for node_type in unique_types:
            nodes = [node for node, nt in node_types.items() if nt == node_type]
            if not nodes:
                continue
                
            nx.draw_networkx_nodes(
                graph, pos,
                nodelist=nodes,
                node_color=[color_dict[node_type]] * len(nodes),
                node_size=[100 + 10 * graph.degree(node) for node in nodes],
                alpha=0.8,
                label=node_type
            )
        
        # Draw edges
        nx.draw_networkx_edges(
            graph, pos,
            width=1.0,
            alpha=0.5,
            arrows=True,
            arrowstyle='-|>',
            arrowsize=10
        )
        
        # Draw labels with smaller font
        nx.draw_networkx_labels(
            graph, pos,
            font_size=8,
            font_color='black'
        )
        
        plt.title(title)
        plt.legend()
        plt.axis('off')
        plt.tight_layout()
        
        # Save figure if path provided
        if save_path:
            output_path = save_path if os.path.dirname(save_path) else os.path.join(self.output_dir or '.', save_path)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def visualize_tumor_progression(self,
                                  temporal_graph: nx.MultiDiGraph,
                                  patient_id: int,
                                  save_path: Optional[str] = None) -> Union[plt.Figure, go.Figure]:
        """
        Visualize tumor progression over time.

        Args:
            temporal_graph: Temporal graph with multiple timepoints
            patient_id: Patient identifier
            save_path: Optional path to save visualization

        Returns:
            Matplotlib or Plotly figure
        """
        from .temporal_graph import TemporalGraphBuilder
        
        # Extract progression metrics
        temporal_builder = TemporalGraphBuilder()
        metrics = temporal_builder.compute_progression_metrics(temporal_graph, patient_id)
        
        # Get volumes and timepoints
        tumor_volumes = metrics.get('tumor_volumes', {})
        timepoints = sorted(tumor_volumes.keys())
        
        if not timepoints:
            # Create empty plot with message
            if self.use_plotly:
                fig = go.Figure()
                fig.add_annotation(
                    text="No tumor volume data available",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5,
                    showarrow=False,
                    font=dict(size=16)
                )
                return fig
            else:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.text(0.5, 0.5, "No tumor volume data available",
                        horizontalalignment='center',
                        verticalalignment='center',
                        transform=ax.transAxes,
                        fontsize=16)
                return fig
        
        # Get treatment timepoints
        treatment_timepoints = {}
        for node, attrs in temporal_graph.nodes(data=True):
            if attrs.get('type') == 'treatment' and f"{patient_id}" in node:
                timepoint = attrs.get('timepoint')
                if timepoint is not None:
                    treatment_category = attrs.get('category', 'unknown')
                    if timepoint not in treatment_timepoints:
                        treatment_timepoints[timepoint] = []
                    treatment_timepoints[timepoint].append(treatment_category)
        
        # Generate volume plot
        if self.use_plotly:
            return self._plotly_progression(timepoints, tumor_volumes, treatment_timepoints, patient_id, save_path)
        else:
            return self._matplotlib_progression(timepoints, tumor_volumes, treatment_timepoints, patient_id, save_path)
    
    def _plotly_progression(self,
                           timepoints: List[int],
                           tumor_volumes: Dict[int, float],
                           treatment_timepoints: Dict[int, List[str]],
                           patient_id: int,
                           save_path: Optional[str] = None) -> go.Figure:
        """
        Create Plotly visualization of tumor progression.

        Args:
            timepoints: List of timepoints
            tumor_volumes: Dictionary of tumor volumes by timepoint
            treatment_timepoints: Dictionary of treatments by timepoint
            patient_id: Patient identifier
            save_path: Optional path to save visualization

        Returns:
            Plotly figure
        """
        # Create volume trace
        volumes = [tumor_volumes[tp] for tp in timepoints]
        volume_trace = go.Scatter(
            x=timepoints,
            y=volumes,
            mode='lines+markers',
            name='Tumor Volume',
            line=dict(color='firebrick', width=3),
            marker=dict(size=10)
        )
        
        # Create figure
        fig = go.Figure(data=[volume_trace])
        
        # Add treatment markers
        for tp, treatments in treatment_timepoints.items():
            if tp not in tumor_volumes:
                continue
                
            volume_at_tp = tumor_volumes[tp]
            
            # Get unique treatment categories
            unique_treatments = list(set(treatments))
            treatment_label = '<br>'.join(unique_treatments)
            
            fig.add_annotation(
                x=tp,
                y=volume_at_tp,
                text=treatment_label,
                showarrow=True,
                arrowhead=5,
                ax=0,
                ay=-40
            )
        
        # Customize layout
        fig.update_layout(
            title=f"Tumor Progression for Patient {patient_id}",
            xaxis_title="Timepoint",
            yaxis_title="Tumor Volume (mm³)",
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Add growth rate annotations
        if len(timepoints) > 1:
            growth_rates = []
            for i in range(len(timepoints) - 1):
                tp1 = timepoints[i]
                tp2 = timepoints[i + 1]
                vol1 = tumor_volumes[tp1]
                vol2 = tumor_volumes[tp2]
                
                if vol1 > 0:
                    growth_rate = (vol2 - vol1) / vol1 * 100  # Percentage
                    growth_rates.append(growth_rate)
                    
                    # Add growth rate annotation
                    avg_x = (tp1 + tp2) / 2
                    avg_y = (vol1 + vol2) / 2
                    
                    # Format growth rate string
                    growth_str = f"{growth_rate:.1f}%" if growth_rate >= 0 else f"{growth_rate:.1f}%"
                    color = "red" if growth_rate > 0 else "green"
                    
                    fig.add_annotation(
                        x=avg_x,
                        y=avg_y,
                        text=growth_str,
                        showarrow=False,
                        font=dict(size=12, color=color)
                    )
        
        # Save figure if path provided
        if save_path:
            output_path = save_path if os.path.dirname(save_path) else os.path.join(self.output_dir or '.', save_path)
            fig.write_html(output_path)
        
        return fig
    
    def _matplotlib_progression(self,
                               timepoints: List[int],
                               tumor_volumes: Dict[int, float],
                               treatment_timepoints: Dict[int, List[str]],
                               patient_id: int,
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Create Matplotlib visualization of tumor progression.

        Args:
            timepoints: List of timepoints
            tumor_volumes: Dictionary of tumor volumes by timepoint
            treatment_timepoints: Dictionary of treatments by timepoint
            patient_id: Patient identifier
            save_path: Optional path to save visualization

        Returns:
            Matplotlib figure
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot tumor volumes
        volumes = [tumor_volumes[tp] for tp in timepoints]
        ax.plot(timepoints, volumes, 'o-', color='firebrick', linewidth=2, markersize=8)
        
        # Add growth rates
        if len(timepoints) > 1:
            for i in range(len(timepoints) - 1):
                tp1 = timepoints[i]
                tp2 = timepoints[i + 1]
                vol1 = tumor_volumes[tp1]
                vol2 = tumor_volumes[tp2]
                
                if vol1 > 0:
                    growth_rate = (vol2 - vol1) / vol1 * 100  # Percentage
                    
                    # Calculate position for annotation
                    avg_x = (tp1 + tp2) / 2
                    avg_y = (vol1 + vol2) / 2
                    
                    # Format growth rate string
                    growth_str = f"{growth_rate:.1f}%" if growth_rate >= 0 else f"{growth_rate:.1f}%"
                    color = "red" if growth_rate > 0 else "green"
                    
                    ax.annotate(
                        growth_str,
                        xy=(avg_x, avg_y),
                        color=color,
                        fontsize=10,
                        ha='center'
                    )
        
        # Add treatment markers
        for tp, treatments in treatment_timepoints.items():
            if tp not in tumor_volumes:
                continue
                
            volume_at_tp = tumor_volumes[tp]
            
            # Get unique treatment categories
            unique_treatments = list(set(treatments))
            treatment_label = '\n'.join(unique_treatments)
            
            ax.annotate(
                treatment_label,
                xy=(tp, volume_at_tp),
                xytext=(0, -30),
                textcoords='offset points',
                arrowprops=dict(arrowstyle='->'),
                fontsize=9,
                ha='center'
            )
        
        # Customize plot
        ax.set_title(f"Tumor Progression for Patient {patient_id}")
        ax.set_xlabel("Timepoint")
        ax.set_ylabel("Tumor Volume (mm³)")
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Set x-axis ticks to be integers
        ax.set_xticks(timepoints)
        
        plt.tight_layout()
        
        # Save figure if path provided
        if save_path:
            output_path = save_path if os.path.dirname(save_path) else os.path.join(self.output_dir or '.', save_path)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def visualize_brain_with_tumor(self,
                                  mri_path: str,
                                  mask_path: str,
                                  patient_id: int,
                                  timepoint: int = 1,
                                  save_path: Optional[str] = None) -> go.Figure:
        """
        Create 3D visualization of brain with tumor.

        Args:
            mri_path: Path to MRI NIfTI file
            mask_path: Path to tumor mask NIfTI file
            patient_id: Patient identifier
            timepoint: Timepoint number
            save_path: Optional path to save visualization

        Returns:
            Plotly figure with 3D visualization
        """
        # Only implemented for Plotly
        if not self.use_plotly:
            print("Warning: 3D brain visualization only available with Plotly.")
            self.use_plotly = True
        
        # Load MRI and mask
        try:
            mri_img = nib.load(mri_path)
            mask_img = nib.load(mask_path)
            
            mri_data = mri_img.get_fdata()
            mask_data = mask_img.get_fdata()
        except Exception as e:
            print(f"Error loading MRI/mask data: {e}")
            # Create empty figure with error message
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error loading MRI data: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=14)
            )
            return fig
        
        # Create figure with 4 subplots - 3 slices + 3D view
        fig = make_subplots(
            rows=2, cols=2,
            specs=[
                [{"type": "surface"}, {"type": "heatmap"}],
                [{"type": "heatmap"}, {"type": "heatmap"}]
            ],
            subplot_titles=["3D Surface", "Axial Slice", "Coronal Slice", "Sagittal Slice"]
        )
        
        # Get dimensions
        nx, ny, nz = mri_data.shape
        
        # 3D Surface Visualization - Tumor only
        try:
            # Create isosurface of tumor
            from skimage import measure
            
            # Generate surface vertices and triangles at mask boundary
            verts, faces, _, _ = measure.marching_cubes(mask_data, level=0.5)
            
            # Create 3D mesh
            x, y, z = verts.T
            i, j, k = faces.T
            
            # Create mesh3d trace
            fig.add_trace(
                go.Mesh3d(
                    x=z, y=y, z=x,  # Swap x and z for radiology convention
                    i=i, j=j, k=k,
                    opacity=0.5,
                    color='red',
                    name='Tumor'
                ),
                row=1, col=1
            )
            
            # Add brain outline
            # Downsample brain for performance
            brain_mask = mri_data > np.percentile(mri_data, 50)  # Threshold at 50th percentile
            
            # Downsample for performance
            downsample = 4
            brain_mask_small = brain_mask[::downsample, ::downsample, ::downsample]
            
            # Generate surface
            verts, faces, _, _ = measure.marching_cubes(brain_mask_small, level=0.5)
            
            # Scale vertices back to original size
            verts = verts * downsample
            
            # Create mesh3d trace for brain outline
            x, y, z = verts.T
            i, j, k = faces.T
            
            fig.add_trace(
                go.Mesh3d(
                    x=z, y=y, z=x,  # Swap x and z for radiology convention
                    i=i, j=j, k=k,
                    opacity=0.1,
                    color='gray',
                    name='Brain Outline'
                ),
                row=1, col=1
            )
        except Exception as e:
            print(f"Error generating 3D surface: {e}")
            # Add text note instead
            fig.add_annotation(
                text="3D rendering failed",
                xref="x domain", yref="y domain",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=14),
                row=1, col=1
            )
        
        # Find center of tumor for slice positioning
        tumor_indices = np.where(mask_data > 0)
        if len(tumor_indices[0]) > 0:
            center_x = int(np.median(tumor_indices[0]))
            center_y = int(np.median(tumor_indices[1]))
            center_z = int(np.median(tumor_indices[2]))
        else:
            # Default to image center if no tumor
            center_x, center_y, center_z = nx // 2, ny // 2, nz // 2
        
        # Create 2D slices through tumor center
        
        # Axial slice (top-down view)
        axial_slice = mri_data[:, :, center_z].T
        axial_mask = mask_data[:, :, center_z].T
        
        # Create combined image with tumor overlay
        axial_img = np.zeros((*axial_slice.shape, 3), dtype=np.uint8)
        # Normalize MRI data to 0-255
        norm_slice = (axial_slice - axial_slice.min()) / (axial_slice.max() - axial_slice.min()) * 255
        # Create RGB channels
        for i in range(3):
            axial_img[:, :, i] = norm_slice.astype(np.uint8)
        # Overlay tumor in red
        axial_img[:, :, 0][axial_mask > 0] = 255
        axial_img[:, :, 1][axial_mask > 0] = 0
        axial_img[:, :, 2][axial_mask > 0] = 0
        
        fig.add_trace(
            go.Image(z=axial_img),
            row=1, col=2
        )
        
        # Coronal slice (front view)
        coronal_slice = mri_data[:, center_y, :].T
        coronal_mask = mask_data[:, center_y, :].T
        
        # Create combined image with tumor overlay
        coronal_img = np.zeros((*coronal_slice.shape, 3), dtype=np.uint8)
        # Normalize MRI data to 0-255
        norm_slice = (coronal_slice - coronal_slice.min()) / (coronal_slice.max() - coronal_slice.min()) * 255
        # Create RGB channels
        for i in range(3):
            coronal_img[:, :, i] = norm_slice.astype(np.uint8)
        # Overlay tumor in red
        coronal_img[:, :, 0][coronal_mask > 0] = 255
        coronal_img[:, :, 1][coronal_mask > 0] = 0
        coronal_img[:, :, 2][coronal_mask > 0] = 0
        
        fig.add_trace(
            go.Image(z=coronal_img),
            row=2, col=1
        )
        
        # Sagittal slice (side view)
        sagittal_slice = mri_data[center_x, :, :].T
        sagittal_mask = mask_data[center_x, :, :].T
        
        # Create combined image with tumor overlay
        sagittal_img = np.zeros((*sagittal_slice.shape, 3), dtype=np.uint8)
        # Normalize MRI data to 0-255
        norm_slice = (sagittal_slice - sagittal_slice.min()) / (sagittal_slice.max() - sagittal_slice.min()) * 255
        # Create RGB channels
        for i in range(3):
            sagittal_img[:, :, i] = norm_slice.astype(np.uint8)
        # Overlay tumor in red
        sagittal_img[:, :, 0][sagittal_mask > 0] = 255
        sagittal_img[:, :, 1][sagittal_mask > 0] = 0
        sagittal_img[:, :, 2][sagittal_mask > 0] = 0
        
        fig.add_trace(
            go.Image(z=sagittal_img),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title=f"Brain MRI Visualization for Patient {patient_id}, Timepoint {timepoint}",
            height=800,
            width=1000,
            margin=dict(l=0, r=0, b=0, t=40),
            showlegend=False
        )
        
        # Turn off axis for image plots
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        
        # Set aspect ratio for 3D plot
        fig.update_scenes(
            aspectmode='data'
        )
        
        # Save figure if path provided
        if save_path:
            output_path = save_path if os.path.dirname(save_path) else os.path.join(self.output_dir or '.', save_path)
            fig.write_html(output_path)
        
        return fig
    
    def visualize_tumor_regions(self,
                              graph: nx.Graph,
                              patient_id: int,
                              save_path: Optional[str] = None) -> go.Figure:
        """
        Visualize tumor regions from graph.

        Args:
            graph: Graph with tumor region information
            patient_id: Patient identifier
            save_path: Optional path to save visualization

        Returns:
            Plotly figure with tumor region visualization
        """
        # Only implemented for Plotly
        if not self.use_plotly:
            print("Warning: Tumor region visualization only available with Plotly.")
            self.use_plotly = True
        
        # Extract region nodes
        region_nodes = [
            node for node, attrs in graph.nodes(data=True)
            if attrs.get('type') == 'region' and f"{patient_id}" in node
        ]
        
        if not region_nodes:
            # Create empty figure with message
            fig = go.Figure()
            fig.add_annotation(
                text="No tumor region data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16)
            )
            return fig
        
        # Extract region centers and volumes
        region_centers = []
        region_volumes = []
        region_ids = []
        
        for node in region_nodes:
            attrs = graph.nodes[node]
            
            x = attrs.get('center_x')
            y = attrs.get('center_y')
            z = attrs.get('center_z')
            volume = attrs.get('volume_voxels', 1.0)
            
            if x is not None and y is not None and z is not None:
                region_centers.append((x, y, z))
                region_volumes.append(volume)
                region_ids.append(node)
        
        # Create 3D scatter plot
        fig = go.Figure()
        
        # Add regions as spheres
        x_vals = [center[0] for center in region_centers]
        y_vals = [center[1] for center in region_centers]
        z_vals = [center[2] for center in region_centers]
        
        # Scale sizes based on volumes
        min_size = 10
        max_size = 50
        if region_volumes:
            min_vol = min(region_volumes)
            max_vol = max(region_volumes) if max(region_volumes) > min_vol else min_vol + 1
            sizes = [min_size + (vol - min_vol) * (max_size - min_size) / (max_vol - min_vol) for vol in region_volumes]
        else:
            sizes = [20] * len(region_centers)
        
        # Create hover text with region info
        hover_texts = []
        for node, volume in zip(region_ids, region_volumes):
            region_id = node.split('_')[1] if '_' in node else 'unknown'
            hover_texts.append(f"Region: {region_id}<br>Volume: {volume:.2f}")
        
        # Add regions as scatter points
        fig.add_trace(
            go.Scatter3d(
                x=x_vals,
                y=y_vals,
                z=z_vals,
                mode='markers',
                marker=dict(
                    size=sizes,
                    color=list(range(len(x_vals))),
                    colorscale='Viridis',
                    opacity=0.8,
                    colorbar=dict(title="Region ID")
                ),
                text=hover_texts,
                hoverinfo='text',
                name='Tumor Regions'
            )
        )
        
        # Add edges between regions
        edge_x = []
        edge_y = []
        edge_z = []
        
        for i, node1 in enumerate(region_ids):
            for j, node2 in enumerate(region_ids):
                if i < j and graph.has_edge(node1, node2):
                    # Add line between regions
                    edge_x.extend([x_vals[i], x_vals[j], None])
                    edge_y.extend([y_vals[i], y_vals[j], None])
                    edge_z.extend([z_vals[i], z_vals[j], None])
        
        if edge_x:
            fig.add_trace(
                go.Scatter3d(
                    x=edge_x,
                    y=edge_y,
                    z=edge_z,
                    mode='lines',
                    line=dict(
                        color='rgba(125, 125, 125, 0.5)',
                        width=2
                    ),
                    hoverinfo='none',
                    name='Region Connections'
                )
            )
        
        # Update layout
        fig.update_layout(
            title=f"Tumor Regions for Patient {patient_id}",
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
                aspectmode='data'
            ),
            margin=dict(l=0, r=0, b=0, t=40),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            height=700
        )
        
        # Save figure if path provided
        if save_path:
            output_path = save_path if os.path.dirname(save_path) else os.path.join(self.output_dir or '.', save_path)
            fig.write_html(output_path)
        
        return fig
    
    def visualize_treatment_comparison(self,
                                     treatment_simulation: Dict[str, Dict[str, Any]],
                                     patient_id: int,
                                     save_path: Optional[str] = None) -> go.Figure:
        """
        Visualize comparison of different treatment options.

        Args:
            treatment_simulation: Dictionary of treatment simulation results
            patient_id: Patient identifier
            save_path: Optional path to save visualization

        Returns:
            Plotly figure comparing treatment options
        """
        # Check if we have simulation results
        if not treatment_simulation:
            # Create empty figure with message
            if self.use_plotly:
                fig = go.Figure()
                fig.add_annotation(
                    text="No treatment simulation data available",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5,
                    showarrow=False,
                    font=dict(size=16)
                )
                return fig
            else:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.text(0.5, 0.5, "No treatment simulation data available",
                        horizontalalignment='center',
                        verticalalignment='center',
                        transform=ax.transAxes,
                        fontsize=16)
                return fig
        
        # Extract data for comparison
        treatment_names = []
        response_probs = []
        survival_days = []
        uncertainties = []
        
        for option_id, result in treatment_simulation.items():
            config = result.get('config', {})
            
            # Get treatment name
            category = config.get('category', 'unknown')
            name = f"{category.capitalize()}"
            if 'dose' in config and config['dose'] is not None:
                name += f" ({config['dose']})"
            
            treatment_names.append(name)
            response_probs.append(result.get('response_prob', 0) * 100)  # Convert to percentage
            survival_days.append(result.get('survival_days', 0))
            uncertainties.append(result.get('uncertainty', 0))
        
        if self.use_plotly:
            return self._plotly_treatment_comparison(
                treatment_names, response_probs, survival_days, uncertainties, patient_id, save_path
            )
        else:
            return self._matplotlib_treatment_comparison(
                treatment_names, response_probs, survival_days, uncertainties, patient_id, save_path
            )
    
    def _plotly_treatment_comparison(self,
                                    treatment_names: List[str],
                                    response_probs: List[float],
                                    survival_days: List[float],
                                    uncertainties: List[float],
                                    patient_id: int,
                                    save_path: Optional[str] = None) -> go.Figure:
        """
        Create Plotly visualization of treatment comparison.

        Args:
            treatment_names: List of treatment names
            response_probs: List of response probabilities
            survival_days: List of survival days
            uncertainties: List of uncertainty values
            patient_id: Patient identifier
            save_path: Optional path to save visualization

        Returns:
            Plotly figure
        """
        # Create subplots: 1 row, 2 columns
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=["Response Probability", "Survival Days"],
            specs=[[{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Add bars for response probability
        fig.add_trace(
            go.Bar(
                x=treatment_names,
                y=response_probs,
                error_y=dict(
                    type='data',
                    array=uncertainties,
                    visible=True
                ),
                marker_color='rgba(58, 171, 195, 0.6)',
                name='Response Probability (%)'
            ),
            row=1, col=1
        )
        
        # Add bars for survival days
        fig.add_trace(
            go.Bar(
                x=treatment_names,
                y=survival_days,
                error_y=dict(
                    type='data',
                    array=[u * 100 for u in uncertainties],  # Scale uncertainty for survival
                    visible=True
                ),
                marker_color='rgba(222, 67, 67, 0.6)',
                name='Survival Days'
            ),
            row=1, col=2
        )
        
        # Update layout
        fig.update_layout(
            title=f"Treatment Comparison for Patient {patient_id}",
            showlegend=True,
            height=500,
            width=1000,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Update yaxis
        fig.update_yaxes(title_text="Probability (%)", range=[0, 100], row=1, col=1)
        fig.update_yaxes(title_text="Days", row=1, col=2)
        
        # Save figure if path provided
        if save_path:
            output_path = save_path if os.path.dirname(save_path) else os.path.join(self.output_dir or '.', save_path)
            fig.write_html(output_path)
        
        return fig
    
    def _matplotlib_treatment_comparison(self,
                                        treatment_names: List[str],
                                        response_probs: List[float],
                                        survival_days: List[float],
                                        uncertainties: List[float],
                                        patient_id: int,
                                        save_path: Optional[str] = None) -> plt.Figure:
        """
        Create Matplotlib visualization of treatment comparison.

        Args:
            treatment_names: List of treatment names
            response_probs: List of response probabilities
            survival_days: List of survival days
            uncertainties: List of uncertainty values
            patient_id: Patient identifier
            save_path: Optional path to save visualization

        Returns:
            Matplotlib figure
        """
        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot response probability
        bars1 = ax1.bar(treatment_names, response_probs, yerr=uncertainties, 
                       alpha=0.7, color='royalblue', capsize=5)
        ax1.set_ylabel('Probability (%)')
        ax1.set_title('Response Probability')
        ax1.set_ylim(0, 100)
        
        # Plot survival days
        bars2 = ax2.bar(treatment_names, survival_days, yerr=[u * 100 for u in uncertainties],
                       alpha=0.7, color='firebrick', capsize=5)
        ax2.set_ylabel('Days')
        ax2.set_title('Survival Days')
        
        # Add data labels on bars
        def add_labels(bars, ax, format_str='{:.1f}'):
            for bar in bars:
                height = bar.get_height()
                ax.annotate(format_str.format(height),
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom',
                            fontsize=9)
        
        add_labels(bars1, ax1, '{:.1f}%')
        add_labels(bars2, ax2, '{:.0f}')
        
        # Rotate x labels
        for ax in [ax1, ax2]:
            plt.sca(ax)
            plt.xticks(rotation=30, ha='right')
        
        # Add overall title
        fig.suptitle(f"Treatment Comparison for Patient {patient_id}", fontsize=16)
        
        plt.tight_layout()
        
        # Save figure if path provided
        if save_path:
            output_path = save_path if os.path.dirname(save_path) else os.path.join(self.output_dir or '.', save_path)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        return fig