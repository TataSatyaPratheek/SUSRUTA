"""
Explanation generation for glioma treatment recommendations.

Implements feature attribution and visualization for treatment decisions.
"""

from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
import torch
import torch.nn.functional as F
import networkx as nx
from torch_geometric.data import HeteroData

from ..models.gnn import GliomaGNN


class ExplainableGliomaTreatment:
    """Explainable treatment recommendations with confidence estimates."""
    
    def __init__(self, 
                model: GliomaGNN, 
                graph_data: HeteroData, 
                original_graph: Optional[nx.MultiDiGraph] = None, 
                device: Optional[torch.device] = None):
        """
        Initialize explainable treatment module.
        
        Args:
            model: Trained GNN model
            graph_data: PyTorch Geometric data
            original_graph: Original NetworkX graph for attribute lookup
            device: Device to run computations on
        """
        self.model = model
        self.data = graph_data
        self.original_graph = original_graph  # NetworkX graph for attribute lookup
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.eval()  # Set to evaluation mode
    
    def explain_treatment_prediction(self, treatment_idx: int, k: int = 5) -> Dict[str, Any]:
        """
        Explain treatment outcome prediction.
        
        Args:
            treatment_idx: Index of treatment node to explain
            k: Number of top features to include in explanation
            
        Returns:
            Dictionary of explanations and visualizations
        """
        # Get treatment node features
        x = self.data['treatment'].x[treatment_idx].clone().to(self.device)
        x.requires_grad_(True)
        
        # Forward pass with gradients
        self.model.zero_grad()
        
        # Create input dictionaries
        x_dict = {k: v.x.to(self.device) if k != 'treatment' else torch.stack([x])
                 for k, v in self.data.items()}
        
        edge_indices_dict = {}
        for edge_type in self.data.edge_types:
            edge_indices_dict[edge_type] = self.data[edge_type].edge_index.to(self.device)
            
        # Forward pass
        output_dict, _ = self.model(x_dict, edge_indices_dict)
        
        # Extract predictions
        response_pred = output_dict['response'][0]
        survival_pred = output_dict['survival'][0]
        uncertainty = output_dict['uncertainty'][0]
        
        # Compute gradients for response prediction
        response_pred.backward(retain_graph=True)
        response_gradients = x.grad.clone()
        
        # Zero gradients and compute for survival prediction
        self.model.zero_grad()
        x.grad = None
        survival_pred.backward()
        survival_gradients = x.grad.clone()
        
        # Feature importance scores (absolute gradient values)
        response_importance = torch.abs(response_gradients).cpu().numpy()
        survival_importance = torch.abs(survival_gradients).cpu().numpy()
        
        # Get original treatment node ID
        treatment_id = self.data['treatment'].original_ids[treatment_idx] if hasattr(
            self.data['treatment'], 'original_ids') else f"treatment_{treatment_idx}"
        
        # Get treatment attributes and connected patient/tumor
        treatment_attrs = {}
        patient_id = None
        tumor_id = None
        
        if self.original_graph is not None and treatment_id in self.original_graph:
            treatment_attrs = dict(self.original_graph.nodes[treatment_id])
            
            # Find connected tumor
            for src, dst in self.original_graph.in_edges(treatment_id):
                if 'tumor' in src:
                    tumor_id = src
                    break
            
            # Find connected patient
            if tumor_id:
                for src, dst in self.original_graph.in_edges(tumor_id):
                    if 'patient' in src:
                        patient_id = src
                        break
        
        # Create feature explanation
        feature_names = ['category_surgery', 'category_radiation', 'category_chemo', 
                        'category_combined', 'dose', 'duration', 'intensity']
        
        # Truncate to actual feature length
        feature_names = feature_names[:len(response_importance)]
        
        # Sort features by importance
        response_indices = np.argsort(-response_importance)[:k]
        survival_indices = np.argsort(-survival_importance)[:k]
        
        response_top_features = [(feature_names[i], float(response_importance[i])) 
                               for i in response_indices]
        survival_top_features = [(feature_names[i], float(survival_importance[i])) 
                                for i in survival_indices]
        
        # Create explanation dictionary
        explanation = {
            'treatment_id': treatment_id,
            'treatment_attributes': treatment_attrs,
            'patient_id': patient_id,
            'tumor_id': tumor_id,
            'predictions': {
                'response_probability': float(response_pred.item()),
                'survival_days': float(survival_pred.item()),
                'uncertainty': float(uncertainty.item())
            },
            'feature_importance': {
                'response': response_top_features,
                'survival': survival_top_features
            }
        }
        
        return explanation
    
    def get_treatment_comparison(self, treatment_indices: List[int]) -> Dict[str, Any]:
        """
        Generate comparison between multiple treatments.
        
        Args:
            treatment_indices: List of treatment indices to compare
            
        Returns:
            Comparison data dictionary
        """
        comparison = {
            'treatments': [],
            'predictions': [],
            'common_features': set()
        }
        
        # Get explanations for each treatment
        for idx in treatment_indices:
            explanation = self.explain_treatment_prediction(idx)
            comparison['treatments'].append(explanation)
            comparison['predictions'].append(explanation['predictions'])
            
            # Track common important features
            if not comparison['common_features']:
                comparison['common_features'] = set(f[0] for f in explanation['feature_importance']['response'])
            else:
                comparison['common_features'] &= set(f[0] for f in explanation['feature_importance']['response'])
        
        # Find differentiating features
        all_features = set()
        for explanation in comparison['treatments']:
            all_features.update(f[0] for f in explanation['feature_importance']['response'])
            all_features.update(f[0] for f in explanation['feature_importance']['survival'])
        
        differentiating_features = all_features - comparison['common_features']
        comparison['differentiating_features'] = list(differentiating_features)
        
        return comparison
    
    def generate_natural_language_explanation(self, explanation: Dict[str, Any]) -> str:
        """
        Generate natural language explanation for treatment prediction.
        
        Args:
            explanation: Explanation dictionary from explain_treatment_prediction
            
        Returns:
            Natural language explanation string
        """
        # Extract key information
        treatment_attrs = explanation['treatment_attributes']
        predictions = explanation['predictions']
        response_features = explanation['feature_importance']['response']
        survival_features = explanation['feature_importance']['survival']
        
        # Get treatment category
        category = treatment_attrs.get('category', 'unknown')
        
        # Format predictions
        response_prob = predictions['response_probability'] * 100
        survival_days = predictions['survival_days']
        uncertainty = predictions['uncertainty']
        
        # Generate explanation text
        explanation_text = f"The {category} treatment is predicted to have a {response_prob:.1f}% "
        explanation_text += f"probability of positive response, with an estimated survival of {survival_days:.0f} days. "
        
        # Add uncertainty information
        if uncertainty <= 0.1:
            uncertainty_level = "high confidence"
        elif uncertainty < 0.3:
            uncertainty_level = "moderate confidence"
        else:
            uncertainty_level = "low confidence"
        
        explanation_text += f"This prediction has {uncertainty_level} (uncertainty: {uncertainty:.2f}). "
        
        # Add feature importance information
        if response_features:
            explanation_text += "\n\nKey factors influencing response prediction: "
            for i, (feature, importance) in enumerate(response_features[:3]):
                if i > 0:
                    explanation_text += ", "
                explanation_text += f"{feature} ({importance:.3f})"
        
        if survival_features:
            explanation_text += "\n\nKey factors influencing survival prediction: "
            for i, (feature, importance) in enumerate(survival_features[:3]):
                if i > 0:
                    explanation_text += ", "
                explanation_text += f"{feature} ({importance:.3f})"
        
        # Add treatment details
        explanation_text += "\n\nTreatment details: "
        for key, value in treatment_attrs.items():
            if key not in ['type', 'category'] and value is not None:
                explanation_text += f"{key}: {value}, "
        
        return explanation_text