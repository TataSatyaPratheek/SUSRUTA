# susruta/src/susruta/viz/explain.py
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
        self.model.to(self.device) # Ensure model is on correct device
        self.model.eval()  # Set to evaluation mode

    def explain_treatment_prediction(self, treatment_idx: int, k: int = 5) -> Dict[str, Any]:
        """
        Explain treatment outcome prediction using gradient attribution.

        Args:
            treatment_idx: Index of treatment node to explain within the PyG data
            k: Number of top features to include in explanation

        Returns:
            Dictionary containing explanation details
        """
        # Use num_nodes for bounds check
        if 'treatment' not in self.data.node_types or \
           not hasattr(self.data['treatment'], 'num_nodes') or \
           treatment_idx >= self.data['treatment'].num_nodes:
            num_nodes = self.data['treatment'].num_nodes if ('treatment' in self.data.node_types and hasattr(self.data['treatment'], 'num_nodes')) else 0
            raise ValueError(f"Treatment index {treatment_idx} is out of bounds for {num_nodes} treatment nodes.")

        # Ensure features exist for the treatment node
        if not hasattr(self.data['treatment'], 'x') or self.data['treatment'].x is None:
             raise ValueError("Treatment node features (data['treatment'].x) are missing.")
        num_features = self.data['treatment'].x.shape[1]
        if self.data['treatment'].num_nodes > 0 and num_features == 0:
             raise ValueError("Treatment node features have dimension 0.")


        # --- Gradient Calculation Setup ---
        data_clone = self.data.clone().to(self.device)
        for node_type in data_clone.node_types:
            if hasattr(data_clone[node_type], 'x') and data_clone[node_type].x is not None:
                data_clone[node_type].x = data_clone[node_type].x.detach()

        if data_clone['treatment'].x.shape[0] > treatment_idx:
             data_clone['treatment'].x.requires_grad_(True)
        else:
             raise RuntimeError(f"Cannot enable gradients: Treatment index {treatment_idx} still out of bounds after cloning.")

        x_dict = {nt: data_clone[nt].x for nt in data_clone.node_types if hasattr(data_clone[nt], 'x')}
        edge_index_dict = {et: data_clone[et].edge_index for et in data_clone.edge_types if hasattr(data_clone[et], 'edge_index')}

        # --- Forward Pass and Gradient Computation ---
        self.model.zero_grad()
        predictions, _ = self.model(x_dict, edge_index_dict)

        if 'response' not in predictions or predictions['response'].shape[0] <= treatment_idx:
             raise RuntimeError(f"Model did not produce predictions for treatment index {treatment_idx}.")

        response_pred = predictions['response'][treatment_idx]
        survival_pred = predictions['survival'][treatment_idx]
        uncertainty = predictions['uncertainty'][treatment_idx]

        # Compute gradients w.r.t. response prediction
        self.model.zero_grad()
        response_gradients = torch.zeros_like(data_clone['treatment'].x[treatment_idx])
        if response_pred.requires_grad:
            grad_outputs_resp = torch.ones_like(response_pred)
            response_pred.backward(gradient=grad_outputs_resp, retain_graph=True)
            if data_clone['treatment'].x.grad is not None:
                response_gradients = data_clone['treatment'].x.grad[treatment_idx].clone()
                data_clone['treatment'].x.grad.zero_()
            else:
                 print("Warning: Gradients for treatment features are None after response backward pass.")
        else:
            print("Warning: Response prediction does not require grad. Gradients will be zero.")

        # Compute gradients w.r.t. survival prediction
        self.model.zero_grad()
        survival_gradients = torch.zeros_like(data_clone['treatment'].x[treatment_idx])
        if survival_pred.requires_grad:
            grad_outputs_surv = torch.ones_like(survival_pred)
            survival_pred.backward(gradient=grad_outputs_surv)
            if data_clone['treatment'].x.grad is not None:
                survival_gradients = data_clone['treatment'].x.grad[treatment_idx].clone()
            else:
                 print("Warning: Gradients for treatment features are None after survival backward pass.")
        else:
            print("Warning: Survival prediction does not require grad. Gradients will be zero.")

        # --- Feature Importance and Explanation Generation ---
        response_importance = torch.abs(response_gradients).cpu().numpy()
        survival_importance = torch.abs(survival_gradients).cpu().numpy()

        treatment_id_str = "unknown_treatment"
        if hasattr(self.data['treatment'], 'original_ids') and treatment_idx < len(self.data['treatment'].original_ids):
             tid = self.data['treatment'].original_ids[treatment_idx]
             treatment_id_str = str(tid) if not isinstance(tid, str) else tid

        treatment_attrs = {}
        patient_id = None
        tumor_id = None
        if self.original_graph is not None and self.original_graph.has_node(treatment_id_str):
            treatment_attrs = dict(self.original_graph.nodes[treatment_id_str])
            for u, v, data in self.original_graph.in_edges(treatment_id_str, data=True):
                if data.get('relation') == 'treated_with' and self.original_graph.nodes[u].get('type') == 'tumor':
                    tumor_id = u
                    for p_u, p_v, p_data in self.original_graph.in_edges(tumor_id, data=True):
                         if p_data.get('relation') == 'has_tumor' and self.original_graph.nodes[p_u].get('type') == 'patient':
                             patient_id = p_u
                             break
                    break

        # --- Start Fix: Infer feature names based on model's expected input ---
        # Get the expected feature dimension for 'treatment' from the model
        expected_treatment_dim = self.model.node_encoders['treatment'].in_features
        # Assume features correspond to sorted numerical attributes from graph conversion
        # This relies on consistency between graph building and model init.
        feature_names = []
        if self.original_graph and self.original_graph.has_node(treatment_id_str):
             node_data = self.original_graph.nodes[treatment_id_str]
             numeric_keys = sorted([
                 key for key, value in node_data.items()
                 if key != 'type' and isinstance(value, (int, float, np.number)) and not np.isnan(value)
             ])
             feature_names = numeric_keys

        # Pad or truncate feature_names to match the expected dimension from the model/gradients
        num_features_grad = max(len(response_importance), len(survival_importance), expected_treatment_dim)
        if len(feature_names) < num_features_grad:
             feature_names.extend([f'feature_{i}' for i in range(len(feature_names), num_features_grad)])
        elif len(feature_names) > num_features_grad:
             feature_names = feature_names[:num_features_grad]
        # --- End Fix ---

        # Sort features by importance
        response_indices = np.argsort(-response_importance)[:k]
        survival_indices = np.argsort(-survival_importance)[:k]

        response_top_features = [(feature_names[i], float(response_importance[i]))
                            for i in response_indices if i < len(feature_names)]
        survival_top_features = [(feature_names[i], float(survival_importance[i]))
                                for i in survival_indices if i < len(feature_names)]

        explanation = {
            'treatment_id': treatment_id_str,
            'treatment_attributes': {k: (float(v) if isinstance(v, np.number) else v) for k, v in treatment_attrs.items()},
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

        del data_clone, x_dict, edge_index_dict, predictions
        torch.cuda.empty_cache() if self.device.type == 'cuda' else None

        return explanation

    def get_treatment_comparison(self, treatment_indices: List[int]) -> Dict[str, Any]:
        """
        Generate comparison between multiple treatments.

        Args:
            treatment_indices: List of treatment indices to compare

        Returns:
            Comparison data dictionary
        """
        comparison_results: Dict[str, Any] = {
            'treatments': [],
            'common_response_features': set(),
            'common_survival_features': set(),
            'all_response_features': set(),
            'all_survival_features': set(),
        }
        all_response_feature_names = set()
        all_survival_feature_names = set()

        for i, idx in enumerate(treatment_indices):
            try:
                explanation = self.explain_treatment_prediction(idx)
                comparison_results['treatments'].append(explanation)

                current_response_features = set(f[0] for f in explanation['feature_importance']['response'])
                current_survival_features = set(f[0] for f in explanation['feature_importance']['survival'])

                all_response_feature_names.update(current_response_features)
                all_survival_feature_names.update(current_survival_features)

                if i == 0:
                    comparison_results['common_response_features'] = current_response_features.copy()
                    comparison_results['common_survival_features'] = current_survival_features.copy()
                else:
                    comparison_results['common_response_features'] &= current_response_features
                    comparison_results['common_survival_features'] &= current_survival_features

            except ValueError as e:
                print(f"Skipping comparison for treatment index {idx}: {e}")
                continue
            except RuntimeError as e:
                 print(f"Runtime error during explanation for index {idx}: {e}")
                 continue

        # --- Start Fix: Assign updated sets back to results dict ---
        comparison_results['all_response_features'] = all_response_feature_names
        comparison_results['all_survival_features'] = all_survival_feature_names
        # --- End Fix ---

        comparison_results['differentiating_response_features'] = list(all_response_feature_names - comparison_results['common_response_features'])
        comparison_results['differentiating_survival_features'] = list(all_survival_feature_names - comparison_results['common_survival_features'])

        comparison_results['common_features'] = comparison_results['common_response_features'].union(comparison_results['common_survival_features'])
        comparison_results['differentiating_features'] = set(comparison_results['differentiating_response_features']).union(set(comparison_results['differentiating_survival_features']))

        return comparison_results

    def generate_natural_language_explanation(self, explanation: Dict[str, Any]) -> str:
        """
        Generate natural language explanation for treatment prediction.

        Args:
            explanation: Explanation dictionary from explain_treatment_prediction

        Returns:
            Natural language explanation string
        """
        treatment_attrs = explanation.get('treatment_attributes', {})
        predictions = explanation.get('predictions', {})
        feature_importance = explanation.get('feature_importance', {})
        response_features = feature_importance.get('response', [])
        survival_features = feature_importance.get('survival', [])

        category = treatment_attrs.get('category', 'unknown')
        treatment_name = treatment_attrs.get('name', f"{category} treatment")

        response_prob = predictions.get('response_probability', 0.0) * 100
        survival_days = predictions.get('survival_days', 0.0)
        uncertainty = predictions.get('uncertainty', 1.0)

        explanation_text = f"For {treatment_name}, the predicted probability of a positive response is {response_prob:.1f}%, "
        explanation_text += f"with an estimated survival of {survival_days:.0f} days. "

        if uncertainty <= 0.1:
            uncertainty_level = "high confidence"
        elif uncertainty < 0.3:
            uncertainty_level = "moderate confidence"
        else:
            uncertainty_level = "low confidence (high uncertainty)"

        explanation_text += f"The model prediction has {uncertainty_level} (uncertainty score: {uncertainty:.3f})." # Added period

        if response_features:
            explanation_text += "\n\nKey factors influencing the response prediction: "
            feature_strs = [f"{feature} ({importance:.3f})" for feature, importance in response_features]
            explanation_text += ", ".join(feature_strs) + "."

        if survival_features:
            explanation_text += "\nKey factors influencing the survival prediction: "
            feature_strs = [f"{feature} ({importance:.3f})" for feature, importance in survival_features]
            explanation_text += ", ".join(feature_strs) + "."

        details = []
        for key, value in treatment_attrs.items():
            if key not in ['type', 'category', 'name'] and value is not None:
                 if isinstance(value, (float, np.float32)):
                     details.append(f"{key}: {value:.1f}")
                 else:
                     details.append(f"{key}: {value}")
        if details:
             explanation_text += "\n\nTreatment details considered: " + ", ".join(details) + "."

        return explanation_text.strip()
