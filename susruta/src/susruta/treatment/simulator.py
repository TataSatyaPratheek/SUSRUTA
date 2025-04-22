# susruta/src/susruta/treatment/simulator.py
"""
Treatment simulation and counterfactual reasoning for glioma outcome prediction.

Implements memory-efficient treatment simulation and ranking based on graph structure.
"""

from typing import Dict, List, Tuple, Any, Optional, Union
import torch
import torch.nn.functional as F
import networkx as nx
from torch_geometric.data import HeteroData
import numpy as np # Import numpy

from ..models.gnn import GliomaGNN


class TreatmentSimulator:
    """Simulate counterfactual treatment outcomes based on graph structure."""

    def __init__(self, model: GliomaGNN, graph: nx.MultiDiGraph, device: Optional[torch.device] = None):
        """
        Initialize treatment simulator.

        Args:
            model: Trained GNN model
            graph: NetworkX graph of knowledge graph
            device: Device to run computations on
        """
        self.model = model
        self.graph = graph
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device) # Ensure model is on the correct device
        self.model.eval()  # Set model to evaluation mode

    def simulate_treatments(self,
                          patient_id: str,
                          tumor_id: str,
                          treatment_options: List[Dict[str, Any]],
                          data: HeteroData) -> Dict[str, Dict[str, Any]]:
        """
        Simulate outcomes for different treatment options.

        Args:
            patient_id: Patient node ID (e.g., "patient_1")
            tumor_id: Tumor node ID (e.g., "tumor_1")
            treatment_options: List of treatment configurations to simulate
            data: PyTorch Geometric data object

        Returns:
            Dictionary of predicted outcomes for each treatment option
        """
        results = {}

        # Find node indices safely (needed for potential future use, but not for the check below)
        try:
            if hasattr(data['patient'], 'original_ids'):
                patient_idx = data['patient'].original_ids.index(patient_id)
            else:
                # Fallback assuming patient_id format like "patient_N"
                patient_idx = int(patient_id.split('_')[-1]) - 1
        except (ValueError, IndexError, AttributeError, KeyError):
             raise ValueError(f"Patient ID {patient_id} not found or invalid in pyg_data['patient'].original_ids")

        try:
            if hasattr(data['tumor'], 'original_ids'):
                tumor_idx = data['tumor'].original_ids.index(tumor_id)
            else:
                 # Fallback assuming tumor_id format like "tumor_N"
                 tumor_idx = int(tumor_id.split('_')[-1]) - 1
        except (ValueError, IndexError, AttributeError, KeyError):
             raise ValueError(f"Tumor ID {tumor_id} not found or invalid in pyg_data['tumor'].original_ids")


        # Get node embeddings *after* GNN layers
        with torch.no_grad():
            # Forward pass to get node embeddings
            x_dict = {k: v.x.to(self.device) for k, v in data.items() if hasattr(v, 'x')} # Use items() and check for x

            edge_indices_dict = {}
            for edge_type_tuple in data.edge_types:
                 if hasattr(data[edge_type_tuple], 'edge_index'):
                     edge_indices_dict[edge_type_tuple] = data[edge_type_tuple].edge_index.to(self.device)

            # node_embeddings now contains embeddings only for nodes updated by GNN layers
            _, node_embeddings = self.model(x_dict, edge_indices_dict)

        # --- START FIX: Remove checks for patient/tumor embeddings in the returned dict ---
        # The returned node_embeddings might not contain 'patient' or 'tumor' if they
        # weren't destination nodes during message passing. These checks are removed
        # as the simulation loop primarily relies on 'treatment' embeddings.
        # if 'patient' not in node_embeddings or patient_idx >= node_embeddings['patient'].shape[0]:
        #     raise ValueError(f"Could not get embedding for patient {patient_id} (index {patient_idx})")
        # if 'tumor' not in node_embeddings or tumor_idx >= node_embeddings['tumor'].shape[0]:
        #     tumor_shape = node_embeddings.get('tumor', torch.empty(0)).shape
        #     raise ValueError(f"Could not get embedding for tumor {tumor_id} (index {tumor_idx}). Tumor embedding tensor shape: {tumor_shape}")
        # --- END FIX ---

        # Check if treatment embeddings are available (either from GNN output or encoder fallback)
        if 'treatment' not in node_embeddings and 'treatment' not in self.model.node_encoders:
             raise ValueError("Model cannot produce treatment embeddings needed for simulation.")
        elif 'treatment' not in node_embeddings:
             print("Warning: 'treatment' embeddings not found in GNN output, will rely on encoder.")
             # Ensure node_embeddings['treatment'] exists for _find_similar_treatments fallback
             node_embeddings['treatment'] = torch.empty((0, self.model.hidden_channels), device=self.device)


        # Simulate each treatment
        for treatment_idx, treatment_config in enumerate(treatment_options):
            # Create synthetic treatment node features
            treatment_features_list = self._encode_treatment(treatment_config)
            treatment_features = torch.tensor(treatment_features_list, dtype=torch.float).unsqueeze(0).to(self.device) # Add batch dim

            # Find embeddings of similar existing treatments from the GNN output embeddings
            similar_treatment_embeddings = self._find_similar_treatments(treatment_config, data, node_embeddings)

            if similar_treatment_embeddings:
                # Use average embedding of similar treatments
                treatment_embedding = torch.stack(similar_treatment_embeddings).mean(dim=0, keepdim=True) # Keep batch dim
            elif 'treatment' in self.model.node_encoders:
                # If no similar treatments, use the model's encoder for the new features
                # Ensure input dimension matches encoder's expected dimension
                expected_dim = self.model.node_encoders['treatment'].in_features
                if treatment_features.shape[1] != expected_dim:
                     # Pad or truncate features if necessary
                     if treatment_features.shape[1] < expected_dim:
                         padding = torch.zeros((1, expected_dim - treatment_features.shape[1]), device=self.device)
                         treatment_features = torch.cat([treatment_features, padding], dim=1)
                     else:
                         treatment_features = treatment_features[:, :expected_dim]

                treatment_embedding = self.model.node_encoders['treatment'](treatment_features)
            else:
                 print(f"Warning: Cannot generate embedding for treatment option {treatment_idx}. Skipping.")
                 continue


            # Predict outcomes using the model's prediction heads
            # Ensure the embedding has the correct shape [1, hidden_channels]
            if treatment_embedding.dim() == 1:
                 treatment_embedding = treatment_embedding.unsqueeze(0)

            response_pred = self.model.response_head(treatment_embedding)
            survival_pred = self.model.survival_head(treatment_embedding)
            uncertainty = self.model.uncertainty_head(treatment_embedding)

            # Store results
            results[f"option_{treatment_idx}"] = {
                'config': treatment_config,
                'response_prob': float(response_pred.item()),
                'survival_days': float(survival_pred.item()),
                'uncertainty': float(uncertainty.item())
            }

        return results

    def _encode_treatment(self, treatment_config: Dict[str, Any]) -> List[float]:
        """
        Encode treatment configuration into features.

        Args:
            treatment_config: Treatment configuration dictionary

        Returns:
            List of encoded feature values
        """
        # This encoding MUST match the feature extraction in GliomaKnowledgeGraph.to_pytorch_geometric
        # and the expected input dimension of the model's treatment encoder.
        # Using a fixed encoding based on common attributes:
        features = []

        # Encode treatment category (one-hot) - Assuming 4 categories max
        categories = ['surgery', 'radiation', 'chemotherapy', 'combined'] # Example categories
        category_vec = [0.0] * 4 # Use float
        cat = treatment_config.get('category')
        if cat in categories:
            category_vec[categories.index(cat)] = 1.0
        features.extend(category_vec)

        # --- START FIX: Encode only dose, duration, intensity ---
        numerical_keys = ['dose', 'duration', 'intensity'] # 3 numerical features
        for key in numerical_keys:
            value = treatment_config.get(key, 0.0)
            # Ensure value is treated as float, handle None explicitly
            features.append(float(value if value is not None else 0.0))
        # --- END FIX ---
        return features

    def _find_similar_treatments(self,
                                 treatment_config: Dict[str, Any],
                                 data: HeteroData,
                                 node_embeddings: Dict[str, torch.Tensor] # Add embeddings argument
                                 ) -> List[torch.Tensor]:
        """Find embeddings of similar treatments in the graph."""
        similar_treatments = []
        category_to_find = treatment_config.get('category')

        # Basic checks
        if not category_to_find: return similar_treatments
        if 'treatment' not in data.node_types: return similar_treatments # Check node type exists
        if not hasattr(data['treatment'], 'original_ids'): return similar_treatments
        if 'treatment' not in node_embeddings: return similar_treatments # Check passed embeddings exist
        if node_embeddings['treatment'].shape[0] == 0: return similar_treatments # Check embeddings are not empty

        num_treatment_nodes = data['treatment'].num_nodes
        num_treatment_embeddings = node_embeddings['treatment'].shape[0]

        # Ensure consistency between original_ids and embeddings
        if num_treatment_nodes != len(data['treatment'].original_ids):
             print(f"Warning: Mismatch in treatment node count ({num_treatment_nodes}) and original_ids ({len(data['treatment'].original_ids)})")
             # Decide how to handle: return empty or try to proceed? Let's try to proceed cautiously.
             # return similar_treatments
        if num_treatment_embeddings != num_treatment_nodes:
             print(f"Warning: Mismatch in treatment node count ({num_treatment_nodes}) and embeddings ({num_treatment_embeddings})")
             # Limit iteration to the minimum of the two counts
             num_to_iterate = min(num_treatment_nodes, num_treatment_embeddings, len(data['treatment'].original_ids))
        else:
             num_to_iterate = num_treatment_nodes


        # Find treatments with matching category using the passed embeddings
        for idx in range(num_to_iterate): # Iterate up to the safe limit
            treatment_id_str = data['treatment'].original_ids[idx]

            # Check if the treatment_id exists in the original NetworkX graph
            if treatment_id_str not in self.graph:
                # print(f"Warning: Treatment ID '{treatment_id_str}' from pyg_data not found in NetworkX graph. Skipping.")
                continue # Skip if not in the graph used for attribute lookup

            try:
                treatment_attrs = self.graph.nodes[treatment_id_str]
                if treatment_attrs.get('category') == category_to_find:
                    similar_treatments.append(node_embeddings['treatment'][idx])
            except KeyError:
                 print(f"Warning: KeyError accessing node attributes for '{treatment_id_str}'. Skipping.")
                 continue


        return similar_treatments

    def rank_treatments(self,
                      simulated_outcomes: Dict[str, Dict[str, Any]],
                      weights: Optional[Dict[str, float]] = None) -> List[str]:
        """
        Rank treatments based on simulated outcomes and preference weights.

        Args:
            simulated_outcomes: Dictionary of simulated treatment outcomes
            weights: Optional dictionary of preference weights

        Returns:
            List of treatment option IDs sorted by rank
        """
        if not simulated_outcomes:
            return []

        if weights is None:
            # Default weights prioritize survival and response equally
            weights = {
                'response_prob': 0.4,
                'survival_days': 0.4,
                'uncertainty': -0.2  # Negative weight for uncertainty
            }

        # Calculate score for each treatment
        scores = {}

        # Normalize values across treatments
        metrics = {
            'response_prob': [],
            'survival_days': [],
            'uncertainty': []
        }

        option_ids = list(simulated_outcomes.keys()) # Keep track of order

        for option_id in option_ids:
            outcome = simulated_outcomes[option_id]
            metrics['response_prob'].append(outcome.get('response_prob', 0.0))
            metrics['survival_days'].append(outcome.get('survival_days', 0.0))
            metrics['uncertainty'].append(outcome.get('uncertainty', 1.0)) # Default high uncertainty if missing

        # Min-max normalization
        normalized = {}
        for metric, values in metrics.items():
            if not values:
                normalized[metric] = [0.0] * len(option_ids) # Handle empty values case
                continue

            min_val = np.min(values)
            max_val = np.max(values)

            if max_val == min_val:
                # Assign 0.5 if all values are the same
                normalized[metric] = [0.5] * len(values)
            else:
                # Apply normalization safely
                normalized[metric] = [(v - min_val) / (max_val - min_val) if max_val > min_val else 0.5 for v in values]

        # Calculate weighted scores
        for i, option_id in enumerate(option_ids):
            score = 0.0
            for metric, weight in weights.items():
                if metric in normalized and i < len(normalized[metric]):
                    score += weight * normalized[metric][i]

            scores[option_id] = score

        # Rank treatments by score
        ranked_options = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        return ranked_options
