"""
Treatment simulation and counterfactual reasoning for glioma outcome prediction.

Implements memory-efficient treatment simulation and ranking based on graph structure.
"""

from typing import Dict, List, Tuple, Any, Optional, Union
import torch
import torch.nn.functional as F
import networkx as nx
from torch_geometric.data import HeteroData

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
        self.model.eval()  # Set model to evaluation mode
    
    def simulate_treatments(self, 
                          patient_id: str, 
                          tumor_id: str, 
                          treatment_options: List[Dict[str, Any]], 
                          data: HeteroData) -> Dict[str, Dict[str, Any]]:
        """
        Simulate outcomes for different treatment options.
        
        Args:
            patient_id: Patient node ID
            tumor_id: Tumor node ID
            treatment_options: List of treatment configurations to simulate
            data: PyTorch Geometric data object
            
        Returns:
            Dictionary of predicted outcomes for each treatment option
        """
        results = {}
        
        # Find node indices
        if hasattr(data['patient'], 'original_ids'):
            patient_idx = data['patient'].original_ids.index(patient_id)
            tumor_idx = data['tumor'].original_ids.index(tumor_id)
        else:
            # If original_ids not available, assume indices match
            patient_idx = int(patient_id.split('_')[1])
            tumor_idx = int(tumor_id.split('_')[1])
        
        # Get patient and tumor embeddings
        with torch.no_grad():
            # Forward pass to get node embeddings
            _, node_embeddings = self.model(
                {k: v.x.to(self.device) for k, v in data.items()},
                {k: v.edge_index.to(self.device) for k, v in data.edge_types().items()}
            )
        
        # Get tumor embedding
        tumor_embedding = node_embeddings['tumor'][tumor_idx]
        
        # Simulate each treatment
        for treatment_idx, treatment_config in enumerate(treatment_options):
            # Create synthetic treatment node with configuration
            treatment_features = self._encode_treatment(treatment_config)
            
            # Add this treatment to the graph
            treatment_id = f"treatment_sim_{treatment_idx}"
            
            # Connect tumor to treatment
            # This is a simplified version; in practice, you'd modify the PyG data object
            # Here we're creating a synthetic treatment embedding
            
            # Average treatment embeddings of similar type
            similar_treatments = self._find_similar_treatments(treatment_config, data)
            if similar_treatments:
                # Use average embedding of similar treatments
                treatment_embedding = torch.stack(similar_treatments).mean(dim=0)
            else:
                # Create new embedding from treatment encoder
                treatment_embedding = self.model.node_encoders['treatment'](
                    torch.tensor(treatment_features, dtype=torch.float).to(self.device)
                )
            
            # Predict outcomes using the model's prediction heads
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
        # This is a simplified encoding, adapt to your feature space
        features = []
        
        # Encode treatment category (one-hot)
        categories = ['surgery', 'radiation', 'chemotherapy', 'combined']
        category_vec = [0] * len(categories)
        if treatment_config.get('category') in categories:
            category_vec[categories.index(treatment_config['category'])] = 1
        features.extend(category_vec)
        
        # Encode numerical features
        for key in ['dose', 'duration', 'intensity']:
            if key in treatment_config:
                features.append(float(treatment_config[key]))
            else:
                features.append(0.0)
        
        return features
    
    def _find_similar_treatments(self, 
                               treatment_config: Dict[str, Any], 
                               data: HeteroData) -> List[torch.Tensor]:
        """
        Find embeddings of similar treatments in the graph.
        
        Args:
            treatment_config: Treatment configuration
            data: PyTorch Geometric data object
            
        Returns:
            List of embeddings for similar treatments
        """
        similar_treatments = []
        
        # Get treatment category
        category = treatment_config.get('category')
        if not category:
            return similar_treatments
        
        # This is a simplified version; in practice, you'd query the graph structure
        # For demonstration, we'll just look for treatments of the same category
        # that are close in feature space
        
        # This implementation depends on your data structure
        if 'treatment' in data:
            if hasattr(data['treatment'], 'original_ids'):
                # Forward pass to get all node embeddings
                with torch.no_grad():
                    _, node_embeddings = self.model(
                        {k: v.x.to(self.device) for k, v in data.items()},
                        {k: v.edge_index.to(self.device) for k, v in data.edge_types().items()}
                    )
                
                for idx, treatment_id in enumerate(data['treatment'].original_ids):
                    if treatment_id in self.graph.nodes():
                        treatment_attrs = self.graph.nodes[treatment_id]
                        if treatment_attrs.get('category') == category:
                            # Get the embedding
                            similar_treatments.append(node_embeddings['treatment'][idx])
        
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
        
        for option_id, outcome in simulated_outcomes.items():
            metrics['response_prob'].append(outcome['response_prob'])
            metrics['survival_days'].append(outcome['survival_days'])
            metrics['uncertainty'].append(outcome['uncertainty'])
        
        # Min-max normalization
        normalized = {}
        for metric, values in metrics.items():
            if not values:
                continue
                
            min_val = min(values)
            max_val = max(values)
            
            if max_val == min_val:
                normalized[metric] = [0.5] * len(values)
            else:
                normalized[metric] = [(v - min_val) / (max_val - min_val) for v in values]
        
        # Calculate weighted scores
        for i, option_id in enumerate(simulated_outcomes.keys()):
            score = 0
            for metric, weight in weights.items():
                if metric in normalized:
                    score += weight * normalized[metric][i]
            
            scores[option_id] = score
        
        # Rank treatments by score
        ranked_options = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        
        return ranked_options