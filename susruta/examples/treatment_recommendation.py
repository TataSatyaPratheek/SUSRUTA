"""Example script for generating and explaining treatment recommendations."""

import os
import gc
import time
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from susruta.data import ClinicalDataProcessor
from susruta.graph import GliomaKnowledgeGraph
from susruta.models import GliomaGNN
from susruta.treatment import TreatmentSimulator
from susruta.viz import ExplainableGliomaTreatment
from susruta.utils import MemoryTracker


def create_synthetic_data():
    """Create synthetic data for demonstration."""
    # Create synthetic clinical data for a smaller test set
    clinical_data = pd.DataFrame({
        'patient_id': range(1, 11),
        'age': np.random.randint(30, 80, 10),
        'sex': np.random.choice(['M', 'F'], 10),
        'karnofsky_score': np.random.randint(60, 100, 10),
        'grade': np.random.choice(['II', 'III', 'IV'], 10),
        'histology': np.random.choice(['Astrocytoma', 'Oligodendroglioma', 'GBM'], 10),
        'location': np.random.choice(['Frontal', 'Temporal', 'Parietal'], 10),
        'idh_mutation': np.random.choice([0, 1], 10),
        'mgmt_methylation': np.random.choice([0, 1], 10)
    })
    
    # Create synthetic treatment data
    treatments = []
    for patient_id in range(1, 11):
        # Each patient gets 1-3 treatments
        num_treatments = np.random.randint(1, 4)
        for i in range(num_treatments):
            treatment_id = len(treatments) + 1
            category = np.random.choice(['surgery', 'radiation', 'chemotherapy'])
            
            # Add treatment specifics based on category
            if category == 'surgery':
                treatment_name = np.random.choice(['Gross total resection', 'Subtotal resection'])
                dose = None
            elif category == 'radiation':
                treatment_name = 'External beam radiation'
                dose = np.random.choice([45.0, 54.0, 60.0])  # Gy
            else:  # chemotherapy
                treatment_name = np.random.choice(['Temozolomide', 'PCV', 'Bevacizumab'])
                dose = np.random.randint(100, 200)  # mg/m²
            
            duration_days = np.random.randint(1, 180)
            start_day = np.random.randint(0, 100)
            
            # Add outcome
            response = np.random.choice([0, 1], p=[0.3, 0.7])  # Binary outcome for simplicity
            progression_free_days = np.random.randint(30, 1000)
            survival_days = progression_free_days + np.random.randint(0, 500)
            
            treatments.append({
                'patient_id': patient_id,
                'treatment_id': treatment_id,
                'category': category,
                'treatment_name': treatment_name,
                'dose': dose,
                'duration_days': duration_days,
                'start_day': start_day,
                'response': response,
                'progression_free_days': progression_free_days,
                'survival_days': survival_days
            })
    
    treatments_df = pd.DataFrame(treatments)
    
    # Create synthetic imaging features
    imaging_features = {}
    for patient_id in range(1, 11):
        patient_features = {}
        
        # T1c features
        patient_features['t1c'] = {
            'mean': np.random.uniform(100, 200),
            'std': np.random.uniform(10, 50),
            'max': np.random.uniform(200, 300),
            'volume_voxels': np.random.randint(1000, 5000)
        }
        
        # T2 features
        patient_features['t2w'] = {
            'mean': np.random.uniform(150, 250),
            'std': np.random.uniform(20, 60),
            'max': np.random.uniform(250, 350),
            'volume_voxels': np.random.randint(1000, 5000)
        }
        
        # Tumor features
        patient_features['tumor'] = {
            'volume_mm3': np.random.uniform(1000, 30000),
            'surface_area': np.random.uniform(500, 5000),
            'elongation': np.random.uniform(0.2, 0.8),
            'roundness': np.random.uniform(0.3, 0.9)
        }
        
        imaging_features[patient_id] = patient_features
    
    return clinical_data, treatments_df, imaging_features


def build_knowledge_graph(clinical_data, treatments_df, imaging_features):
    """Build knowledge graph from data."""
    kg_builder = GliomaKnowledgeGraph(memory_limit_mb=3000)
    
    # Add data to graph
    kg_builder.add_clinical_data(clinical_data)
    kg_builder.add_imaging_features(imaging_features)
    kg_builder.add_treatments(treatments_df)
    kg_builder.add_similarity_edges()
    
    # Convert to PyTorch Geometric
    pyg_data = kg_builder.to_pytorch_geometric()
    
    return kg_builder.G, pyg_data


def create_treatment_options():
    """Create a set of candidate treatment options for simulation."""
    treatment_options = [
        {
            'category': 'surgery',
            'name': 'Gross total resection',
            'intensity': 0.9,
            'duration': 1,
            'description': 'Complete removal of visible tumor'
        },
        {
            'category': 'surgery',
            'name': 'Subtotal resection',
            'intensity': 0.6,
            'duration': 1,
            'description': 'Partial removal of tumor'
        },
        {
            'category': 'radiation',
            'name': 'Standard radiation therapy',
            'dose': 60.0,
            'duration': 30,
            'description': 'External beam radiation, 2 Gy per fraction, 30 fractions'
        },
        {
            'category': 'radiation',
            'name': 'Hypofractionated radiation',
            'dose': 40.0,
            'duration': 15,
            'description': 'Higher dose per fraction, shorter overall treatment time'
        },
        {
            'category': 'chemotherapy',
            'name': 'Temozolomide',
            'dose': 150,
            'duration': 180,
            'description': 'Alkylating agent, standard of care for glioblastoma'
        },
        {
            'category': 'chemotherapy',
            'name': 'PCV',
            'dose': 110,
            'duration': 120,
            'description': 'Procarbazine, CCNU, and vincristine combination'
        },
        {
            'category': 'chemotherapy',
            'name': 'Bevacizumab',
            'dose': 10,
            'duration': 150,
            'description': 'VEGF inhibitor, useful for recurrent disease'
        }
    ]
    
    return treatment_options


def initialize_model(pyg_data, model_path=None):
    """Initialize GNN model and load weights if available."""
    # Initialize model with data dimensions
    node_feature_dims = {node_type: data.x.size(1) for node_type, data in pyg_data.items()}
    
    # Initialize edge_feature_dims
    edge_feature_dims = {}
    for edge_type in pyg_data.edge_types():
        edge_feature_dims[edge_type] = 1  # Default edge feature dimension
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GliomaGNN(
        node_feature_dims=node_feature_dims,
        edge_feature_dims=edge_feature_dims,
        hidden_channels=32,
        dropout=0.3
    ).to(device)
    
    # Load pre-trained weights if available
    if model_path and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded pre-trained model from {model_path}")
    else:
        print("Using randomly initialized model (no pre-trained weights found)")
        # In a real scenario, you would train the model first
    
    # Set model to evaluation mode
    model.eval()
    
    return model, device


def plot_treatment_comparison(simulated_results, treatment_options, patient_id, tumor_grade):
    """Create visualizations for treatment comparison."""
    plt.figure(figsize=(15, 10))
    
    # Extract data for plotting
    options = []
    response_probs = []
    survival_days = []
    uncertainties = []
    
    for option_id, result in simulated_results.items():
        option_idx = int(option_id.split('_')[1])
        options.append(treatment_options[option_idx]['name'])
        response_probs.append(result['response_prob'] * 100)  # Convert to percentage
        survival_days.append(result['survival_days'])
        uncertainties.append(result['uncertainty'])
    
    # Sort by survival days descending
    sort_indices = np.argsort(survival_days)[::-1]
    options = [options[i] for i in sort_indices]
    response_probs = [response_probs[i] for i in sort_indices]
    survival_days = [survival_days[i] for i in sort_indices]
    uncertainties = [uncertainties[i] for i in sort_indices]
    
    # Plot response probability
    plt.subplot(2, 2, 1)
    bars = plt.barh(options, response_probs, color='skyblue')
    plt.xlabel('Response Probability (%)')
    plt.ylabel('Treatment Option')
    plt.title('Predicted Response Probability')
    plt.xlim(0, 100)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width + 2, bar.get_y() + bar.get_height()/2, 
                f'{response_probs[i]:.1f}%', 
                va='center')
    
    # Plot survival days
    plt.subplot(2, 2, 2)
    bars = plt.barh(options, survival_days, color='lightgreen')
    plt.xlabel('Predicted Survival (days)')
    plt.ylabel('Treatment Option')
    plt.title('Predicted Survival Duration')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width + 10, bar.get_y() + bar.get_height()/2, 
                f'{survival_days[i]:.0f}', 
                va='center')
    
    # Plot uncertainty
    plt.subplot(2, 2, 3)
    bars = plt.barh(options, uncertainties, color='salmon')
    plt.xlabel('Prediction Uncertainty')
    plt.ylabel('Treatment Option')
    plt.title('Prediction Uncertainty')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{uncertainties[i]:.2f}', 
                va='center')
    
    # Plot risk-benefit chart
    plt.subplot(2, 2, 4)
    colors = plt.cm.viridis(np.array(uncertainties) / max(uncertainties))
    plt.scatter(response_probs, survival_days, c=colors, s=100, alpha=0.7)
    
    # Add labels to each point
    for i, option in enumerate(options):
        plt.annotate(option, 
                    (response_probs[i], survival_days[i]),
                    xytext=(5, 5),
                    textcoords='offset points')
    
    plt.xlabel('Response Probability (%)')
    plt.ylabel('Survival (days)')
    plt.title('Risk-Benefit Analysis')
    plt.grid(True, linestyle='--', alpha=0.7)
    cbar = plt.colorbar()
    cbar.set_label('Uncertainty')
    
    # Add title
    plt.suptitle(f'Treatment Comparison for Patient {patient_id} (Grade {tumor_grade} Glioma)', 
                fontsize=16)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Save the figure
    plt.savefig(f'patient_{patient_id}_treatment_comparison.png')
    print(f"Treatment comparison plot saved as 'patient_{patient_id}_treatment_comparison.png'")
    
    return plt.gcf()


def main():
    """Run treatment recommendation and explanation example."""
    # Initialize memory tracker
    tracker = MemoryTracker()
    tracker.log_memory("Initial")
    
    print("=== Treatment Recommendation Example ===")
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Create synthetic data
    print("\nCreating synthetic data...")
    clinical_data, treatments_df, imaging_features = create_synthetic_data()
    tracker.log_memory("After data creation")
    
    # Build knowledge graph
    print("\nBuilding knowledge graph...")
    graph, pyg_data = build_knowledge_graph(clinical_data, treatments_df, imaging_features)
    tracker.log_memory("After graph construction")
    
    # Initialize model
    print("\nInitializing GNN model...")
    model, device = initialize_model(pyg_data, model_path='glioma_gnn_model.pt')
    tracker.log_memory("After model initialization")
    
    # Create treatment simulator
    print("\nCreating treatment simulator...")
    simulator = TreatmentSimulator(model, graph, device)
    tracker.log_memory("After simulator creation")
    
    # Create explainable treatment module
    print("\nCreating explainable treatment module...")
    explainer = ExplainableGliomaTreatment(model, pyg_data, graph, device)
    tracker.log_memory("After explainer creation")
    
    # Create treatment options
    treatment_options = create_treatment_options()
    
    # Select a patient for demonstration
    patient_id = 3  # Example patient ID
    patient_row = clinical_data[clinical_data['patient_id'] == patient_id].iloc[0]
    tumor_grade = patient_row['grade']
    
    print(f"\nGenerating treatment recommendations for Patient {patient_id} (Grade {tumor_grade} Glioma)...")
    print(f"  Age: {patient_row['age']}, Sex: {patient_row['sex']}")
    print(f"  Karnofsky Score: {patient_row['karnofsky_score']}")
    print(f"  Tumor Location: {patient_row['location']}")
    print(f"  Histology: {patient_row['histology']}")
    print(f"  IDH Mutation: {patient_row['idh_mutation']}")
    print(f"  MGMT Methylation: {patient_row['mgmt_methylation']}")
    
    # Generate treatment recommendations
    simulated_results = simulator.simulate_treatments(
        f"patient_{patient_id}", 
        f"tumor_{patient_id}", 
        treatment_options, 
        pyg_data
    )
    tracker.log_memory("After treatment simulation")
    
    # Rank treatments
    ranked_treatments = simulator.rank_treatments(simulated_results)
    
    # Display results
    print("\nTreatment Options (Ranked):")
    for i, option_id in enumerate(ranked_treatments):
        option_idx = int(option_id.split('_')[1])
        result = simulated_results[option_id]
        
        print(f"\n{i+1}. {treatment_options[option_idx]['name']} ({treatment_options[option_idx]['category']}):")
        print(f"   Description: {treatment_options[option_idx]['description']}")
        print(f"   Response Probability: {result['response_prob']*100:.1f}%")
        print(f"   Predicted Survival: {result['survival_days']:.0f} days")
        print(f"   Uncertainty: {result['uncertainty']:.3f}")
    
    # Generate explanation for top treatment
    top_option_id = ranked_treatments[0]
    option_idx = int(top_option_id.split('_')[1])
    
    # Get the index of existing treatment with same category for explanation
    existing_treatments = treatments_df[treatments_df['patient_id'] == patient_id]
    similar_category_treatments = existing_treatments[
        existing_treatments['category'] == treatment_options[option_idx]['category']
    ]
    
    if len(similar_category_treatments) > 0:
        treatment_idx = 0  # Use first available treatment for explanation
        print("\nGenerating explanation for top treatment...")
        explanation = explainer.explain_treatment_prediction(treatment_idx)
        explanation_text = explainer.generate_natural_language_explanation(explanation)
        print("\nTreatment Explanation:")
        print(explanation_text)
    else:
        print("\nNo existing similar treatments found for detailed explanation.")
    
    # Plot treatment comparison
    plot_treatment_comparison(simulated_results, treatment_options, patient_id, tumor_grade)
    
    # Plot memory usage
    fig = tracker.plot_memory_usage()
    fig.savefig('treatment_recommendation_memory.png')
    print("\nMemory usage plot saved as 'treatment_recommendation_memory.png'")
    
    print("\nTreatment recommendation example completed successfully!")


if __name__ == "__main__":
    main()