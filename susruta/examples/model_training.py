"""Example script for training the GNN model with memory efficiency."""

import os
import gc
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from susruta.data import ClinicalDataProcessor
from susruta.graph import GliomaKnowledgeGraph
from susruta.models import GliomaGNN
from susruta.utils import MemoryTracker


def create_synthetic_data():
    """Create synthetic data for demonstration."""
    # Create synthetic clinical data
    clinical_data = pd.DataFrame({
        'patient_id': range(1, 71),
        'age': np.random.randint(30, 80, 70),
        'sex': np.random.choice(['M', 'F'], 70),
        'karnofsky_score': np.random.randint(60, 100, 70),
        'grade': np.random.choice(['II', 'III', 'IV'], 70),
        'histology': np.random.choice(['Astrocytoma', 'Oligodendroglioma', 'GBM'], 70),
        'location': np.random.choice(['Frontal', 'Temporal', 'Parietal'], 70),
        'idh_mutation': np.random.choice([0, 1], 70),
        'mgmt_methylation': np.random.choice([0, 1], 70)
    })
    
    # Create synthetic treatment data
    treatments = []
    for patient_id in range(1, 71):
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
    for patient_id in range(1, 71):
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


def split_data_into_folds(clinical_data, treatments_df, imaging_features, test_size=0.2, val_size=0.1, random_state=42):
    """Split data into train/val/test sets at the patient level."""
    # Get unique patient IDs
    patient_ids = clinical_data['patient_id'].unique()
    
    # First split: train+val vs test
    train_val_patients, test_patients = train_test_split(
        patient_ids, test_size=test_size, random_state=random_state
    )
    
    # Second split: train vs val
    train_patients, val_patients = train_test_split(
        train_val_patients, test_size=val_size/(1-test_size), random_state=random_state
    )
    
    # Split clinical data
    train_clinical = clinical_data[clinical_data['patient_id'].isin(train_patients)]
    val_clinical = clinical_data[clinical_data['patient_id'].isin(val_patients)]
    test_clinical = clinical_data[clinical_data['patient_id'].isin(test_patients)]
    
    # Split treatments data
    train_treatments = treatments_df[treatments_df['patient_id'].isin(train_patients)]
    val_treatments = treatments_df[treatments_df['patient_id'].isin(val_patients)]
    test_treatments = treatments_df[treatments_df['patient_id'].isin(test_patients)]
    
    # Split imaging features
    train_imaging = {pid: imaging_features[pid] for pid in train_patients if pid in imaging_features}
    val_imaging = {pid: imaging_features[pid] for pid in val_patients if pid in imaging_features}
    test_imaging = {pid: imaging_features[pid] for pid in test_patients if pid in imaging_features}
    
    return {
        'train': (train_clinical, train_treatments, train_imaging),
        'val': (val_clinical, val_treatments, val_imaging),
        'test': (test_clinical, test_treatments, test_imaging),
    }


def build_graph_for_split(clinical_data, treatments_df, imaging_features, memory_limit_mb=3000):
    """Build knowledge graph for a specific data split."""
    kg_builder = GliomaKnowledgeGraph(memory_limit_mb=memory_limit_mb)
    
    # Add data to graph
    kg_builder.add_clinical_data(clinical_data)
    kg_builder.add_imaging_features(imaging_features)
    kg_builder.add_treatments(treatments_df)
    kg_builder.add_similarity_edges(threshold=0.6)
    
    # Convert to PyTorch Geometric
    pyg_data = kg_builder.to_pytorch_geometric()
    
    return kg_builder.G, pyg_data


def prepare_training_data(pyg_data, treatments_df):
    """Prepare target labels and indices for model training."""
    # Extract treatment indices and target labels
    treatment_indices = []
    response_labels = []
    survival_labels = []
    
    # Map treatment IDs to indices in PyG data
    if hasattr(pyg_data['treatment'], 'original_ids'):
        treatment_id_map = {tid: idx for idx, tid in enumerate(pyg_data['treatment'].original_ids)}
        
        for _, row in treatments_df.iterrows():
            treatment_id = f"treatment_{row['treatment_id']}"
            if treatment_id in treatment_id_map:
                treatment_indices.append(treatment_id_map[treatment_id])
                response_labels.append(float(row['response']))
                survival_labels.append(float(row['survival_days']))
    
    # Convert to tensors
    treatment_indices = torch.tensor(treatment_indices, dtype=torch.long)
    response_labels = torch.tensor(response_labels, dtype=torch.float).view(-1, 1)
    survival_labels = torch.tensor(survival_labels, dtype=torch.float).view(-1, 1)
    
    return treatment_indices, response_labels, survival_labels


def train_epoch(model, pyg_data, treatment_indices, response_labels, survival_labels, 
               optimizer, device, batch_size=16):
    """Train model for one epoch."""
    model.train()
    
    # Process in mini-batches to save memory
    perm = torch.randperm(len(treatment_indices))
    
    epoch_loss = 0
    num_batches = 0
    
    for i in range(0, len(treatment_indices), batch_size):
        # Get batch indices
        batch_indices = perm[i:i+batch_size]
        
        # Get batch data
        batch_treatment_indices = treatment_indices[batch_indices]
        batch_response_labels = response_labels[batch_indices].to(device)
        batch_survival_labels = survival_labels[batch_indices].to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        predictions, _ = model(
            {k: v.x.to(device) for k, v in pyg_data.items()},
            {k: v.edge_index.to(device) for k, v in pyg_data.edge_types().items()}
        )
        
        # Extract predictions for the batch
        batch_response_preds = predictions['response'][batch_treatment_indices]
        batch_survival_preds = predictions['survival'][batch_treatment_indices]
        
        # Compute loss
        response_loss = F.binary_cross_entropy(batch_response_preds, batch_response_labels)
        survival_loss = F.mse_loss(batch_survival_preds, batch_survival_labels)
        
        # Combined loss with weighting
        loss = response_loss + 0.01 * survival_loss
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        num_batches += 1
        
        # Free up memory
        del batch_response_preds, batch_survival_preds, loss
        torch.cuda.empty_cache() if torch.cuda.is_available() else gc.collect()
    
    return epoch_loss / num_batches


def evaluate(model, pyg_data, treatment_indices, response_labels, survival_labels, device):
    """Evaluate model on validation or test data."""
    model.eval()
    
    with torch.no_grad():
        # Forward pass
        predictions, _ = model(
            {k: v.x.to(device) for k, v in pyg_data.items()},
            {k: v.edge_index.to(device) for k, v in pyg_data.edge_types().items()}
        )
        
        # Extract predictions
        response_preds = predictions['response'][treatment_indices]
        survival_preds = predictions['survival'][treatment_indices]
        
        # Compute metrics
        response_loss = F.binary_cross_entropy(
            response_preds.to(device), 
            response_labels.to(device)
        ).item()
        
        survival_loss = F.mse_loss(
            survival_preds.to(device), 
            survival_labels.to(device)
        ).item()
        
        # Calculate accuracy for response prediction
        response_binary = (response_preds > 0.5).float()
        accuracy = (response_binary == response_labels.to(device)).float().mean().item()
    
    return {
        'response_loss': response_loss,
        'survival_loss': survival_loss,
        'response_accuracy': accuracy
    }


def main():
    """Run GNN model training example with memory efficiency."""
    # Initialize memory tracker
    tracker = MemoryTracker()
    tracker.log_memory("Initial")
    
    print("=== GNN Model Training Example ===")
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Select device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Create synthetic data
    print("\nCreating synthetic data...")
    clinical_data, treatments_df, imaging_features = create_synthetic_data()
    tracker.log_memory("After data creation")
    
    # Split data into train/val/test
    print("\nSplitting data into train/val/test sets...")
    data_splits = split_data_into_folds(clinical_data, treatments_df, imaging_features)
    tracker.log_memory("After data splitting")
    
    # Build knowledge graph for training data
    print("\nBuilding knowledge graph for training data...")
    train_clinical, train_treatments, train_imaging = data_splits['train']
    train_graph, train_pyg_data = build_graph_for_split(
        train_clinical, train_treatments, train_imaging
    )
    tracker.log_memory("After training graph construction")
    
    # Prepare training data
    print("\nPreparing training data...")
    train_indices, train_response, train_survival = prepare_training_data(
        train_pyg_data, train_treatments
    )
    tracker.log_memory("After preparing training data")
    
    # Build knowledge graph for validation data
    print("\nBuilding knowledge graph for validation data...")
    val_clinical, val_treatments, val_imaging = data_splits['val']
    val_graph, val_pyg_data = build_graph_for_split(
        val_clinical, val_treatments, val_imaging
    )
    tracker.log_memory("After validation graph construction")
    
    # Prepare validation data
    print("\nPreparing validation data...")
    val_indices, val_response, val_survival = prepare_training_data(
        val_pyg_data, val_treatments
    )
    tracker.log_memory("After preparing validation data")
    
    # Initialize model
    print("\nInitializing GNN model...")
    node_feature_dims = {node_type: data.x.size(1) for node_type, data in train_pyg_data.items()}
    
    # Initialize edge_feature_dims
    edge_feature_dims = {}
    for edge_type in train_pyg_data.edge_types():
        edge_feature_dims[edge_type] = 1  # Default edge feature dimension
    
    model = GliomaGNN(
        node_feature_dims=node_feature_dims,
        edge_feature_dims=edge_feature_dims,
        hidden_channels=32,
        dropout=0.3
    ).to(device)
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    
    # Training hyperparameters
    num_epochs = 10
    batch_size = 16
    
    # Training loop
    print("\nTraining model...")
    tracker.log_memory("Before training")
    
    train_losses = []
    val_metrics = []
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Train for one epoch
        train_loss = train_epoch(
            model, train_pyg_data, train_indices, train_response, train_survival,
            optimizer, device, batch_size
        )
        train_losses.append(train_loss)
        
        # Evaluate on validation set
        val_metric = evaluate(
            model, val_pyg_data, val_indices, val_response, val_survival, device
        )
        val_metrics.append(val_metric)
        
        # Print metrics
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f}, "
              f"Val Response Loss: {val_metric['response_loss']:.4f}, "
              f"Val Accuracy: {val_metric['response_accuracy']:.4f}, "
              f"Val Survival Loss: {val_metric['survival_loss']:.4f}, "
              f"Time: {epoch_time:.2f}s")
        
        # Log memory
        tracker.log_memory(f"After epoch {epoch+1}")
        
        # Force garbage collection
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Plot training curves
    print("\nPlotting training curves...")
    plt.figure(figsize=(12, 8))
    
    # Plot training loss
    plt.subplot(2, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    # Plot validation response loss
    plt.subplot(2, 2, 2)
    plt.plot([m['response_loss'] for m in val_metrics])
    plt.title('Validation Response Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    # Plot validation accuracy
    plt.subplot(2, 2, 3)
    plt.plot([m['response_accuracy'] for m in val_metrics])
    plt.title('Validation Response Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    
    # Plot validation survival loss
    plt.subplot(2, 2, 4)
    plt.plot([m['survival_loss'] for m in val_metrics])
    plt.title('Validation Survival Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    print("Training curves saved as 'training_curves.png'")
    
    # Plot memory usage
    fig = tracker.plot_memory_usage()
    fig.savefig('training_memory.png')
    print("Memory usage plot saved as 'training_memory.png'")
    
    # Save model
    torch.save(model.state_dict(), 'glioma_gnn_model.pt')
    print("Model saved as 'glioma_gnn_model.pt'")
    
    print("\nModel training example completed successfully!")


if __name__ == "__main__":
    main()