"""Example script for processing MRI and clinical data with memory efficiency."""

import os
import pandas as pd
import numpy as np
from susruta.data import EfficientMRIProcessor, ClinicalDataProcessor
from susruta.utils import MemoryTracker


def create_synthetic_clinical_data():
    """Create synthetic clinical data for demonstration."""
    clinical_data = pd.DataFrame({
        'patient_id': range(1, 31),
        'age': np.random.randint(30, 80, 30),
        'sex': np.random.choice(['M', 'F'], 30),
        'karnofsky_score': np.random.randint(60, 100, 30),
        'grade': np.random.choice(['II', 'III', 'IV'], 30),
        'histology': np.random.choice(['Astrocytoma', 'Oligodendroglioma', 'GBM'], 30),
        'location': np.random.choice(['Frontal', 'Temporal', 'Parietal'], 30),
        'idh_mutation': np.random.choice([0, 1], 30),
        'mgmt_methylation': np.random.choice([0, 1], 30)
    })
    return clinical_data


def main():
    """Run example data processing pipeline."""
    tracker = MemoryTracker()
    
    print("Starting data processing example...")
    tracker.log_memory("Initial")
    
    # ===== MRI Processing =====
    # For demonstration, we'll simulate processing without actual MRI files
    print("\nSimulating MRI processing...")
    
    mri_processor = EfficientMRIProcessor(memory_limit_mb=2000)
    
    # In a real scenario, you would use:
    # features = mri_processor.extract_features_for_patient(
    #     patient_id=123,
    #     data_dir='path/to/mri/data',
    #     timepoint=1
    # )
    
    # For simulation, create synthetic features
    imaging_features = {}
    for patient_id in range(1, 31):
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
    
    tracker.log_memory("After MRI processing")
    
    # ===== Clinical Data Processing =====
    print("\nProcessing clinical data...")
    
    # Create synthetic clinical data
    clinical_data = create_synthetic_clinical_data()
    
    # Process clinical data
    clinical_processor = ClinicalDataProcessor()
    processed_clinical = clinical_processor.preprocess_clinical_data(clinical_data)
    
    print(f"Clinical data shape: {processed_clinical.shape}")
    print(f"Clinical data columns: {processed_clinical.columns.tolist()[:5]}...")
    
    tracker.log_memory("After clinical processing")
    
    # ===== Create Treatments =====
    print("\nCreating synthetic treatment data...")
    
    # Create synthetic treatment data
    treatments = []
    for patient_id in range(1, 31):
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
            response = np.random.choice(['CR', 'PR', 'SD', 'PD'])
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
    
    # Process treatment data
    processed_treatments = clinical_processor.process_treatment_data(treatments_df)
    
    print(f"Treatment data shape: {processed_treatments.shape}")
    print(f"Treatment data columns: {processed_treatments.columns.tolist()[:5]}...")
    
    tracker.log_memory("After treatment processing")
    
    # ===== Data Integration =====
    print("\nIntegrating multimodal data...")
    
    integrated_data = clinical_processor.integrate_multimodal_data(
        processed_clinical, 
        imaging_features
    )
    
    print(f"Integrated data shape: {integrated_data.shape}")
    print(f"Total features: {len(integrated_data.columns)}")
    
    tracker.log_memory("After data integration")
    
    # ===== Memory Usage Summary =====
    print("\nMemory Usage Summary:")
    tracker.plot_memory_usage()
    
    print("\nData processing example completed successfully!")


if __name__ == "__main__":
    main()