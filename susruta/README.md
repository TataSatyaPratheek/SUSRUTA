# SUSRUTA: System for Unified Graph-based Heuristic Recommendation Using Treatment Analytics

SUSRUTA is a memory-efficient, graph-based clinical decision support system for glioma treatment outcome prediction. It processes multimodal data (MRI, clinical, genomic) to build a knowledge graph and uses Graph Neural Networks to predict treatment responses and provide personalized therapeutic recommendations.

## Features

- **Memory-efficient MRI processing**: Processes large MRI volumes within 8GB RAM constraints
- **Heterogeneous knowledge graph construction**: Represents complex relationships between patients, tumors, treatments, and outcomes
- **Graph neural network modeling**: Predicts treatment responses and survival outcomes
- **Counterfactual treatment simulation**: Evaluates alternative treatment options
- **Explainable recommendations**: Provides transparent reasoning for predictions

## Installation

```bash
# Install from PyPI
pip install susruta

# Install with visualization dependencies
pip install susruta[viz]

# Install development dependencies
pip install susruta[dev]

# Quick start
```python
from susruta.data import EfficientMRIProcessor, ClinicalDataProcessor
from susruta.graph import GliomaKnowledgeGraph
from susruta.models import GliomaGNN
from susruta.treatment import TreatmentSimulator
from susruta.viz import ExplainableGliomaTreatment

# 1. Process MRI data efficiently
mri_processor = EfficientMRIProcessor(memory_limit_mb=2000)
patient_features = mri_processor.extract_features_for_patient(
    patient_id=1,
    data_dir='path/to/mri/data',
    timepoint=1
)

# 2. Process clinical data
clinical_processor = ClinicalDataProcessor()
clinical_df = clinical_processor.preprocess_clinical_data(raw_clinical_data)

# 3. Build knowledge graph
kg_builder = GliomaKnowledgeGraph(memory_limit_mb=3000)
kg_builder.add_clinical_data(clinical_df)
kg_builder.add_imaging_features(imaging_features)
kg_builder.add_treatments(treatments_df)
kg_builder.add_similarity_edges()

# 4. Convert to PyTorch Geometric format
pyg_data = kg_builder.to_pytorch_geometric()

# 5. Initialize and train model
model = GliomaGNN(
    node_feature_dims={node_type: data.x.size(1) for node_type, data in pyg_data.items()},
    hidden_channels=32
)
# ... train model ...

# 6. Simulate treatments
treatment_simulator = TreatmentSimulator(model, kg_builder.G)
treatment_options = [
    {'category': 'surgery', 'intensity': 0.8, 'duration': 1},
    {'category': 'radiation', 'dose': 60.0, 'duration': 30},
    {'category': 'chemotherapy', 'dose': 150, 'duration': 120}
]
results = treatment_simulator.simulate_treatments('patient_1', 'tumor_1', treatment_options, pyg_data)

# 7. Generate explanations
explainer = ExplainableGliomaTreatment(model, pyg_data, kg_builder.G)
explanation = explainer.explain_treatment_prediction(0)
explanation_text = explainer.generate_natural_language_explanation(explanation)
print(explanation_text)
```

# System Requirements

- Python 3.9+
- 8GB RAM minimum (works within memory constraints of M1 MacBook Air)
- GPU is optional but recommended for faster model training

# Documentation
Detailed documentation is available at https://docs.susruta.io

# Citation
If you use SUSRUTA in your research, please cite:
@article{susruta2024,
  title={SUSRUTA: A Memory-Efficient Graph-Based Clinical Decision Support System for Glioma Treatment Outcome Prediction},
  author={Your Name},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2024}
}

# License
This project is licensed under the MIT License - see the LICENSE file for details.