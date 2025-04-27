# SUSRUTA 🧠

![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![License](https://img.shields.io/badge/license-MIT-green)

**S**ystem for **U**nified graph-based neuropathology **S**imulation **U**sing **T**reatment **A**nalytics


> Memory-efficient graph-based glioma treatment analysis that runs on 8GB RAM

## 🚀 What is SUSRUTA?

SUSRUTA is a memory-optimized graph neural network system that helps neuro-oncologists predict treatment outcomes for glioma patients. Built to run efficiently on standard hardware (including MacBooks with 8GB RAM), it combines multimodal medical data (MRI, clinical records, genomics) into a knowledge graph to simulate alternative treatment outcomes and provide explainable AI-driven recommendations.

## ✨ Key Features

- **Memory Efficient** - Process large MRI volumes on standard hardware (8GB RAM)
- **Multimodal Integration** - Combine imaging, clinical data, and genomics in a single graph
- **Treatment Simulation** - Predict outcomes of different treatment options
- **Explainable AI** - Understand why the model makes specific predictions
- **Temporal Analysis** - Track tumor progression over multiple timepoints
- **Interactive Visualization** - Explore tumor regions and treatment comparisons

## 📋 Data Requirements

SUSRUTA is designed to work with the [MU-Glioma-Post dataset](https://doi.org/10.7937/7k9k-3c83) available from The Cancer Imaging Archive (TCIA).

### 📜 TCIA Data Usage Policy

**Important:** When using the MU-Glioma-Post dataset, you must comply with the TCIA Data Usage Policy:

- **Never attempt to identify patients** in the dataset
- **Acknowledge the source** in all presentations or publications
- **Include the DOI** (https://doi.org/10.7937/7k9k-3c83) in citations
- Follow the detailed [TCIA Data Usage Policy](https://www.cancerimagingarchive.net/data-usage-policies-and-restrictions/)

## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/YourUsername/susruta.git
cd susruta

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the base package
pip install -e .

# With visualization tools
pip install -e .[viz]

# With development tools
pip install -e .[dev]
```

## 📊 Usage Examples

### Processing MRI Data

```python
from susruta.data import EfficientMRIProcessor

# Initialize processor with memory limits
processor = EfficientMRIProcessor(memory_limit_mb=2000)

# Extract features from a single patient/timepoint
features = processor.extract_features_for_patient(
    patient_id=3,
    data_dir="path/to/PatientID_0003/Timepoint_1",
    timepoint=1
)

print(f"Extracted features for T1c: {features['t1c']}")
print(f"Tumor volume: {features['tumor']['volume_mm3']} mm³")
```

### Building the Knowledge Graph

```python
from susruta.graph import GliomaKnowledgeGraph
import pandas as pd

# Load clinical data
clinical_data = pd.read_excel("MUGliomaPost_ClinicalDataFINAL032025.xlsx")

# Create graph builder
kg_builder = GliomaKnowledgeGraph(memory_limit_mb=3000)

# Add patient/clinical data
kg_builder.add_clinical_data(clinical_data)

# Add tumor features from MRI
kg_builder.add_imaging_features(features_dict)

# Add treatment data
treatments_df = pd.read_excel("treatments.xlsx")
kg_builder.add_treatments(treatments_df)

# Create similarity connections
kg_builder.add_similarity_edges()

# Convert to PyTorch Geometric format
pyg_data = kg_builder.to_pytorch_geometric()
```

### Treatment Simulation and Explanation

```python
from susruta.models import GliomaGNN
from susruta.treatment import TreatmentSimulator
from susruta.viz import ExplainableGliomaTreatment

# Create and train model (or load pre-trained)
model = GliomaGNN(
    node_feature_dims={node_type: data.x.size(1) for node_type, data in pyg_data.items()},
    hidden_channels=32
)
# model.load_state_dict(torch.load("pretrained_model.pt"))

# Simulate treatment options
simulator = TreatmentSimulator(model, kg_builder.G)
treatment_options = [
    {'category': 'surgery', 'intensity': 0.8, 'duration': 1},
    {'category': 'radiation', 'dose': 60.0, 'duration': 30},
    {'category': 'chemotherapy', 'dose': 150, 'duration': 120}
]
results = simulator.simulate_treatments('patient_3', 'tumor_3', treatment_options, pyg_data)

# Generate explanation for best option
explainer = ExplainableGliomaTreatment(model, pyg_data, kg_builder.G)
explanation = explainer.explain_treatment_prediction(0)
print(explainer.generate_natural_language_explanation(explanation))
```

## 📁 Project Structure

```
susruta/
├── data/                 # MRI and clinical data processing
│   ├── mri.py            # Efficient MRI processing
│   ├── clinical.py       # Clinical data processing
│   └── excel_loader.py   # Excel data integration
├── graph/                # Knowledge graph construction
│   └── knowledge_graph.py# Main graph builder
├── graph_builder/        # Specialized graph builders
│   ├── mri_graph.py      # MRI-specific graph building
│   ├── temporal_graph.py # Temporal progression graph
│   └── visualization.py  # Graph visualization tools
├── models/               # Neural network models
│   └── gnn.py            # Graph neural network implementation
├── treatment/            # Treatment simulation
│   └── simulator.py      # Treatment outcome simulation
├── viz/                  # Visualization and explainability
│   └── explainable.py    # Explanation generation
└── utils/                # Utility functions
    └── memory.py         # Memory management
```

## 🔍 System Requirements

- **Python:** 3.9+
- **RAM:** 8GB minimum (designed for MacBook Air M1)
- **GPU:** Optional (CPU-only mode available)
- **OS:** Windows, macOS, or Linux
- **Dependencies:**
  - PyTorch 2.0+
  - PyTorch Geometric
  - NetworkX
  - SimpleITK & nibabel (MRI processing)
  - Pandas & NumPy
  - Matplotlib & Plotly (visualization)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Data Acknowledgment

This project utilizes the [MU-Glioma-Post dataset](https://doi.org/10.7937/7k9k-3c83) from The Cancer Imaging Archive (TCIA):

> Yaseen, D., Garrett, F., Gass, J., Greaser, J., Isufi, E., Layfield, L. J., Nada, A., Porgorzelski, K., Sinclair, J., Tahon, N. H. M., & Thacker, J. (2025). University of Missouri Post-operative Glioma Dataset (MU-Glioma-Post) (Version 1) [Data set]. The Cancer Imaging Archive. https://doi.org/10.7937/7k9k-3c83