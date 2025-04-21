susruta/
├── pyproject.toml         # Modern package definition
├── src/                   # Source code in src layout
│   └── susruta/           # Main package
│       ├── __init__.py    # Package initialization
│       ├── data/          # Data processing modules
│       │   ├── __init__.py
│       │   ├── mri.py     # MRI processing
│       │   └── clinical.py # Clinical data processing
│       ├── graph/         # Graph-related modules
│       │   ├── __init__.py
│       │   └── knowledge_graph.py # Knowledge graph construction
│       ├── models/        # ML model modules
│       │   ├── __init__.py
│       │   └── gnn.py     # Graph neural network model
│       ├── treatment/     # Treatment modules
│       │   ├── __init__.py
│       │   └── simulator.py # Treatment simulation
│       ├── viz/           # Visualization modules
│       │   ├── __init__.py
│       │   └── explain.py # Explanation visualization
│       └── utils/         # Utility modules
│           ├── __init__.py
│           └── memory.py  # Memory tracking utilities
├── tests/                 # Test directory
│   ├── __init__.py
│   ├── conftest.py        # Test configuration
│   ├── test_data.py       # Data processing tests
│   ├── test_graph.py      # Graph construction tests
│   ├── test_models.py     # Model tests
│   └── test_treatment.py  # Treatment simulation tests
├── examples/              # Example usage scripts
│   ├── data_processing.py
│   ├── graph_construction.py
│   ├── model_training.py
│   └── treatment_recommendation.py
├── README.md              # Project README
└── LICENSE                # License file