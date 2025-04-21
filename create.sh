#!/bin/bash

# Create the main project directory and navigate into it
echo "Creating root directory: susruta/"
mkdir susruta
cd susruta || exit # Exit if cd fails

# Create top-level files
echo "Creating top-level files..."
touch pyproject.toml README.md LICENSE

# Create main directories
echo "Creating main directories: src/, tests/, examples/"
mkdir src tests examples

# Create src structure
echo "Creating src/ structure..."
mkdir -p src/susruta/data
mkdir -p src/susruta/graph
mkdir -p src/susruta/models
mkdir -p src/susruta/treatment
mkdir -p src/susruta/viz
mkdir -p src/susruta/utils

# Create files within src/susruta/
echo "Creating files within src/susruta/..."
touch src/susruta/__init__.py
touch src/susruta/data/__init__.py src/susruta/data/mri.py src/susruta/data/clinical.py
touch src/susruta/graph/__init__.py src/susruta/graph/knowledge_graph.py
touch src/susruta/models/__init__.py src/susruta/models/gnn.py
touch src/susruta/treatment/__init__.py src/susruta/treatment/simulator.py
touch src/susruta/viz/__init__.py src/susruta/viz/explain.py
touch src/susruta/utils/__init__.py src/susruta/utils/memory.py

# Create tests structure
echo "Creating tests/ structure..."
touch tests/__init__.py tests/conftest.py
touch tests/test_data.py tests/test_graph.py tests/test_models.py tests/test_treatment.py

# Create examples structure
echo "Creating examples/ structure..."
touch examples/data_processing.py examples/graph_construction.py examples/model_training.py examples/treatment_recommendation.py

echo ""
echo "Directory structure for 'susruta' created successfully in the current directory."

# Optional: Display the created structure using tree (if installed)
if command -v tree &> /dev/null
then
    echo "Displaying created structure:"
    tree .
else
    echo "Install 'tree' command to visualize the structure (e.g., 'sudo apt install tree' or 'brew install tree')."
fi

# Go back to the original directory (optional)
cd ..
