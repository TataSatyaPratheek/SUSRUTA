# susruta/src/susruta/__init__.py
"""
SUSRUTA: System for Unified Graph-based Heuristic Recommendation Using Treatment Analytics.

A memory-efficient, graph-based clinical decision support system for glioma treatment outcome prediction.
"""

__version__ = "0.1.1" # Updated to match pyproject.toml

# Import submodules to make them available within the package structure
# (e.g., for relative imports within susruta itself if needed)
from . import data
from . import graph
from . import models
from . import treatment
from . import viz
from . import utils

# Define __all__ to control 'from susruta import *' behavior.
# It's generally better to encourage explicit submodule imports,
# so we can leave this empty or list only truly essential top-level items.
# For now, let's just expose the version.
__all__ = [
    "__version__",
    # Add other top-level exports here if absolutely necessary,
    # but prefer submodule imports like:
    # from susruta.data import ClinicalDataProcessor
]

# Optional: Clean up namespace after imports if submodules aren't needed directly
# del data, graph, models, treatment, viz, utils
