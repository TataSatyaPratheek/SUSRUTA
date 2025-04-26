# susruta/src/susruta/data/__init__.py
"""
Data processing module for glioma treatment outcome prediction.

Handles MRI, clinical, and Excel data loading, processing, and integration.
"""

# Ensure all necessary classes are imported and listed in __all__
from .mri import EfficientMRIProcessor
from .clinical import ClinicalDataProcessor
from .excel_loader import ExcelDataLoader
from .excel_integration import MultimodalDataIntegrator

__all__ = [
    "EfficientMRIProcessor",    # From Phase 1
    "ClinicalDataProcessor",    # For Phase 2 and beyond
    "ExcelDataLoader",          # For Phase 2 and beyond
    "MultimodalDataIntegrator", # For Phase 2 and beyond
]
