"""
Excel data integration with MRI and clinical workflows.

Connects external Excel data sources with the core data processing 
pipeline for a comprehensive multimodal analysis.
"""

from typing import Dict, List, Any, Optional, Union, Tuple
import os
import pandas as pd
import numpy as np
from pathlib import Path
import logging

from .excel_loader import ExcelDataLoader
from .clinical import ClinicalDataProcessor
from ..utils.memory import MemoryTracker

# Set up logging
logger = logging.getLogger(__name__)


class MultimodalDataIntegrator:
    """Integrates Excel data with MRI and clinical data streams."""

    def __init__(self, memory_limit_mb: float = 3000):
        """
        Initialize the multimodal data integrator.

        Args:
            memory_limit_mb: Memory limit in MB
        """
        self.memory_limit_mb = memory_limit_mb
        self.memory_tracker = MemoryTracker(threshold_mb=memory_limit_mb)
        
        # Initialize loaders
        self.excel_loader = ExcelDataLoader(memory_limit_mb=memory_limit_mb * 0.3)  # Allocate 30% to Excel
        self.clinical_processor = ClinicalDataProcessor()
        
        # Cache for integrated data
        self._integrated_data_cache = None

    def load_all_excel_data(self, 
                           scanner_path: Optional[Union[str, Path]] = None, 
                           clinical_path: Optional[Union[str, Path]] = None,
                           segmentation_path: Optional[Union[str, Path]] = None,
                           timepoint: int = 1,
                           force_reload: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Load all available Excel data sources.

        Args:
            scanner_path: Path to scanner metadata Excel file
            clinical_path: Path to clinical data Excel file
            segmentation_path: Path to segmentation volumes Excel file
            timepoint: Timepoint to filter for (default: 1)
            force_reload: Force reload from files even if cached

        Returns:
            Dictionary of DataFrames for each data source
        """
        self.memory_tracker.log_memory("Before loading all Excel data")
        
        data_sources = {}
        
        # Load scanner data if path provided
        if scanner_path and os.path.exists(scanner_path):
            try:
                data_sources['scanner'] = self.excel_loader.load_scanner_data(
                    scanner_path, force_reload=force_reload)
                logger.info(f"Loaded scanner data: {len(data_sources['scanner'])} rows")
            except Exception as e:
                logger.error(f"Error loading scanner data: {e}")
                data_sources['scanner'] = pd.DataFrame()
        
        # Load clinical data if path provided
        if clinical_path and os.path.exists(clinical_path):
            try:
                data_sources['clinical'] = self.excel_loader.load_clinical_data(
                    clinical_path, force_reload=force_reload)
                logger.info(f"Loaded clinical Excel data: {len(data_sources['clinical'])} rows")
            except Exception as e:
                logger.error(f"Error loading clinical Excel data: {e}")
                data_sources['clinical'] = pd.DataFrame()
        
        # Load segmentation data if path provided
        if segmentation_path and os.path.exists(segmentation_path):
            try:
                data_sources['segmentation'] = self.excel_loader.load_segmentation_volumes(
                    segmentation_path, force_reload=force_reload)
                logger.info(f"Loaded segmentation data: {len(data_sources['segmentation'])} rows")
            except Exception as e:
                logger.error(f"Error loading segmentation data: {e}")
                data_sources['segmentation'] = pd.DataFrame()
        
        self.memory_tracker.log_memory("After loading all Excel data")
        
        return data_sources

    def integrate_with_clinical_data(self, 
                                   clinical_df: pd.DataFrame,
                                   excel_data_sources: Dict[str, pd.DataFrame],
                                   timepoint: int = 1) -> pd.DataFrame:
        """
        Integrate Excel data sources with core clinical data.

        Args:
            clinical_df: Processed clinical DataFrame
            excel_data_sources: Dictionary of Excel data sources
            timepoint: Timepoint to filter for (default: 1)

        Returns:
            Integrated DataFrame with all available data
        """
        self.memory_tracker.log_memory("Before integrating with clinical data")
        
        # Get data from sources
        scanner_data = excel_data_sources.get('scanner', None)
        segmentation_data = excel_data_sources.get('segmentation', None)
        excel_clinical = excel_data_sources.get('clinical', None)
        
        # Augment clinical data with Excel clinical data if available
        if excel_clinical is not None and not excel_clinical.empty:
            # If there's Excel clinical data, merge with existing clinical data
            clinical_merged = self._merge_clinical_sources(clinical_df, excel_clinical)
        else:
            clinical_merged = clinical_df.copy()
        
        # Merge all data sources
        integrated_data = self.excel_loader.merge_data_sources(
            clinical_data=clinical_merged,
            scanner_data=scanner_data,
            segmentation_data=segmentation_data,
            timepoint=timepoint
        )
        
        # Process for graph construction and feature engineering
        processed_data = self.excel_loader.process_for_graph(integrated_data)
        
        # Cache the result
        self._integrated_data_cache = processed_data
        
        self.memory_tracker.log_memory("After integrating with clinical data")
        
        return processed_data

    def _merge_clinical_sources(self, 
                               primary_clinical: pd.DataFrame, 
                               excel_clinical: pd.DataFrame) -> pd.DataFrame:
        """
        Merge two clinical data sources with priority handling.

        Args:
            primary_clinical: Primary clinical DataFrame 
            excel_clinical: Excel clinical DataFrame

        Returns:
            Merged clinical DataFrame
        """
        # Create copies to avoid modifying inputs
        primary_df = primary_clinical.copy()
        excel_df = excel_clinical.copy()
        
        # Ensure patient_id columns are of the same type
        primary_df['patient_id'] = primary_df['patient_id'].astype(str)
        excel_df['patient_id'] = excel_df['patient_id'].astype(str)
        
        # Find overlapping columns (except patient_id)
        common_cols = [col for col in primary_df.columns.intersection(excel_df.columns)
                      if col != 'patient_id']
        
        # For overlapping columns, add suffix to Excel columns
        if common_cols:
            excel_df = excel_df.rename(columns={col: f"{col}_excel" for col in common_cols})
        
        # Merge the DataFrames
        merged_df = pd.merge(primary_df, excel_df, on='patient_id', how='outer')
        
        # For each common column, fill NaNs in primary with values from Excel
        for col in common_cols:
            excel_col = f"{col}_excel"
            # Where primary is NaN and Excel has value, use Excel value
            mask = merged_df[col].isna() & merged_df[excel_col].notna()
            merged_df.loc[mask, col] = merged_df.loc[mask, excel_col]
            # Drop the Excel column
            merged_df = merged_df.drop(columns=[excel_col])
        
        return merged_df

    def prepare_for_graph_construction(self, 
                                     integrated_data: pd.DataFrame,
                                     imaging_features: Dict[int, Dict[str, Dict[str, float]]]) -> Tuple[pd.DataFrame, Dict[int, Dict[str, Dict[str, float]]]]:
        """
        Prepare integrated data for graph construction.

        Args:
            integrated_data: Integrated DataFrame with all available data
            imaging_features: Dictionary of imaging features

        Returns:
            Tuple of (processed DataFrame, enhanced imaging features)
        """
        self.memory_tracker.log_memory("Before preparing for graph construction")
        
        # Process clinical data with ClinicalDataProcessor
        processed_df = self.clinical_processor.preprocess_clinical_data(integrated_data)
        
        # Enhance imaging features with segmentation volumes if available
        enhanced_imaging = self._enhance_imaging_features(imaging_features, integrated_data)
        
        self.memory_tracker.log_memory("After preparing for graph construction")
        
        return processed_df, enhanced_imaging

    def _enhance_imaging_features(self, 
                                imaging_features: Dict[int, Dict[str, Dict[str, float]]],
                                integrated_data: pd.DataFrame) -> Dict[int, Dict[str, Dict[str, float]]]:
        """
        Enhance imaging features with segmentation volumes.

        Args:
            imaging_features: Dictionary of imaging features
            integrated_data: Integrated DataFrame with all available data

        Returns:
            Enhanced dictionary of imaging features
        """
        # Create deep copy to avoid modifying the original
        enhanced = {
            patient_id: {
                seq: {feat: val for feat, val in seq_feats.items()}
                for seq, seq_feats in patient_feats.items()
            }
            for patient_id, patient_feats in imaging_features.items()
        }
        
        # Check if segmentation columns exist
        seg_cols = [col for col in integrated_data.columns if col.startswith('seg_')]
        if not seg_cols:
            return enhanced
        
        # Add segmentation data to imaging features
        for patient_id, patient_feats in enhanced.items():
            # Find patient in integrated data
            patient_row = integrated_data[integrated_data['patient_id'] == patient_id]
            if patient_row.empty:
                continue
            
            # Create or update 'segmentation' sequence
            if 'segmentation' not in patient_feats:
                patient_feats['segmentation'] = {}
            
            # Add segmentation features
            for col in seg_cols:
                if col in patient_row.columns and not patient_row[col].isna().all():
                    # Get the feature name (remove 'seg_' prefix)
                    feat_name = col[4:]
                    # Get the value (first row since patient_id should be unique)
                    value = patient_row[col].iloc[0]
                    # Store as float
                    patient_feats['segmentation'][feat_name] = float(value)
        
        return enhanced

    def generate_patient_summaries(self, integrated_data: pd.DataFrame) -> Dict[int, Dict[str, Any]]:
        """
        Generate comprehensive patient summaries from integrated data.

        Args:
            integrated_data: Integrated DataFrame with all available data

        Returns:
            Dictionary mapping patient IDs to summary dictionaries
        """
        summaries = {}
        
        for _, row in integrated_data.iterrows():
            patient_id = row['patient_id']
            if pd.isna(patient_id):
                continue
                
            # Convert to integer if string is numeric
            if isinstance(patient_id, str) and patient_id.isdigit():
                patient_id = int(patient_id)
                
            # Initialize patient summary
            summary = {}
            
            # Add demographic information
            demographics = {}
            for col in ['age', 'sex', 'karnofsky_score']:
                if col in row and not pd.isna(row[col]):
                    demographics[col] = row[col]
            if demographics:
                summary['demographics'] = demographics
                
            # Add tumor characteristics
            tumor = {}
            for col in ['grade', 'histology', 'location', 'idh_mutation', 'mgmt_methylation']:
                if col in row and not pd.isna(row[col]):
                    tumor[col] = row[col]
            
            # Add segmentation volumes
            volumes = {}
            for col in row.index:
                if col.startswith('seg_') and 'volume' in col.lower() and not pd.isna(row[col]):
                    vol_name = col[4:].lower()  # Remove 'seg_' prefix
                    volumes[vol_name] = float(row[col])
            if volumes:
                tumor['volumes'] = volumes
            
            if tumor:
                summary['tumor'] = tumor
                
            # Add scanner information
            scanner = {}
            for col in row.index:
                if col.startswith('scanner_') and not pd.isna(row[col]):
                    scanner_name = col[8:]  # Remove 'scanner_' prefix
                    scanner[scanner_name] = row[col]
            if scanner:
                summary['scanner'] = scanner
                
            # Add risk assessment
            risk = {}
            for col in ['risk_score', 'risk_category']:
                if col in row and not pd.isna(row[col]):
                    risk[col] = row[col]
            if risk:
                summary['risk_assessment'] = risk
                
            summaries[patient_id] = summary
            
        return summaries