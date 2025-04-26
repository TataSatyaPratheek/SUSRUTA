# susruta/src/susruta/data/excel_integration.py
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

# Adjust imports based on final structure
try:
    from .excel_loader import ExcelDataLoader
    from .clinical import ClinicalDataProcessor
    from ..utils.memory import MemoryTracker
except ImportError:
    # Fallback if running script directly might cause issues
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[2])) # Add project root
    from susruta.data.excel_loader import ExcelDataLoader
    from susruta.data.clinical import ClinicalDataProcessor
    from susruta.utils.memory import MemoryTracker


# Set up logging
logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


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

        # Initialize loaders/processors with memory allocation if needed
        # Allocate portions of the total memory limit
        excel_mem = memory_limit_mb * 0.3
        clinical_mem = memory_limit_mb * 0.3 # Placeholder, ClinicalDataProcessor doesn't take limit yet
        self.excel_loader = ExcelDataLoader(memory_limit_mb=excel_mem)
        self.clinical_processor = ClinicalDataProcessor() # Doesn't currently accept memory limit

        # Cache for integrated data
        self._integrated_data_cache: Optional[pd.DataFrame] = None
        logger.info(f"MultimodalDataIntegrator initialized with memory limit {memory_limit_mb}MB.")

    def load_all_excel_data(self,
                           scanner_path: Optional[Union[str, Path]] = None,
                           clinical_path: Optional[Union[str, Path]] = None,
                           segmentation_path: Optional[Union[str, Path]] = None,
                           # Add paths for other potential Excel files (e.g., treatments)
                           force_reload: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Load all available Excel data sources using ExcelDataLoader.

        Args:
            scanner_path: Path to scanner metadata Excel file
            clinical_path: Path to clinical data Excel file
            segmentation_path: Path to segmentation volumes Excel file
            force_reload: Force reload from files even if cached in ExcelDataLoader

        Returns:
            Dictionary of DataFrames for each data source (keys: 'scanner', 'clinical', 'segmentation')
        """
        self.memory_tracker.log_memory("Before loading all Excel data")
        logger.info("Loading all specified Excel data sources...")

        data_sources: Dict[str, pd.DataFrame] = {}

        # Load scanner data if path provided
        if scanner_path:
            scanner_path = Path(scanner_path)
            if scanner_path.exists():
                try:
                    data_sources['scanner'] = self.excel_loader.load_scanner_data(
                        scanner_path, force_reload=force_reload)
                    logger.info(f"Loaded scanner data: {data_sources['scanner'].shape}")
                except Exception as e:
                    logger.error(f"Error loading scanner data from {scanner_path}: {e}")
                    data_sources['scanner'] = pd.DataFrame() # Empty df on error
            else:
                 logger.warning(f"Scanner data path does not exist: {scanner_path}")
                 data_sources['scanner'] = pd.DataFrame()
        else:
            logger.info("No scanner data path provided.")
            data_sources['scanner'] = pd.DataFrame()


        # Load clinical data if path provided
        if clinical_path:
            clinical_path = Path(clinical_path)
            if clinical_path.exists():
                try:
                    data_sources['clinical'] = self.excel_loader.load_clinical_data(
                        clinical_path, force_reload=force_reload)
                    logger.info(f"Loaded clinical Excel data: {data_sources['clinical'].shape}")
                except Exception as e:
                    logger.error(f"Error loading clinical Excel data from {clinical_path}: {e}")
                    data_sources['clinical'] = pd.DataFrame()
            else:
                 logger.warning(f"Clinical data path does not exist: {clinical_path}")
                 data_sources['clinical'] = pd.DataFrame()
        else:
            # Clinical data is usually essential
            logger.error("No clinical data path provided. This might cause issues downstream.")
            data_sources['clinical'] = pd.DataFrame()


        # Load segmentation data if path provided
        if segmentation_path:
            segmentation_path = Path(segmentation_path)
            if segmentation_path.exists():
                try:
                    data_sources['segmentation'] = self.excel_loader.load_segmentation_volumes(
                        segmentation_path, force_reload=force_reload)
                    logger.info(f"Loaded segmentation data: {data_sources['segmentation'].shape}")
                except Exception as e:
                    logger.error(f"Error loading segmentation data from {segmentation_path}: {e}")
                    data_sources['segmentation'] = pd.DataFrame()
            else:
                 logger.warning(f"Segmentation data path does not exist: {segmentation_path}")
                 data_sources['segmentation'] = pd.DataFrame()
        else:
            logger.info("No segmentation data path provided.")
            data_sources['segmentation'] = pd.DataFrame()

        # Add loading for other Excel files here if needed

        self.memory_tracker.log_memory("After loading all Excel data")

        # Basic validation
        if data_sources.get('clinical', pd.DataFrame()).empty:
             logger.warning("Loaded clinical data is empty. Integration might be incomplete.")

        return data_sources

    def integrate_excel_sources(self,
                                   excel_data_sources: Dict[str, pd.DataFrame],
                                   timepoint: Optional[int] = None) -> pd.DataFrame:
        """
        Integrate multiple loaded Excel data sources using ExcelDataLoader's merge.
        Applies initial processing suitable for graph building.

        Args:
            excel_data_sources: Dictionary of loaded & cleaned Excel data sources
                                (output of load_all_excel_data)
            timepoint: Timepoint to filter for during merging (default: None)

        Returns:
            Integrated DataFrame with initial processing applied.
        """
        self.memory_tracker.log_memory("Before integrating Excel sources")
        logger.info(f"Integrating Excel sources. Timepoint filter: {timepoint}")

        # Get data from sources, ensuring they are DataFrames
        clinical_data = excel_data_sources.get('clinical', pd.DataFrame())
        scanner_data = excel_data_sources.get('scanner', pd.DataFrame())
        segmentation_data = excel_data_sources.get('segmentation', pd.DataFrame())

        if clinical_data.empty:
             logger.error("Cannot integrate Excel sources: Clinical data is empty.")
             # Return an empty DataFrame or raise error? Returning empty for now.
             return pd.DataFrame()

        # Merge using ExcelDataLoader's method
        integrated_data = self.excel_loader.merge_data_sources(
            clinical_data=clinical_data,
            scanner_data=scanner_data,
            segmentation_data=segmentation_data,
            timepoint=timepoint
        )
        logger.info(f"Shape after merging sources: {integrated_data.shape}")

        # Apply initial processing/feature engineering suitable for graph stage
        processed_data = self.excel_loader.process_for_graph(integrated_data)
        logger.info(f"Shape after process_for_graph: {processed_data.shape}")

        # Cache the result (optional)
        self._integrated_data_cache = processed_data

        self.memory_tracker.log_memory("After integrating Excel sources")

        return processed_data

    # --- Methods below might be more relevant for Phase 3 (Graph Building) ---
    # --- but are kept here as they were in the provided code ---

    def _merge_clinical_sources(self,
                               primary_clinical: pd.DataFrame,
                               excel_clinical: pd.DataFrame) -> pd.DataFrame:
        """
        Merge two clinical data sources with priority handling.
        (Potentially useful if clinical data comes from multiple files/sources)

        Args:
            primary_clinical: Primary clinical DataFrame
            excel_clinical: Excel clinical DataFrame

        Returns:
            Merged clinical DataFrame
        """
        logger.debug("Merging primary and Excel clinical sources.")
        # Create copies to avoid modifying inputs
        primary_df = primary_clinical.copy()
        excel_df = excel_clinical.copy()

        # Ensure patient_id columns exist and are string
        if 'patient_id' not in primary_df.columns or 'patient_id' not in excel_df.columns:
             logger.error("Both clinical sources must have 'patient_id' to merge.")
             return primary_df # Return primary if merge fails

        primary_df['patient_id'] = primary_df['patient_id'].astype(str)
        excel_df['patient_id'] = excel_df['patient_id'].astype(str)

        # Find overlapping columns (except patient_id)
        common_cols = [col for col in primary_df.columns.intersection(excel_df.columns)
                      if col != 'patient_id']
        logger.debug(f"Common columns found: {common_cols}")

        # For overlapping columns, add suffix to Excel columns before merge
        excel_df = excel_df.rename(columns={col: f"{col}_excel" for col in common_cols})

        # Merge the DataFrames using outer join to keep all patients
        merged_df = pd.merge(primary_df, excel_df, on='patient_id', how='outer')
        logger.debug(f"Shape after outer merge: {merged_df.shape}")

        # For each common column, fill NaNs in primary with values from Excel
        for col in common_cols:
            excel_col = f"{col}_excel"
            if excel_col in merged_df.columns:
                # Where primary is NaN and Excel has value, use Excel value
                mask = merged_df[col].isna() & merged_df[excel_col].notna()
                merged_df.loc[mask, col] = merged_df.loc[mask, excel_col]
                # Drop the temporary Excel column
                merged_df = merged_df.drop(columns=[excel_col])
                logger.debug(f"Filled NaNs in '{col}' using '{excel_col}'.")

        logger.debug(f"Shape after filling NaNs from Excel source: {merged_df.shape}")
        return merged_df

    def prepare_for_graph_construction(self,
                                     integrated_data: pd.DataFrame,
                                     imaging_features: Optional[Dict[Union[int, str], Dict[str, Dict[str, float]]]] = None
                                     ) -> Tuple[pd.DataFrame, Optional[Dict[Union[int, str], Dict[str, Dict[str, float]]]]]:
        """
        Prepare integrated data for graph construction using ClinicalDataProcessor
        and potentially enhance imaging features.

        Args:
            integrated_data: Integrated DataFrame (output of integrate_excel_sources)
            imaging_features: Optional dictionary of imaging features (e.g., from HDF5)

        Returns:
            Tuple of (fully processed DataFrame, enhanced imaging features)
        """
        self.memory_tracker.log_memory("Before preparing for graph construction")
        logger.info("Preparing integrated data for graph construction...")

        # Process clinical data with ClinicalDataProcessor for detailed preprocessing
        # This applies imputation, encoding, scaling etc.
        processed_df = self.clinical_processor.preprocess_clinical_data(integrated_data)
        logger.info("Applied detailed preprocessing using ClinicalDataProcessor.")

        enhanced_imaging = imaging_features
        if imaging_features is not None:
            # Enhance imaging features with segmentation volumes if available in processed_df
            enhanced_imaging = self._enhance_imaging_features(imaging_features, processed_df)
            logger.info("Enhanced imaging features with segmentation data.")

        self.memory_tracker.log_memory("After preparing for graph construction")

        return processed_df, enhanced_imaging

    def _enhance_imaging_features(self,
                                imaging_features: Dict[Union[int, str], Dict[str, Dict[str, float]]],
                                integrated_data: pd.DataFrame) -> Dict[Union[int, str], Dict[str, Dict[str, float]]]:
        """
        Enhance imaging features dict with segmentation volumes from the integrated DataFrame.

        Args:
            imaging_features: Dictionary of imaging features {patient_id: {sequence: {feature: value}}}
            integrated_data: Integrated DataFrame possibly containing 'seg_*' columns

        Returns:
            Enhanced dictionary of imaging features
        """
        logger.debug("Enhancing imaging features with segmentation volumes from DataFrame.")
        # Create deep copy to avoid modifying the original
        enhanced = {
            str(patient_id): { # Ensure keys are strings
                seq: {feat: val for feat, val in seq_feats.items()}
                for seq, seq_feats in patient_feats.items()
            }
            for patient_id, patient_feats in imaging_features.items()
        }

        # Check if segmentation columns exist in the DataFrame
        seg_cols = [col for col in integrated_data.columns if col.startswith('seg_') and ('volume' in col or 'ratio' in col)]
        if not seg_cols:
            logger.debug("No segmentation columns found in DataFrame to enhance imaging features.")
            return enhanced
        logger.debug(f"Found segmentation columns to add: {seg_cols}")

        # Ensure DataFrame patient_id is string for matching
        integrated_data_str_id = integrated_data.copy()
        integrated_data_str_id['patient_id'] = integrated_data_str_id['patient_id'].astype(str)
        # Set patient_id as index for faster lookup
        integrated_data_str_id = integrated_data_str_id.set_index('patient_id')


        # Add segmentation data to imaging features dictionary
        for patient_id_str, patient_feats in enhanced.items():
            if patient_id_str in integrated_data_str_id.index:
                # Get patient row data (can be multiple if timepoints weren't handled properly before)
                # Assuming unique patient_id or taking the first row
                patient_row = integrated_data_str_id.loc[[patient_id_str]].iloc[0] # Get Series for the patient

                # Create or update 'segmentation' sequence in the dict
                if 'segmentation' not in patient_feats:
                    patient_feats['segmentation'] = {}

                # Add segmentation features from the row
                for col in seg_cols:
                    if col in patient_row.index and pd.notna(patient_row[col]):
                        # Get the feature name (remove 'seg_' prefix)
                        feat_name = col.replace('seg_', '', 1)
                        value = patient_row[col]
                        # Store as float
                        patient_feats['segmentation'][feat_name] = float(value)
                # logger.debug(f"Added segmentation features for patient {patient_id_str}: {patient_feats.get('segmentation', {})}")

            else:
                 logger.debug(f"Patient {patient_id_str} not found in integrated DataFrame index for enhancing imaging features.")


        return enhanced

    def generate_patient_summaries(self, integrated_data: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Generate comprehensive patient summaries from integrated data.
        (Useful for inspecting the integrated data)

        Args:
            integrated_data: Integrated DataFrame with all available data

        Returns:
            Dictionary mapping patient IDs (str) to summary dictionaries
        """
        logger.info("Generating patient summaries from integrated data...")
        summaries = {}

        # Ensure patient_id is string
        df_summary = integrated_data.copy()
        df_summary['patient_id'] = df_summary['patient_id'].astype(str)

        for _, row in df_summary.iterrows():
            patient_id = row['patient_id']
            if pd.isna(patient_id):
                logger.warning("Skipping row with missing patient_id during summary generation.")
                continue

            # Initialize patient summary
            summary = {}

            # Add demographic information
            demographics = {}
            for col in ['age', 'sex', 'karnofsky_score']: # Use cleaned names
                if col in row and pd.notna(row[col]):
                    demographics[col] = row[col]
            if demographics:
                summary['demographics'] = demographics

            # Add tumor characteristics
            tumor = {}
            # Use cleaned names, check for original and mapped/numeric versions
            tumor_char_cols = ['grade', 'grade_numeric', 'histology', 'location', 'idh_mutation', 'mgmt_methylation']
            for col in tumor_char_cols:
                if col in row and pd.notna(row[col]):
                    tumor[col] = row[col]

            # Add segmentation volumes/ratios
            volumes = {}
            for col in row.index:
                if col.startswith('seg_') and ('volume' in col or 'ratio' in col) and pd.notna(row[col]):
                    vol_name = col.replace('seg_', '', 1) # Remove 'seg_' prefix
                    volumes[vol_name] = float(row[col])
            if volumes:
                tumor['volumes'] = volumes

            if tumor:
                summary['tumor'] = tumor

            # Add scanner information
            scanner = {}
            for col in row.index:
                if col.startswith('scanner_') and pd.notna(row[col]):
                    scanner_name = col.replace('scanner_', '', 1) # Remove 'scanner_' prefix
                    scanner[scanner_name] = row[col]
            if scanner:
                summary['scanner'] = scanner

            # Add risk assessment (if calculated)
            risk = {}
            for col in ['risk_score', 'risk_category']:
                if col in row and pd.notna(row[col]):
                    risk[col] = row[col]
            if risk:
                summary['risk_assessment'] = risk

            # Add timepoint if present
            if 'timepoint' in row and pd.notna(row['timepoint']):
                 summary['timepoint'] = int(row['timepoint'])


            summaries[patient_id] = summary

        logger.info(f"Generated summaries for {len(summaries)} patients.")
        return summaries

