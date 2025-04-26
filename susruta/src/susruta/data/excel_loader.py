# susruta/src/susruta/data/excel_loader.py
"""
Excel data loading and processing for glioma treatment prediction.

Provides efficient loaders for scanner metadata, clinical data, and segmentation volumes
from Excel files, with memory optimization for large datasets.
"""

from typing import Dict, List, Any, Optional, Union, Tuple, Set
import os
import pandas as pd
import numpy as np
from pathlib import Path
import gc
import logging
# Ensure MemoryTracker is correctly imported from its location
# Adjust the relative path if necessary based on your final structure
try:
    from ..utils.memory import MemoryTracker
except ImportError:
    # Fallback if running script directly might cause issues
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[2])) # Add project root
    from susruta.utils.memory import MemoryTracker


# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ExcelDataLoader:
    """Memory-efficient loader for Excel-based glioma data."""

    def __init__(self, memory_limit_mb: float = 2000):
        """
        Initialize the Excel data loader.

        Args:
            memory_limit_mb: Memory limit in MB
        """
        self.memory_limit_mb = memory_limit_mb
        self.memory_tracker = MemoryTracker(threshold_mb=memory_limit_mb)

        # Initialize cache for processed data
        self._scanner_data_cache = None
        self._clinical_data_cache = None
        self._segmentation_data_cache = None

    def load_scanner_data(self, file_path: Union[str, Path], force_reload: bool = False) -> pd.DataFrame:
        """
        Load MR scanner metadata from Excel with memory efficiency.

        Args:
            file_path: Path to MR_Scanner_data.xlsx
            force_reload: Force reload from file even if cached

        Returns:
            DataFrame with scanner data
        """
        file_path = Path(file_path) # Ensure it's a Path object
        if self._scanner_data_cache is not None and not force_reload:
            logger.info("Returning cached scanner data.")
            return self._scanner_data_cache

        if not file_path.exists():
            logger.error(f"Scanner data file not found: {file_path}")
            return pd.DataFrame(columns=['PatientID', 'Timepoint', 'ScannerManufacturer',
                                        'ScannerModel', 'FieldStrength', 'SequenceType']) # Return empty with expected cols

        self.memory_tracker.log_memory("Before loading scanner data")

        try:
            # Use read_excel with specific columns if known, otherwise read all
            scanner_data = pd.read_excel(
                file_path,
                engine='openpyxl',  # More memory efficient for large files
                dtype={
                    # Specify dtypes for known columns to save memory and prevent errors
                    # Adjust these based on the actual Excel file content
                    # 'PatientID': str, # Let's handle ID cleaning later for flexibility
                    # 'Timepoint': int,
                    # 'ScannerManufacturer': 'category',
                    # 'ScannerModel': 'category',
                    # 'FieldStrength': float,
                    # 'SequenceType': 'category'
                }
            )

            self.memory_tracker.log_memory("After loading scanner data")

            # Clean and standardize
            scanner_data = self._clean_scanner_data(scanner_data)

            # Cache for reuse
            self._scanner_data_cache = scanner_data
            logger.info(f"Loaded and cleaned scanner data: {scanner_data.shape}")

            return scanner_data

        except Exception as e:
            logger.error(f"Error loading scanner data from {file_path}: {e}")
            # Return empty DataFrame with expected columns rather than None
            return pd.DataFrame(columns=['PatientID', 'Timepoint', 'ScannerManufacturer',
                                        'ScannerModel', 'FieldStrength', 'SequenceType'])

    def _clean_scanner_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize scanner data.

        Args:
            df: Raw scanner data DataFrame

        Returns:
            Cleaned scanner data DataFrame
        """
        # Create a copy to avoid modifying the input
        df = df.copy()

        # Standardize column names (lowercase, underscore)
        df.columns = [col.strip().lower().replace(' ', '_').replace('-', '_') for col in df.columns]
        logger.debug(f"Scanner columns after cleaning names: {df.columns.tolist()}")

        # Ensure patient ID is string and standardize format
        id_col = next((col for col in df.columns if 'patientid' in col), None) # Find patient ID column flexibly
        if id_col:
            df = df.rename(columns={id_col: 'patient_id'}) # Standardize name
            # Convert to string, remove non-numeric, handle potential '.0'
            df['patient_id'] = df['patient_id'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
            logger.info(f"Standardized 'patient_id' column in scanner data.")
        else:
            logger.warning("Could not find a 'PatientID' column in scanner data.")
            # Add an empty patient_id column if missing, maybe log error?
            # df['patient_id'] = None # Or raise error depending on requirement

        # Convert field strength to float safely
        fs_col = next((col for col in df.columns if 'fieldstrength' in col or 'field_strength' in col), None)
        if fs_col:
            df[fs_col] = pd.to_numeric(df[fs_col], errors='coerce')
            df = df.rename(columns={fs_col: 'field_strength'}) # Standardize name
            logger.info(f"Processed '{df.columns[df.columns.get_loc('field_strength')]}' column.")


        # Use category type for string columns to save memory
        for col in df.select_dtypes(include=['object']).columns:
            if col != 'patient_id': # Don't convert ID yet
                 # Check cardinality before converting to category
                 if df[col].nunique() / len(df) < 0.5: # Heuristic: if less than 50% unique values
                     df[col] = df[col].astype('category')
                     logger.debug(f"Converted '{col}' to category type.")

        self.memory_tracker.log_memory("After cleaning scanner data")
        return df

    def load_clinical_data(self, file_path: Union[str, Path], force_reload: bool = False) -> pd.DataFrame:
        """
        Load comprehensive clinical data from Excel with memory efficiency.

        Args:
            file_path: Path to MUGliomaPost_ClinicalDataFINAL032025.xlsx
            force_reload: Force reload from file even if cached

        Returns:
            DataFrame with clinical data
        """
        file_path = Path(file_path)
        if self._clinical_data_cache is not None and not force_reload:
            logger.info("Returning cached clinical data.")
            return self._clinical_data_cache

        if not file_path.exists():
            logger.error(f"Clinical data file not found: {file_path}")
            return pd.DataFrame(columns=['patient_id', 'age', 'sex', 'karnofsky_score',
                                        'grade', 'histology', 'location', 'idh_mutation',
                                        'mgmt_methylation']) # Return empty with expected cols

        self.memory_tracker.log_memory("Before loading clinical data")

        try:
            # Read in chunks if needed for very large files
            clinical_data = pd.read_excel(
                file_path,
                engine='openpyxl',
                # dtype={ # Specify dtypes for known columns
                #     'patient_id': str, # Let's handle ID cleaning later
                #     'age': float,
                #     'sex': 'category',
                #     'karnofsky_score': float,
                #     'grade': 'category',
                #     'histology': 'category',
                #     'location': 'category',
                #     'idh_mutation': 'category',
                #     'mgmt_methylation': 'category'
                # }
            )

            self.memory_tracker.log_memory("After loading clinical data")

            # Clean and standardize
            clinical_data = self._clean_clinical_data(clinical_data)

            # Cache for reuse
            self._clinical_data_cache = clinical_data
            logger.info(f"Loaded and cleaned clinical data: {clinical_data.shape}")

            return clinical_data

        except Exception as e:
            logger.error(f"Error loading clinical data from {file_path}: {e}")
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=['patient_id', 'age', 'sex', 'karnofsky_score',
                                        'grade', 'histology', 'location', 'idh_mutation',
                                        'mgmt_methylation'])

    def _clean_clinical_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize clinical data.

        Args:
            df: Raw clinical data DataFrame

        Returns:
            Cleaned clinical data DataFrame
        """
        # Create a copy to avoid modifying the input
        df = df.copy()

        # Standardize column names to snake_case
        df.columns = [col.strip().lower().replace(' ', '_').replace('-', '_').replace('.', '_') for col in df.columns]
        logger.debug(f"Clinical columns after cleaning names: {df.columns.tolist()}")


        # Ensure patient ID is string and standardize format
        id_col = next((col for col in df.columns if 'patient_id' in col), None)
        if id_col:
            df = df.rename(columns={id_col: 'patient_id'}) # Standardize name
            # Convert to string, remove non-numeric, handle potential '.0'
            df['patient_id'] = df['patient_id'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
            logger.info(f"Standardized 'patient_id' column in clinical data.")
        else:
            logger.error("CRITICAL: 'patient_id' column not found in clinical data. Cannot proceed with merging.")
            # Consider raising an error here
            raise ValueError("Clinical data MUST contain a 'patient_id' column.")

        # Convert numeric columns to appropriate types
        numeric_cols = ['age', 'karnofsky_score'] # Add other known numeric cols
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                logger.debug(f"Converted '{col}' to numeric.")

        # Standardize boolean/categorical values (example for IDH/MGMT)
        binary_cols = ['idh_mutation', 'mgmt_methylation']
        for col in binary_cols:
            if col in df.columns:
                # Convert various representations (0/1, Yes/No, True/False, Positive/Negative, +/-) to 0/1
                df[col] = df[col].astype(str).str.lower() # Ensure string and lowercase
                df[col] = df[col].map(lambda x: 1.0 if x in ('1', 'yes', 'true', 'positive', '+', 'mutated', 'methylated') else
                                                0.0 if x in ('0', 'no', 'false', 'negative', '-', 'wild_type', 'unmethylated', 'wt') else np.nan)
                logger.debug(f"Standardized binary column '{col}' to 0/1/NaN.")


        # Use category type for string columns with low cardinality to save memory
        for col in df.select_dtypes(include=['object']).columns:
             if col != 'patient_id':
                 if df[col].nunique() / len(df) < 0.5:
                     df[col] = df[col].astype('category')
                     logger.debug(f"Converted '{col}' to category type.")

        self.memory_tracker.log_memory("After cleaning clinical data")
        return df

    def load_segmentation_volumes(self, file_path: Union[str, Path], force_reload: bool = False) -> pd.DataFrame:
        """
        Load segmentation volume data from Excel with memory efficiency.

        Args:
            file_path: Path to MUGliomaPost_Segmentation_Volumes.xlsx
            force_reload: Force reload from file even if cached

        Returns:
            DataFrame with segmentation volume data
        """
        file_path = Path(file_path)
        if self._segmentation_data_cache is not None and not force_reload:
            logger.info("Returning cached segmentation data.")
            return self._segmentation_data_cache

        if not file_path.exists():
            logger.error(f"Segmentation data file not found: {file_path}")
            return pd.DataFrame(columns=['PatientID', 'Timepoint', 'TumorVolume_mm3',
                                        'EnhancingVolume_mm3', 'NecrotisCoreVolume_mm3',
                                        'EdemaVolume_mm3']) # Return empty with expected cols

        self.memory_tracker.log_memory("Before loading segmentation data")

        try:
            # Use read_excel with data types specified
            segmentation_data = pd.read_excel(
                file_path,
                engine='openpyxl',
                # dtype={ # Specify dtypes for known columns
                #     'PatientID': str, # Clean later
                #     'Timepoint': int,
                #     'TumorVolume_mm3': float,
                #     'EnhancingVolume_mm3': float,
                #     'NecrotisCoreVolume_mm3': float,
                #     'EdemaVolume_mm3': float
                # }
            )

            self.memory_tracker.log_memory("After loading segmentation data")

            # Clean and standardize
            segmentation_data = self._clean_segmentation_data(segmentation_data)

            # Cache for reuse
            self._segmentation_data_cache = segmentation_data
            logger.info(f"Loaded and cleaned segmentation data: {segmentation_data.shape}")

            return segmentation_data

        except Exception as e:
            logger.error(f"Error loading segmentation volume data from {file_path}: {e}")
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=['PatientID', 'Timepoint', 'TumorVolume_mm3',
                                        'EnhancingVolume_mm3', 'NecrotisCoreVolume_mm3',
                                        'EdemaVolume_mm3'])

    def _clean_segmentation_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize segmentation volume data.

        Args:
            df: Raw segmentation data DataFrame

        Returns:
            Cleaned segmentation data DataFrame
        """
        # Create a copy to avoid modifying the input
        df = df.copy()

        # Standardize column names
        df.columns = [col.strip().lower().replace(' ', '_').replace('-', '_') for col in df.columns]
        logger.debug(f"Segmentation columns after cleaning names: {df.columns.tolist()}")


        # Ensure patient ID is string and standardize format
        id_col = next((col for col in df.columns if 'patientid' in col), None)
        if id_col:
            df = df.rename(columns={id_col: 'patient_id'}) # Standardize name
            # Convert to string, remove non-numeric, handle potential '.0'
            df['patient_id'] = df['patient_id'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
            logger.info(f"Standardized 'patient_id' column in segmentation data.")
        else:
            logger.warning("Could not find a 'PatientID' column in segmentation data.")
            # df['patient_id'] = None # Or raise error

        # Convert volume columns to float
        volume_cols = [col for col in df.columns if 'volume' in col]
        for col in volume_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            logger.debug(f"Converted '{col}' to numeric.")

        # Calculate derived metrics if all required columns exist
        # Standardize volume column names before calculation
        df.columns = [col.replace('tumorvolume_mm3', 'tumor_volume_mm3')
                         .replace('enhancingvolume_mm3', 'enhancing_volume_mm3')
                         .replace('necrotiscorevolume_mm3', 'necrosis_core_volume_mm3') # Correct typo
                         .replace('edemavolume_mm3', 'edema_volume_mm3')
                      for col in df.columns]

        required_vol_cols = ['tumor_volume_mm3', 'enhancing_volume_mm3', 'edema_volume_mm3', 'necrosis_core_volume_mm3']
        if all(col in df.columns for col in required_vol_cols):
            # Replace 0 volume with NaN to avoid division by zero, then fill resulting NaN ratios with 0
            tumor_volume_safe = df['tumor_volume_mm3'].replace(0, np.nan)

            df['enhancing_ratio'] = (df['enhancing_volume_mm3'] / tumor_volume_safe).fillna(0)
            df['edema_ratio'] = (df['edema_volume_mm3'] / tumor_volume_safe).fillna(0)
            df['necrosis_ratio'] = (df['necrosis_core_volume_mm3'] / tumor_volume_safe).fillna(0)
            logger.info("Calculated volume ratios (enhancing, edema, necrosis).")
        else:
            missing_cols = [col for col in required_vol_cols if col not in df.columns]
            if missing_cols:
                logger.warning(f"Cannot calculate volume ratios. Missing columns: {missing_cols}")


        self.memory_tracker.log_memory("After cleaning segmentation data")
        return df

    def merge_data_sources(self, clinical_data: pd.DataFrame,
                        scanner_data: Optional[pd.DataFrame] = None,
                        segmentation_data: Optional[pd.DataFrame] = None,
                        timepoint: Optional[int] = None) -> pd.DataFrame:
        """
        Merge multiple data sources into a single DataFrame.
        If timepoint is None, attempts to merge using ['patient_id', 'timepoint']
        if 'timepoint' exists in both dataframes being merged, otherwise uses ['patient_id'].

        Args:
            clinical_data: Cleaned clinical DataFrame (must have 'patient_id')
            scanner_data: Cleaned scanner DataFrame (optional)
            segmentation_data: Cleaned segmentation DataFrame (optional)
            timepoint: Specific timepoint to filter scanner/segmentation data for (optional).
                       If None, merges all available timepoints.

        Returns:
            Merged DataFrame
        """
        self.memory_tracker.log_memory("Before merging data sources")

        if 'patient_id' not in clinical_data.columns:
             logger.error("Clinical data MUST contain 'patient_id' for merging.")
             raise ValueError("Clinical data missing 'patient_id' column.")

        # Start with clinical data
        result = clinical_data.copy()
        result['patient_id'] = result['patient_id'].astype(str)
        logger.debug(f"Base clinical data shape: {result.shape}")

        # --- Define potential merge keys ---
        base_merge_key = ['patient_id']
        timepoint_merge_key = ['patient_id', 'timepoint']

        # --- Merge scanner data ---
        if scanner_data is not None and not scanner_data.empty:
            scanner_df = scanner_data.copy()
            if 'patient_id' not in scanner_df.columns:
                logger.warning("Scanner data missing 'patient_id', cannot merge.")
            else:
                scanner_df['patient_id'] = scanner_df['patient_id'].astype(str)
                df_to_merge = scanner_df
                merge_key = base_merge_key # Default merge key

                if timepoint is not None:
                    # --- Filter for specific timepoint (original logic) ---
                    if 'timepoint' in scanner_df.columns:
                        scanner_df['timepoint'] = pd.to_numeric(scanner_df['timepoint'], errors='coerce')
                        df_to_merge = scanner_df[scanner_df['timepoint'] == timepoint].copy()
                        # Use timepoint merge key if result also has timepoint
                        if 'timepoint' in result.columns:
                            merge_key = timepoint_merge_key
                        logger.info(f"Filtering scanner data for timepoint {timepoint}. Found {df_to_merge.shape[0]} rows.")
                    else:
                        logger.warning("Timepoint filter requested, but 'timepoint' column missing in scanner data.")
                else:
                    # --- Merge ALL timepoints (NEW logic) ---
                    logger.info("No specific timepoint filter. Merging all available scanner data.")
                    # Check if timepoint exists in both clinical and scanner data
                    if 'timepoint' in result.columns and 'timepoint' in scanner_df.columns:
                        merge_key = timepoint_merge_key
                        logger.info("Merging scanner data using keys: ['patient_id', 'timepoint']")
                    else:
                        merge_key = base_merge_key
                        logger.info("Merging scanner data using key: ['patient_id']")
                        if 'timepoint' in result.columns and 'timepoint' not in scanner_df.columns:
                             logger.warning("Clinical data has 'timepoint', but scanner data does not. Merging on 'patient_id' only.")
                        elif 'timepoint' not in result.columns and 'timepoint' in scanner_df.columns:
                             logger.warning("Scanner data has 'timepoint', but clinical data does not. Merging on 'patient_id' only.")


                if not df_to_merge.empty:
                    # Add prefix to scanner columns (except merge keys)
                    prefix_cols = [col for col in df_to_merge.columns if col not in merge_key]
                    df_to_merge = df_to_merge.rename(columns={col: f'scanner_{col}' for col in prefix_cols})

                    # Perform the merge
                    result = pd.merge(result, df_to_merge, on=merge_key, how='left')
                    logger.info(f"Merged scanner data. Shape after merge: {result.shape}")
                else:
                    logger.info("No scanner data rows found to merge (potentially due to timepoint filter).")

        # --- Merge segmentation data ---
        if segmentation_data is not None and not segmentation_data.empty:
            seg_df = segmentation_data.copy()
            if 'patient_id' not in seg_df.columns:
                logger.warning("Segmentation data missing 'patient_id', cannot merge.")
            else:
                seg_df['patient_id'] = seg_df['patient_id'].astype(str)
                df_to_merge = seg_df
                merge_key = base_merge_key # Default merge key

                if timepoint is not None:
                    # --- Filter for specific timepoint (original logic) ---
                    if 'timepoint' in seg_df.columns:
                        seg_df['timepoint'] = pd.to_numeric(seg_df['timepoint'], errors='coerce')
                        df_to_merge = seg_df[seg_df['timepoint'] == timepoint].copy()
                        # Use timepoint merge key if result also has timepoint
                        if 'timepoint' in result.columns:
                            merge_key = timepoint_merge_key
                        logger.info(f"Filtering segmentation data for timepoint {timepoint}. Found {df_to_merge.shape[0]} rows.")
                    else:
                        logger.warning("Timepoint filter requested, but 'timepoint' column missing in segmentation data.")
                else:
                    # --- Merge ALL timepoints (NEW logic) ---
                    logger.info("No specific timepoint filter. Merging all available segmentation data.")
                    # Check if timepoint exists in both result and segmentation data
                    if 'timepoint' in result.columns and 'timepoint' in seg_df.columns:
                        merge_key = timepoint_merge_key
                        logger.info("Merging segmentation data using keys: ['patient_id', 'timepoint']")
                    else:
                        merge_key = base_merge_key
                        logger.info("Merging segmentation data using key: ['patient_id']")
                        if 'timepoint' in result.columns and 'timepoint' not in seg_df.columns:
                             logger.warning("Result data has 'timepoint', but segmentation data does not. Merging on 'patient_id' only.")
                        elif 'timepoint' not in result.columns and 'timepoint' in seg_df.columns:
                             logger.warning("Segmentation data has 'timepoint', but result data does not. Merging on 'patient_id' only.")


                if not df_to_merge.empty:
                    # Add prefix to segmentation columns (except merge keys)
                    prefix_cols = [col for col in df_to_merge.columns if col not in merge_key]
                    df_to_merge = df_to_merge.rename(columns={col: f'seg_{col}' for col in prefix_cols})

                    # Perform the merge
                    result = pd.merge(result, df_to_merge, on=merge_key, how='left')
                    logger.info(f"Merged segmentation data. Shape after merge: {result.shape}")
                else:
                    logger.info("No segmentation data rows found to merge (potentially due to timepoint filter).")

        self.memory_tracker.log_memory("After merging data sources")
        gc.collect()
        return result

    def process_for_graph(self, merged_data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare merged data for graph construction with additional feature engineering.
        (This applies some basic imputation and feature engineering suitable before detailed preprocessing)

        Args:
            merged_data: Merged DataFrame from multiple sources

        Returns:
            Processed DataFrame ready for graph construction
        """
        self.memory_tracker.log_memory("Before processing for graph")

        # Create a copy to avoid modifying the input
        df = merged_data.copy()

        # --- Basic Imputation (before detailed ClinicalDataProcessor) ---
        # Use median imputation for numerical, mode for categorical/object
        # This is a simpler imputation than KNN used later, suitable for this stage
        logger.info("Applying basic imputation (median/mode) in process_for_graph...")
        num_cols = df.select_dtypes(include=np.number).columns
        for col in num_cols:
            if df[col].isnull().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                logger.debug(f"Filled NaNs in '{col}' with median ({median_val})")

        cat_cols = df.select_dtypes(include=['category', 'object']).columns
        for col in cat_cols:
            # Don't fill ID column or columns that might be all NaN intentionally
            if col != 'patient_id' and df[col].isnull().any() and not df[col].isnull().all():
                mode_val = df[col].mode()[0] if not df[col].mode().empty else "Unknown"
                df[col] = df[col].fillna(mode_val)
                logger.debug(f"Filled NaNs in '{col}' with mode ('{mode_val}')")

        # --- Feature Engineering Examples (can be expanded) ---
        logger.info("Applying feature engineering in process_for_graph...")

        # Calculate age groups (useful for node grouping)
        if 'age' in df.columns:
            try:
                df['age_group'] = pd.cut(
                    df['age'],
                    bins=[0, 40, 60, np.inf], # Use np.inf for upper bound
                    labels=['young', 'middle', 'elder'],
                    right=False # [0, 40), [40, 60), [60, inf)
                ).astype(str) # Ensure string type
                logger.info("Created 'age_group' feature.")
            except Exception as e:
                logger.warning(f"Could not create age groups: {e}")


        # Calculate volume-based features if available
        vol_cols = [col for col in df.columns if 'volume' in col.lower()]
        if vol_cols:
            # Categorize tumors by volume
            tumor_vol_col = next((col for col in vol_cols if 'tumor_volume' in col.lower()), None)
            if tumor_vol_col:
                try:
                    # Ensure volume is numeric and non-negative before qcut
                    numeric_vol = pd.to_numeric(df[tumor_vol_col], errors='coerce').fillna(0).clip(lower=0)
                    if numeric_vol.nunique() > 3: # Need enough unique values for 3 quantiles
                        df['tumor_size_category'] = pd.qcut(
                            numeric_vol,
                            q=3,
                            labels=['small', 'medium', 'large'],
                            duplicates='drop'
                        ).astype(str) # Ensure string type
                        logger.info("Created 'tumor_size_category' feature.")
                    else:
                        logger.warning(f"Not enough unique values in '{tumor_vol_col}' to create 3 size categories.")
                        df['tumor_size_category'] = 'unknown' # Assign default
                except ValueError as e:
                    logger.warning(f"Could not create tumor size category due to qcut error: {e}. Assigning 'unknown'.")
                    df['tumor_size_category'] = 'unknown'
                except Exception as e:
                     logger.warning(f"An unexpected error occurred creating tumor size category: {e}")
                     df['tumor_size_category'] = 'unknown'


        # Calculate combined risk score if relevant features exist
        # Note: Ensure features used here are cleaned/imputed first
        risk_features = ['age', 'karnofsky_score', 'grade', 'idh_mutation', 'mgmt_methylation']
        # Check if columns exist and are numeric (or can be mapped)
        required_present = all(col in df.columns for col in ['age', 'karnofsky_score', 'idh_mutation', 'mgmt_methylation'])
        grade_present = 'grade' in df.columns or 'grade_numeric' in df.columns

        if required_present and grade_present:
            logger.info("Calculating simplified risk score...")
            df['risk_score'] = 0.0
            num_risk_components = 0

            # Age factor (higher age = higher risk) - Normalize 0-1
            if 'age' in df.columns and df['age'].notna().any():
                age_min, age_max = df['age'].min(), df['age'].max()
                if age_max > age_min:
                    df['risk_score'] += (df['age'] - age_min) / (age_max - age_min)
                num_risk_components += 1

            # KPS factor (lower KPS = higher risk) - Normalize 0-1
            if 'karnofsky_score' in df.columns and df['karnofsky_score'].notna().any():
                kps_min, kps_max = df['karnofsky_score'].min(), df['karnofsky_score'].max()
                if kps_max > kps_min:
                    df['risk_score'] += 1.0 - (df['karnofsky_score'] - kps_min) / (kps_max - kps_min)
                num_risk_components += 1

            # Grade factor (higher grade = higher risk) - Map to 0-1
            grade_col_to_use = None
            if 'grade_numeric' in df.columns and df['grade_numeric'].notna().any():
                 grade_col_to_use = 'grade_numeric'
            elif 'grade' in df.columns and df['grade'].notna().any():
                 # Try mapping from original grade column if numeric doesn't exist
                 grade_map = {'I': 1.0, 'II': 2.0, 'III': 3.0, 'IV': 4.0, # Roman
                              '1': 1.0, '2': 2.0, '3': 3.0, '4': 4.0} # String numbers
                 df['grade_numeric_temp'] = pd.to_numeric(df['grade'].astype(str).str.upper().map(grade_map), errors='coerce')
                 if df['grade_numeric_temp'].notna().any():
                     grade_col_to_use = 'grade_numeric_temp'

            if grade_col_to_use:
                 grade_min, grade_max = df[grade_col_to_use].min(), df[grade_col_to_use].max()
                 if grade_max > grade_min:
                     df['risk_score'] += (df[grade_col_to_use] - grade_min) / (grade_max - grade_min)
                 elif grade_max == grade_min and grade_min is not None: # Handle case where all grades are the same
                     df['risk_score'] += 0.5 # Assign neutral score
                 num_risk_components += 1
                 if 'grade_numeric_temp' in df.columns: df = df.drop(columns=['grade_numeric_temp']) # Clean up temp column


            # IDH mutation (no mutation (0) = higher risk)
            if 'idh_mutation' in df.columns and df['idh_mutation'].notna().any():
                # Assumes idh_mutation is 0 or 1 (or NaN handled by imputation)
                df['risk_score'] += 1.0 - df['idh_mutation']
                num_risk_components += 1

            # MGMT methylation (no methylation (0) = higher risk)
            if 'mgmt_methylation' in df.columns and df['mgmt_methylation'].notna().any():
                # Assumes mgmt_methylation is 0 or 1 (or NaN handled by imputation)
                df['risk_score'] += 1.0 - df['mgmt_methylation']
                num_risk_components += 1

            # Normalize final score
            if num_risk_components > 0:
                df['risk_score'] = (df['risk_score'] / num_risk_components).clip(0, 1) # Ensure score is within [0, 1]
                logger.info(f"Calculated 'risk_score' based on {num_risk_components} components.")

                # Categorize risk
                try:
                    df['risk_category'] = pd.cut(
                        df['risk_score'],
                        bins=[-0.01, 0.33, 0.67, 1.01], # Adjust bins slightly for inclusivity
                        labels=['low', 'medium', 'high']
                    ).astype(str) # Ensure string type
                    logger.info("Created 'risk_category' feature.")
                except Exception as e:
                    logger.warning(f"Could not create risk category: {e}")
            else:
                 logger.warning("Could not calculate risk score, not enough valid components.")


        self.memory_tracker.log_memory("After processing for graph")

        return df
