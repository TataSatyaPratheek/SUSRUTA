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
from ..utils.memory import MemoryTracker

# Set up logging
logger = logging.getLogger(__name__)


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
        if self._scanner_data_cache is not None and not force_reload:
            return self._scanner_data_cache
            
        self.memory_tracker.log_memory("Before loading scanner data")
        
        try:
            # Use read_excel with specific columns if known, otherwise read all
            scanner_data = pd.read_excel(
                file_path,
                engine='openpyxl',  # More memory efficient for large files
                dtype={
                    'PatientID': str,  # Use string for IDs
                    'Timepoint': int,
                    'ScannerManufacturer': 'category',  # Use category for repeated strings
                    'ScannerModel': 'category',
                    'FieldStrength': float,
                    'SequenceType': 'category'
                }
            )
            
            self.memory_tracker.log_memory("After loading scanner data")
            
            # Clean and standardize
            scanner_data = self._clean_scanner_data(scanner_data)
            
            # Cache for reuse
            self._scanner_data_cache = scanner_data
            
            return scanner_data
            
        except Exception as e:
            logger.error(f"Error loading scanner data: {e}")
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
        
        # Standardize column names
        df.columns = [col.strip() for col in df.columns]
        
        # Ensure patient ID is string and standardize format
        if 'PatientID' in df.columns:
            df['PatientID'] = df['PatientID'].astype(str).str.strip()
        
        # Convert field strength to float safely
        if 'FieldStrength' in df.columns:
            df['FieldStrength'] = pd.to_numeric(df['FieldStrength'], errors='coerce')
        
        # Use category type for string columns to save memory
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype('category')
        
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
        if self._clinical_data_cache is not None and not force_reload:
            return self._clinical_data_cache
            
        self.memory_tracker.log_memory("Before loading clinical data")
        
        try:
            # Read in chunks if needed for very large files
            clinical_data = pd.read_excel(
                file_path,
                engine='openpyxl',
                dtype={
                    'patient_id': str,  # Use string for IDs
                    'age': float,
                    'sex': 'category',
                    'karnofsky_score': float,
                    'grade': 'category',
                    'histology': 'category',
                    'location': 'category',
                    'idh_mutation': 'category',
                    'mgmt_methylation': 'category'
                }
            )
            
            self.memory_tracker.log_memory("After loading clinical data")
            
            # Clean and standardize
            clinical_data = self._clean_clinical_data(clinical_data)
            
            # Cache for reuse
            self._clinical_data_cache = clinical_data
            
            return clinical_data
            
        except Exception as e:
            logger.error(f"Error loading clinical data: {e}")
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
        df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
        
        # Ensure patient ID is string and standardize format
        if 'patient_id' in df.columns:
            df['patient_id'] = df['patient_id'].astype(str).str.replace(r'[^0-9]', '', regex=True)
            # Convert to integer if all numeric
            if df['patient_id'].str.isnumeric().all():
                df['patient_id'] = df['patient_id'].astype(int)
        
        # Convert numeric columns to appropriate types
        numeric_cols = ['age', 'karnofsky_score']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Standardize boolean/categorical values
        binary_cols = ['idh_mutation', 'mgmt_methylation']
        for col in binary_cols:
            if col in df.columns:
                # Convert various representations (0/1, Yes/No, True/False) to 0/1
                df[col] = df[col].map(lambda x: 1 if str(x).lower() in ('1', 'yes', 'true', 'positive', '+') else 
                                                0 if str(x).lower() in ('0', 'no', 'false', 'negative', '-') else np.nan)
        
        # Use category type for string columns to save memory
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype('category')
        
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
        if self._segmentation_data_cache is not None and not force_reload:
            return self._segmentation_data_cache
            
        self.memory_tracker.log_memory("Before loading segmentation data")
        
        try:
            # Use read_excel with data types specified
            segmentation_data = pd.read_excel(
                file_path,
                engine='openpyxl',
                dtype={
                    'PatientID': str,  # Use string for IDs
                    'Timepoint': int,
                    'TumorVolume_mm3': float,
                    'EnhancingVolume_mm3': float,
                    'NecrotisCoreVolume_mm3': float,
                    'EdemaVolume_mm3': float
                }
            )
            
            self.memory_tracker.log_memory("After loading segmentation data")
            
            # Clean and standardize
            segmentation_data = self._clean_segmentation_data(segmentation_data)
            
            # Cache for reuse
            self._segmentation_data_cache = segmentation_data
            
            return segmentation_data
            
        except Exception as e:
            logger.error(f"Error loading segmentation volume data: {e}")
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
        df.columns = [col.strip() for col in df.columns]
        
        # Ensure patient ID is string and standardize format
        if 'PatientID' in df.columns:
            df['PatientID'] = df['PatientID'].astype(str).str.replace(r'[^0-9]', '', regex=True)
            # Convert to integer if all numeric
            if df['PatientID'].str.isnumeric().all():
                df['PatientID'] = df['PatientID'].astype(int)
        
        # Convert volume columns to float
        volume_cols = [col for col in df.columns if 'Volume' in col]
        for col in volume_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Calculate derived metrics if all required columns exist
        if 'TumorVolume_mm3' in df.columns and df['TumorVolume_mm3'].notna().any():
            # Replace 0 volume with NaN to avoid division by zero, then fill resulting NaN ratios with 0
            tumor_volume_safe = df['TumorVolume_mm3'].replace(0, np.nan)

            if 'EnhancingVolume_mm3' in df.columns:
                df['EnhancingRatio'] = (df['EnhancingVolume_mm3'] / tumor_volume_safe).fillna(0)
            if 'EdemaVolume_mm3' in df.columns:
                df['EdemaRatio'] = (df['EdemaVolume_mm3'] / tumor_volume_safe).fillna(0)
            if 'NecrotisCoreVolume_mm3' in df.columns:
                df['NecrosisRatio'] = (df['NecrotisCoreVolume_mm3'] / tumor_volume_safe).fillna(0)

        self.memory_tracker.log_memory("After cleaning segmentation data")
        return df

    def merge_data_sources(self, clinical_data: pd.DataFrame,
                        scanner_data: Optional[pd.DataFrame] = None,
                        segmentation_data: Optional[pd.DataFrame] = None,
                        timepoint: int = 1) -> pd.DataFrame:
        """
        Merge multiple data sources into a single patient-level DataFrame.
        ...
        """
        self.memory_tracker.log_memory("Before merging data sources")

        # Start with clinical data (patient level)
        result = clinical_data.copy()
        # --- START FIX: Ensure patient_id is string in the base DataFrame ---
        if 'patient_id' in result.columns:
            result['patient_id'] = result['patient_id'].astype(str)
        # --- END FIX ---

        # Standardize ID column names for merging
        id_mapping = {
            'patient_id': 'patient_id',
            'PatientID': 'patient_id'
        }

        # Merge scanner data if provided
        if scanner_data is not None and not scanner_data.empty:
            scanner_df = scanner_data.copy()

            # Standardize ID column
            for col, target in id_mapping.items():
                if col in scanner_df.columns:
                    scanner_df.rename(columns={col: target}, inplace=True)

            # --- START FIX: Ensure patient_id is string before merge ---
            if 'patient_id' in scanner_df.columns:
                scanner_df['patient_id'] = scanner_df['patient_id'].astype(str)
            # --- END FIX ---

            # Filter for requested timepoint
            if 'Timepoint' in scanner_df.columns:
                scanner_df = scanner_df[scanner_df['Timepoint'] == timepoint]

            # Add prefix to scanner columns (except ID and timepoint)
            prefix_cols = [col for col in scanner_df.columns
                        if col != 'patient_id' and col != 'Timepoint']
            scanner_df = scanner_df.rename(columns={col: f'scanner_{col.lower()}'
                                                for col in prefix_cols})

            # Merge with clinical data
            result = pd.merge(result, scanner_df, on='patient_id', how='left')

        # Merge segmentation data if provided
        if segmentation_data is not None and not segmentation_data.empty:
            seg_df = segmentation_data.copy()

            # Standardize ID column
            for col, target in id_mapping.items():
                if col in seg_df.columns:
                    seg_df.rename(columns={col: target}, inplace=True)

            # --- START FIX: Ensure patient_id is string before merge ---
            if 'patient_id' in seg_df.columns:
                seg_df['patient_id'] = seg_df['patient_id'].astype(str)
            # --- END FIX ---

            # Filter for requested timepoint
            if 'Timepoint' in seg_df.columns:
                seg_df = seg_df[seg_df['Timepoint'] == timepoint]

            # Add prefix to segmentation columns (except ID and timepoint)
            prefix_cols = [col for col in seg_df.columns
                        if col != 'patient_id' and col != 'Timepoint']
            seg_df = seg_df.rename(columns={col: f'seg_{col.lower()}'
                                        for col in prefix_cols})

            # Merge with result
            result = pd.merge(result, seg_df, on='patient_id', how='left')

        self.memory_tracker.log_memory("After merging data sources")

        # Force garbage collection
        gc.collect()

        return result

    def process_for_graph(self, merged_data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare merged data for graph construction with additional feature engineering.

        Args:
            merged_data: Merged DataFrame from multiple sources

        Returns:
            Processed DataFrame ready for graph construction
        """
        self.memory_tracker.log_memory("Before processing for graph")
        
        # Create a copy to avoid modifying the input
        df = merged_data.copy()
        
        # Handle missing values for key features
        # Use median imputation for numerical, mode for categorical
        num_cols = df.select_dtypes(include=['number']).columns
        for col in num_cols:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
        
        cat_cols = df.select_dtypes(include=['category', 'object']).columns
        for col in cat_cols:
            if col != 'patient_id':  # Don't fill ID column
                mode_val = df[col].mode()[0] if not df[col].mode().empty else "Unknown"
                df[col] = df[col].fillna(mode_val)
        
        # Engineer graph-specific features
        
        # Calculate age groups (useful for node grouping)
        if 'age' in df.columns:
            df['age_group'] = pd.cut(
                df['age'], 
                bins=[0, 40, 60, 100], 
                labels=['young', 'middle', 'elder']
            )
        
        # Calculate volume-based features if available
        vol_cols = [col for col in df.columns if 'volume' in col.lower()]
        if vol_cols:
            # Categorize tumors by volume
            tumor_vol_col = next((col for col in vol_cols if 'tumor' in col.lower()), None)
            if tumor_vol_col:
                df['tumor_size_category'] = pd.qcut(
                    df[tumor_vol_col].clip(lower=0), 
                    q=3, 
                    labels=['small', 'medium', 'large']
                )
        
        # Calculate combined risk score if relevant features exist
        risk_features = ['age', 'karnofsky_score', 'grade', 'idh_mutation', 'mgmt_methylation']
        if all(col in df.columns for col in risk_features):
            # Simplified risk score calculation
            df['risk_score'] = 0.0
            
            # Age factor (higher age = higher risk)
            if 'age' in df.columns:
                df['risk_score'] += (df['age'] - df['age'].min()) / (df['age'].max() - df['age'].min())
            
            # KPS factor (lower KPS = higher risk)
            if 'karnofsky_score' in df.columns:
                df['risk_score'] += 1 - (df['karnofsky_score'] - df['karnofsky_score'].min()) / \
                                   (df['karnofsky_score'].max() - df['karnofsky_score'].min())
            
            # Grade factor (higher grade = higher risk)
            if 'grade' in df.columns:
                grade_map = {'I': 0.0, 'II': 0.33, 'III': 0.67, 'IV': 1.0}
                # --- START FIX ---
                # OLD: df['risk_score'] += df['grade'].map(grade_map).fillna(0.5)
                # Convert to numeric *before* fillna
                grade_numeric = pd.to_numeric(df['grade'].map(grade_map), errors='coerce')
                df['risk_score'] += grade_numeric.fillna(0.5)
            # --- END FIX ---

            
            # IDH mutation (no mutation = higher risk)
            if 'idh_mutation' in df.columns:
                # --- START FIX ---
                idh_numeric = pd.to_numeric(df['idh_mutation'], errors='coerce').fillna(0.5) # Convert to numeric, fill NaN with neutral 0.5
                df['risk_score'] += 1 - idh_numeric
                # OLD: df['risk_score'] += 1 - df['idh_mutation']
                # --- END FIX ---
            
            # MGMT methylation (no methylation = higher risk)
            if 'mgmt_methylation' in df.columns:
            # --- START FIX ---
                mgmt_numeric = pd.to_numeric(df['mgmt_methylation'], errors='coerce').fillna(0.5) # Convert to numeric, fill NaN with neutral 0.5
                df['risk_score'] += 1 - mgmt_numeric
                # OLD: df['risk_score'] += 1 - df['mgmt_methylation']
                # --- END FIX ---

            # Normalize to 0-1 range
            # Adjust normalization factor if necessary based on number of components
            num_risk_components = 5 # age, kps, grade, idh, mgmt
            df['risk_score'] = df['risk_score'] / num_risk_components
            df['risk_score'] = df['risk_score'].clip(0, 1) # Ensure score is within [0, 1]

            # Categorize risk
            df['risk_category'] = pd.cut(
                df['risk_score'], 
                bins=[0, 0.33, 0.67, 1], 
                labels=['low', 'medium', 'high']
            )
        
        self.memory_tracker.log_memory("After processing for graph")
        
        return df