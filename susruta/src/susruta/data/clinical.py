"""
Clinical data processing for glioma treatment prediction.

Includes preprocessing, feature engineering, and data integration 
for clinical and genomic information.
"""

from typing import Dict, List, Any, Optional, Union, Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import KNNImputer


class ClinicalDataProcessor:
    """Process and transform clinical data for the treatment prediction model."""
    
    def __init__(self):
        """Initialize the clinical data processor."""
        self.categorical_encoders: Dict[str, OneHotEncoder] = {}
        self.numerical_scalers: Dict[str, StandardScaler] = {}
        self.imputers: Dict[str, KNNImputer] = {}
    
    def preprocess_clinical_data(self, clinical_df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess clinical data including imputation and encoding.
        
        Args:
            clinical_df: DataFrame with clinical information
            
        Returns:
            Preprocessed DataFrame
        """
        # Create a copy to avoid modifying the original
        df = clinical_df.copy()
        
        # Handle missing values
        df = self._impute_missing_values(df)
        
        # Encode categorical variables
        df = self._encode_categorical_variables(df)
        
        # Scale numerical features
        df = self._scale_numerical_features(df)
        
        # Engineer additional features
        df = self._engineer_features(df)
        
        return df
    
    def _impute_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing values in the dataset.
        
        Args:
            df: DataFrame with missing values
            
        Returns:
            DataFrame with imputed values
        """
        # Identify numerical columns for KNN imputation
        numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        # Skip ID columns or timestamp columns
        numerical_cols = [col for col in numerical_cols 
                         if not col.endswith('_id') and 'date' not in col.lower()]
        
        if numerical_cols:
            # Initialize imputer if not already done
            if 'numerical' not in self.imputers:
                self.imputers['numerical'] = KNNImputer(n_neighbors=5)
                # Fit the imputer
                self.imputers['numerical'].fit(df[numerical_cols])
            
            # Apply imputation
            df.loc[:, numerical_cols] = self.imputers['numerical'].transform(df[numerical_cols])
        
        # For categorical variables, fill with mode (most frequent value)
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        for col in categorical_cols:
            mode_value = df[col].mode().iloc[0] if not df[col].mode().empty else None
            df[col].fillna(mode_value, inplace=True)
        
        return df
    
    def _encode_categorical_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical variables using one-hot encoding.
        
        Args:
            df: DataFrame with categorical variables
            
        Returns:
            DataFrame with encoded categorical variables
        """
        # Identify categorical columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Skip ID columns or date columns
        categorical_cols = [col for col in categorical_cols 
                           if not col.endswith('_id') and 'date' not in col.lower()]
        
        for col in categorical_cols:
            if col not in self.categorical_encoders:
                # Initialize and fit encoder
                self.categorical_encoders[col] = OneHotEncoder(sparse=False, handle_unknown='ignore')
                self.categorical_encoders[col].fit(df[[col]])
            
            # Transform the column
            encoded = self.categorical_encoders[col].transform(df[[col]])
            encoded_df = pd.DataFrame(
                encoded, 
                columns=[f"{col}_{cat}" for cat in self.categorical_encoders[col].categories_[0]],
                index=df.index
            )
            
            # Concat with original dataframe
            df = pd.concat([df, encoded_df], axis=1)
            
            # Remove original column
            df.drop(col, axis=1, inplace=True)
        
        return df
    
    def _scale_numerical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Scale numerical features using robust scaling.
        
        Args:
            df: DataFrame with numerical features
            
        Returns:
            DataFrame with scaled numerical features
        """
        # Identify numerical columns
        numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        # Skip ID columns or timestamp columns
        numerical_cols = [col for col in numerical_cols 
                         if not col.endswith('_id') and 'date' not in col.lower()]
        
        if numerical_cols:
            if 'numerical' not in self.numerical_scalers:
                # Initialize and fit scaler
                self.numerical_scalers['numerical'] = StandardScaler()
                self.numerical_scalers['numerical'].fit(df[numerical_cols])
            
            # Transform the columns
            df.loc[:, numerical_cols] = self.numerical_scalers['numerical'].transform(df[numerical_cols])
        
        return df
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer additional features from existing data.
        
        Args:
            df: DataFrame with preprocessed features
            
        Returns:
            DataFrame with additional engineered features
        """
        # Example: Create age groups
        if 'age' in df.columns:
            df['age_group_young'] = (df['age'] < 40).astype(int)
            df['age_group_middle'] = ((df['age'] >= 40) & (df['age'] < 60)).astype(int)
            df['age_group_elder'] = (df['age'] >= 60).astype(int)
        
        # Example: Create performance status groups
        if 'karnofsky_score' in df.columns:
            df['high_performance'] = (df['karnofsky_score'] >= 80).astype(int)
        
        return df
    
    def process_treatment_data(self, treatments_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process treatment data including temporal features.
        
        Args:
            treatments_df: DataFrame with treatment information
            
        Returns:
            Processed treatment DataFrame
        """
        # Create a copy to avoid modifying the original
        df = treatments_df.copy()
        
        # Encode treatment categories
        if 'category' in df.columns:
            if 'treatment_category' not in self.categorical_encoders:
                self.categorical_encoders['treatment_category'] = OneHotEncoder(sparse=False, handle_unknown='ignore')
                self.categorical_encoders['treatment_category'].fit(df[['category']])
            
            # Transform treatment category
            encoded = self.categorical_encoders['treatment_category'].transform(df[['category']])
            encoded_df = pd.DataFrame(
                encoded, 
                columns=[f"category_{cat}" for cat in self.categorical_encoders['treatment_category'].categories_[0]],
                index=df.index
            )
            
            # Concat with original dataframe
            df = pd.concat([df, encoded_df], axis=1)
        
        # Compute treatment duration if start and end dates are available
        if 'start_day' in df.columns and 'end_day' in df.columns:
            df['duration_days'] = df['end_day'] - df['start_day']
        
        # Compute cumulative dose if dose and duration are available
        if 'dose' in df.columns and 'duration_days' in df.columns:
            df['cumulative_dose'] = df['dose'] * df['duration_days']
        
        return df
    
    def integrate_multimodal_data(self, 
                                clinical_df: pd.DataFrame, 
                                imaging_features: Dict[int, Dict[str, Dict[str, float]]]) -> pd.DataFrame:
        """
        Integrate clinical data with imaging features.
        
        Args:
            clinical_df: Preprocessed clinical DataFrame
            imaging_features: Dictionary mapping patient IDs to imaging features
            
        Returns:
            Integrated DataFrame with clinical and imaging features
        """
        # Create a copy of clinical data
        integrated_df = clinical_df.copy()
        
        # Extract patient IDs from clinical data
        patient_ids = integrated_df['patient_id'].unique().tolist()
        
        # Initialize columns for imaging features
        common_features = None
        
        # Find common features across patients for consistent columns
        for patient_id in patient_ids:
            if patient_id in imaging_features:
                patient_features = imaging_features[patient_id]
                
                # Extract feature names from the first available sequence
                if any(patient_features.values()):
                    first_sequence = next(iter(patient_features))
                    if not common_features:
                        common_features = set(patient_features[first_sequence].keys())
                    else:
                        common_features &= set(patient_features[first_sequence].keys())
        
        if not common_features:
            # No common features found
            return integrated_df
        
        # For each patient, add imaging features as new columns
        for patient_id in patient_ids:
            if patient_id in imaging_features:
                patient_features = imaging_features[patient_id]
                
                # For each sequence and feature, create a new column
                for sequence, features in patient_features.items():
                    for feature in common_features:
                        if feature in features:
                            col_name = f"{sequence}_{feature}"
                            if col_name not in integrated_df.columns:
                                integrated_df[col_name] = None
                            
                            # Set the feature value for this patient
                            integrated_df.loc[integrated_df['patient_id'] == patient_id, col_name] = features[feature]
        
        # Impute missing imaging features
        imaging_cols = [col for col in integrated_df.columns if any(seq in col for seq in ['t1c', 't1n', 't2f', 't2w', 'tumor'])]
        
        if imaging_cols:
            if 'imaging' not in self.imputers:
                self.imputers['imaging'] = KNNImputer(n_neighbors=3)
                self.imputers['imaging'].fit(integrated_df[imaging_cols].fillna(0))
            
            integrated_df.loc[:, imaging_cols] = self.imputers['imaging'].transform(integrated_df[imaging_cols].fillna(0))
        
        return integrated_df