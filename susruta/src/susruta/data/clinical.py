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
        numerical_cols = df.select_dtypes(include=np.number).columns.tolist()

        # Skip ID columns or timestamp columns
        numerical_cols = [col for col in numerical_cols
                         if not col.endswith('_id') and 'date' not in col.lower()]

        if numerical_cols:
            # Ensure columns exist
            numerical_cols = [col for col in numerical_cols if col in df.columns]

            if numerical_cols: # Proceed only if there are numerical columns left
                # Initialize imputer if not already done
                if 'numerical' not in self.imputers:
                    self.imputers['numerical'] = KNNImputer(n_neighbors=5)
                    # Fit the imputer only on non-empty, non-all-NaN data
                    fit_df = df[numerical_cols].copy()
                    if not fit_df.empty and not fit_df.isnull().all().all():
                        # Fill NaNs robustly before fitting
                        fit_df_mean = fit_df.mean()
                        fit_df.fillna(fit_df_mean, inplace=True) # Fill with mean for fitting
                        fit_df.fillna(0, inplace=True) # Fill remaining NaNs with 0 if mean was NaN
                        self.imputers['numerical'].fit(fit_df)
                    else:
                        self.imputers.pop('numerical', None) # Remove unusable imputer safely


                if 'numerical' in self.imputers:
                    # Apply imputation
                    impute_subset = df[numerical_cols].copy()
                    # Fill NaNs before transform based on how it was fit
                    impute_subset_mean = impute_subset.mean()
                    impute_subset.fillna(impute_subset_mean, inplace=True)
                    impute_subset.fillna(0, inplace=True)
                    # Use try-except for transform as it might fail if fit failed silently
                    try:
                        imputed_values = self.imputers['numerical'].transform(impute_subset)
                        df.loc[:, numerical_cols] = pd.DataFrame(
                            imputed_values,
                            index=df.index,
                            columns=numerical_cols
                        )
                    except Exception as e:
                        print(f"Warning: Numerical imputation failed. Filling NaNs with 0. Error: {e}")
                        df[numerical_cols] = df[numerical_cols].fillna(0)

                else:
                     # If imputer couldn't be fit, fill remaining NaNs with 0 or mean
                     print("Warning: Numerical imputer not available. Filling NaNs with 0.")
                     df[numerical_cols] = df[numerical_cols].fillna(0)


        # For categorical variables, fill with mode (most frequent value)
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        for col in categorical_cols:
             # Ensure column exists
            if col not in df.columns: continue
            if df[col].isna().any():
                mode_value = df[col].mode().iloc[0] if not df[col].mode().empty else 'Unknown' # Use 'Unknown' if mode is empty
                df[col] = df[col].fillna(mode_value)

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

        df_encoded = df.copy() # Work on a copy

        for col in categorical_cols:
            if col not in df_encoded.columns: continue # Skip if column was already removed

            if col not in self.categorical_encoders:
                self.categorical_encoders[col] = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                # Ensure fit data is 2D and string
                fit_data = df_encoded[[col]].astype(str) # Convert to string to handle mixed types if any
                self.categorical_encoders[col].fit(fit_data)

            # Transform the column
            transform_data = df_encoded[[col]].astype(str)
            encoded = self.categorical_encoders[col].transform(transform_data)
            feature_names = self.categorical_encoders[col].get_feature_names_out([col])

            encoded_df = pd.DataFrame(
                encoded,
                columns=feature_names,
                index=df_encoded.index
            )

            # Concat with original dataframe and remove original column
            df_encoded = pd.concat([df_encoded.drop(col, axis=1), encoded_df], axis=1)

        return df_encoded

    def _scale_numerical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Scale numerical features using StandardScaler.

        Args:
            df: DataFrame with numerical features

        Returns:
            DataFrame with scaled numerical features
        """
        # Identify numerical columns
        numerical_cols = df.select_dtypes(include=np.number).columns.tolist()

        # --- Start Fix: Refine column selection for scaling ---
        # Exclude ID columns, date columns, and explicitly exclude known one-hot encoded columns
        # Get names of columns created by OneHotEncoder if available
        one_hot_cols = set()
        for encoder in self.categorical_encoders.values():
            if hasattr(encoder, 'get_feature_names_out'):
                try:
                    # Need to provide input feature names used during fit
                    # This is tricky as we don't store them explicitly per encoder instance
                    # Using a simpler heuristic for now: exclude columns containing '_'
                    # A more robust solution would involve storing fitted column names.
                    pass # Keep the simple heuristic for now
                except Exception:
                    pass # Ignore errors if get_feature_names_out fails

        numerical_cols_to_scale = []
        for col in numerical_cols:
            is_id = col.endswith('_id')
            is_date = 'date' in col.lower()
            # Heuristic: Assume columns with '_' are one-hot encoded (adjust if needed)
            # Let's be more specific: check if it starts with a known categorical prefix + '_'
            is_one_hot_heuristic = '_' in col and any(col.startswith(cat_col + '_') for cat_col in self.categorical_encoders.keys())

            if not is_id and not is_date and not is_one_hot_heuristic:
                numerical_cols_to_scale.append(col)
        # --- End Fix ---


        if numerical_cols_to_scale: # Use the filtered list
            # Ensure columns actually exist in the dataframe before proceeding
            numerical_cols_to_scale = [col for col in numerical_cols_to_scale if col in df.columns]
            if not numerical_cols_to_scale: # If no numerical columns left, return
                 return df

            # Convert columns to numeric, coercing errors, before fitting/transforming
            df_scaled = df.copy() # Work on a copy
            for col in numerical_cols_to_scale:
                df_scaled[col] = pd.to_numeric(df_scaled[col], errors='coerce')


            if 'numerical' not in self.numerical_scalers:
                # Initialize and fit scaler
                self.numerical_scalers['numerical'] = StandardScaler()
                # Fit only on non-empty, non-all-NaN data
                fit_df = df_scaled[numerical_cols_to_scale].copy()
                if not fit_df.empty and not fit_df.isnull().all().all():
                    # Fill NaNs robustly before fitting
                    fit_df_mean = fit_df.mean()
                    fit_df.fillna(fit_df_mean, inplace=True) # Fill with mean for fitting
                    fit_df.fillna(0, inplace=True) # Fill remaining NaNs with 0 if mean was NaN
                    self.numerical_scalers['numerical'].fit(fit_df)
                else:
                    self.numerical_scalers.pop('numerical', None) # Remove unusable scaler safely


            if 'numerical' in self.numerical_scalers:
                # Transform the columns
                transform_subset = df_scaled[numerical_cols_to_scale].copy()
                # Fill NaNs before transform based on how it was fit
                transform_subset_mean = transform_subset.mean()
                transform_subset.fillna(transform_subset_mean, inplace=True)
                transform_subset.fillna(0, inplace=True)

                # Use try-except for transform
                try:
                    # Transform using the DataFrame directly to preserve column names for sklearn
                    scaled_values = self.numerical_scalers['numerical'].transform(transform_subset)

                    # Assign back using DataFrame constructor to maintain float type and avoid warnings
                    df_scaled.loc[:, numerical_cols_to_scale] = pd.DataFrame(
                        scaled_values,
                        index=df_scaled.index,
                        columns=numerical_cols_to_scale
                    ).astype(float) # Explicitly cast to float
                except Exception as e:
                    print(f"Warning: Numerical scaling failed. Skipping scaling for {numerical_cols_to_scale}. Error: {e}")
                    # Keep original (but coerced to numeric and imputed) values if scaling fails
                    df_scaled[numerical_cols_to_scale] = transform_subset # Assign back the imputed subset

            else:
                 # If scaler couldn't be fit, maybe just return df or fillna
                 print("Warning: Numerical scaler not available. Skipping scaling.")
                 # Ensure NaNs are filled if scaling is skipped
                 df_scaled[numerical_cols_to_scale] = df_scaled[numerical_cols_to_scale].fillna(0)

            return df_scaled # Return the modified copy

        return df # Return original if no numerical columns

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer additional features from existing data.

        Args:
            df: DataFrame with preprocessed features

        Returns:
            DataFrame with additional engineered features
        """
        df_engineered = df.copy() # Work on a copy

        # Example: Create age groups
        # Check if 'age' exists and hasn't been scaled away (if scaling happens before this)
        # If scaling happened, engineered features might need to be based on original values or scaled ones
        age_col_name = 'age' # Assume original name before potential scaling prefix/suffix
        if age_col_name in df_engineered.columns:
            # Ensure age is numeric before comparison
            age_col = pd.to_numeric(df_engineered[age_col_name], errors='coerce')
            df_engineered['age_group_young'] = (age_col < 40).astype(int)
            df_engineered['age_group_middle'] = ((age_col >= 40) & (age_col < 60)).astype(int)
            df_engineered['age_group_elder'] = (age_col >= 60).astype(int)

        # Example: Create performance status groups
        kps_col_name = 'karnofsky_score' # Assume original name
        if kps_col_name in df_engineered.columns:
             # Ensure KPS is numeric
            kps_col = pd.to_numeric(df_engineered[kps_col_name], errors='coerce')
            df_engineered['high_performance'] = (kps_col >= 80).astype(int)

        return df_engineered

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
                self.categorical_encoders['treatment_category'] = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                # Ensure fit data is 2D and string
                fit_data = df[['category']].astype(str)
                self.categorical_encoders['treatment_category'].fit(fit_data)

            # Transform treatment category
            transform_data = df[['category']].astype(str)
            encoded = self.categorical_encoders['treatment_category'].transform(transform_data)
            feature_names = self.categorical_encoders['treatment_category'].get_feature_names_out(['category'])

            encoded_df = pd.DataFrame(
                encoded,
                columns=feature_names,
                index=df.index
            )

            # Concat with original dataframe
            df = pd.concat([df, encoded_df], axis=1)

        # Compute treatment duration if start and end days are available
        # Use 'duration_days' if already present, otherwise calculate if possible
        if 'duration_days' not in df.columns and 'start_day' in df.columns and 'end_day' in df.columns:
            # Ensure columns are numeric
            start_day = pd.to_numeric(df['start_day'], errors='coerce')
            end_day = pd.to_numeric(df['end_day'], errors='coerce')
            df['duration_days'] = (end_day - start_day).fillna(0) # Fill NaN duration with 0
        elif 'duration_days' in df.columns:
             # Ensure existing duration_days is numeric and fill NaNs
             df['duration_days'] = pd.to_numeric(df['duration_days'], errors='coerce').fillna(0)


        # Compute cumulative dose if dose and duration are available
        if 'dose' in df.columns and 'duration_days' in df.columns:
            # Handle null values in dose to avoid warnings
            dose_values = pd.to_numeric(df['dose'], errors='coerce').fillna(0) # Fill NaN dose with 0
            duration_values = pd.to_numeric(df['duration_days'], errors='coerce').fillna(0) # Fill NaN duration with 0
            df['cumulative_dose'] = dose_values * duration_values

        return df

    def integrate_multimodal_data(self,
                                clinical_df: pd.DataFrame,
                                imaging_features: Dict[int, Dict[str, Dict[str, float]]]) -> pd.DataFrame:
        """
        Integrate clinical data with imaging features.

        Args:
            clinical_df: Preprocessed clinical DataFrame (should have 'patient_id')
            imaging_features: Dictionary mapping patient IDs to imaging features

        Returns:
            Integrated DataFrame with clinical and imaging features
        """
        if 'patient_id' not in clinical_df.columns:
             raise ValueError("Clinical DataFrame must contain a 'patient_id' column.")

        integrated_df = clinical_df.copy()
        patient_ids_in_df = set(integrated_df['patient_id']) # Use set for faster lookup

        # --- Start Corrected Logic ---
        all_feature_columns = set()
        # First pass: Collect all possible feature column names from imaging_features
        for patient_id, sequences in imaging_features.items():
            if patient_id in patient_ids_in_df: # Only consider patients relevant to the clinical_df
                for sequence, features in sequences.items():
                    for feature_name in features.keys():
                        all_feature_columns.add(f"{sequence}_{feature_name}")

        # Initialize all potential new columns with NaN
        new_cols = sorted(list(all_feature_columns - set(integrated_df.columns))) # Only add truly new cols
        if new_cols:
             integrated_df = pd.concat([
                  integrated_df,
                  pd.DataFrame(columns=new_cols, index=integrated_df.index, dtype=float)
             ], axis=1)


        # Second pass: Populate the columns using .loc for alignment
        for patient_id, sequences in imaging_features.items():
            if patient_id in patient_ids_in_df:
                # Find the index/indices for this patient_id
                patient_indices = integrated_df.index[integrated_df['patient_id'] == patient_id]
                if not patient_indices.empty:
                    idx = patient_indices[0] # Assume unique patient_id for simplicity, take first if duplicate
                    for sequence, features in sequences.items():
                        for feature_name, feature_value in features.items():
                            col_name = f"{sequence}_{feature_name}"
                            if col_name in integrated_df.columns: # Ensure column exists
                                integrated_df.loc[idx, col_name] = feature_value


        # --- End Corrected Logic ---

        # Impute missing imaging features (using only the columns we just added/populated)
        imaging_cols_to_impute = list(all_feature_columns.intersection(integrated_df.columns))

        if imaging_cols_to_impute:
            # Ensure data is numeric before imputation
            integrated_df[imaging_cols_to_impute] = integrated_df[imaging_cols_to_impute].apply(pd.to_numeric, errors='coerce')

            if 'imaging' not in self.imputers:
                self.imputers['imaging'] = KNNImputer(n_neighbors=3)
                # Fit on non-NaN values if possible, or fillna temporarily
                fit_df = integrated_df[imaging_cols_to_impute].copy()
                # Check if DataFrame is empty or all NaN before fitting
                if not fit_df.empty and not fit_df.isnull().all().all():
                    # Fill NaNs robustly before fitting
                    fit_df_mean = fit_df.mean()
                    fit_df.fillna(fit_df_mean, inplace=True) # Fill with mean for fitting
                    fit_df.fillna(0, inplace=True) # Fill remaining NaNs with 0 if mean was NaN
                    self.imputers['imaging'].fit(fit_df)
                else:
                    self.imputers.pop('imaging', None) # Remove unusable imputer safely


            if 'imaging' in self.imputers:
                # Apply imputation
                impute_subset = integrated_df[imaging_cols_to_impute].copy()
                # Fill NaNs before transform based on how it was fit
                impute_subset_mean = impute_subset.mean()
                impute_subset.fillna(impute_subset_mean, inplace=True)
                impute_subset.fillna(0, inplace=True)

                # Use try-except for transform
                try:
                    imputed_values = self.imputers['imaging'].transform(impute_subset)
                    integrated_df.loc[:, imaging_cols_to_impute] = pd.DataFrame(
                        imputed_values,
                        index=integrated_df.index,
                        columns=imaging_cols_to_impute
                    )
                except Exception as e:
                    print(f"Warning: Imaging imputation failed. Filling NaNs with 0. Error: {e}")
                    integrated_df[imaging_cols_to_impute] = integrated_df[imaging_cols_to_impute].fillna(0)
            else:
                 # If imputer couldn't be fit, fill remaining NaNs with 0 or mean
                 print("Warning: Imaging imputer not available. Filling NaNs with 0.")
                 integrated_df[imaging_cols_to_impute] = integrated_df[imaging_cols_to_impute].fillna(0)


        return integrated_df
