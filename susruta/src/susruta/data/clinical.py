# susruta/src/susruta/data/clinical.py
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
import logging

# Set up logging
logger = logging.getLogger(__name__)
# Configure logging if running this file directly or if not configured upstream
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ClinicalDataProcessor:
    """Process and transform clinical data for the treatment prediction model."""

    def __init__(self):
        """Initialize the clinical data processor."""
        self.categorical_encoders: Dict[str, OneHotEncoder] = {}
        self.numerical_scalers: Dict[str, StandardScaler] = {}
        self.imputers: Dict[str, KNNImputer] = {}
        self.fitted_num_cols_impute: List[str] = []
        self.fitted_num_cols_scale: List[str] = []
        self.fitted_cat_cols_encode: List[str] = []
        self.fitted_imaging_cols_impute: List[str] = []
        logger.info("ClinicalDataProcessor initialized.")

    def preprocess_clinical_data(self, clinical_df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess clinical data including imputation, encoding, and scaling.

        Args:
            clinical_df: DataFrame with clinical information (potentially merged)

        Returns:
            Preprocessed DataFrame
        """
        logger.info(f"Starting preprocessing clinical data. Initial shape: {clinical_df.shape}")
        # Create a copy to avoid modifying the original
        df = clinical_df.copy()

        # --- Step 1: Impute Missing Values ---
        logger.info("Step 1: Imputing missing values...")
        df = self._impute_missing_values(df)
        logger.info(f"Shape after imputation: {df.shape}")

        # --- Step 2: Encode Categorical Variables ---
        logger.info("Step 2: Encoding categorical variables...")
        df = self._encode_categorical_variables(df)
        logger.info(f"Shape after encoding: {df.shape}")

        # --- Step 3: Scale Numerical Features ---
        logger.info("Step 3: Scaling numerical features...")
        df = self._scale_numerical_features(df)
        logger.info(f"Shape after scaling: {df.shape}")

        # --- Step 4: Engineer Additional Features ---
        # Note: Feature engineering might be better before scaling depending on the feature
        logger.info("Step 4: Engineering features...")
        df = self._engineer_features(df)
        logger.info(f"Shape after feature engineering: {df.shape}")

        logger.info("Finished preprocessing clinical data.")
        return df

    def _impute_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing values using KNN for numerical and mode for categorical.

        Args:
            df: DataFrame with missing values

        Returns:
            DataFrame with imputed values
        """
        # Identify numerical columns for KNN imputation
        numerical_cols = df.select_dtypes(include=np.number).columns.tolist()

        # Exclude ID columns or known non-imputable numeric columns
        numerical_cols = [col for col in numerical_cols
                         if not col.endswith('_id') and 'timepoint' not in col.lower()]
        logger.debug(f"Numerical columns identified for potential imputation: {numerical_cols}")

        if numerical_cols:
            # Ensure columns exist
            numerical_cols = [col for col in numerical_cols if col in df.columns]

            if numerical_cols: # Proceed only if there are numerical columns left
                imputer_key = 'numerical_clinical'
                # Initialize imputer if not already done (FIT)
                if imputer_key not in self.imputers:
                    logger.info(f"Fitting KNNImputer ({imputer_key}) for numerical columns.")
                    self.imputers[imputer_key] = KNNImputer(n_neighbors=5)
                    # Fit only on non-empty, non-all-NaN data
                    fit_df = df[numerical_cols].copy()
                    # Check for columns with zero variance - KNNImputer might handle this, but good practice
                    zero_var_cols = fit_df.columns[fit_df.var() == 0]
                    if not zero_var_cols.empty:
                        logger.warning(f"Columns with zero variance found, may affect KNN imputation: {zero_var_cols.tolist()}")
                        # Optionally remove them from fit_df or handle differently
                        # fit_df = fit_df.drop(columns=zero_var_cols)
                        # numerical_cols = fit_df.columns.tolist() # Update list if cols dropped

                    if not fit_df.empty and not fit_df.isnull().all().all():
                        try:
                            # KNNImputer handles NaNs during fit, no need to pre-fill
                            self.imputers[imputer_key].fit(fit_df)
                            self.fitted_num_cols_impute = numerical_cols # Store fitted columns
                            logger.info(f"KNNImputer ({imputer_key}) fitted on columns: {self.fitted_num_cols_impute}")
                        except ValueError as e:
                             logger.error(f"KNNImputer ({imputer_key}) fitting failed: {e}. Removing imputer.")
                             self.imputers.pop(imputer_key, None)
                             self.fitted_num_cols_impute = []
                    else:
                        logger.warning(f"Cannot fit KNNImputer ({imputer_key}): Data is empty or all NaN.")
                        self.imputers.pop(imputer_key, None) # Remove unusable imputer safely
                        self.fitted_num_cols_impute = []


                # Apply imputation (TRANSFORM)
                if imputer_key in self.imputers:
                    # Ensure transform uses the same columns as fit
                    cols_to_transform = [col for col in self.fitted_num_cols_impute if col in df.columns]
                    if cols_to_transform:
                        logger.info(f"Applying KNNImputer ({imputer_key}) transform on columns: {cols_to_transform}")
                        try:
                            imputed_values = self.imputers[imputer_key].transform(df[cols_to_transform])
                            # Create DataFrame with correct index and columns
                            imputed_df = pd.DataFrame(imputed_values, index=df.index, columns=cols_to_transform)
                            # Update original DataFrame only for the transformed columns
                            df.update(imputed_df)
                            logger.info(f"KNN imputation applied successfully to {len(cols_to_transform)} columns.")
                        except Exception as e:
                            logger.error(f"KNNImputer ({imputer_key}) transform failed: {e}. Filling remaining NaNs with 0.")
                            df[cols_to_transform] = df[cols_to_transform].fillna(0)
                    else:
                         logger.warning(f"No columns found to apply KNNImputer ({imputer_key}) transform.")

                else:
                     # If imputer couldn't be fit, fill remaining NaNs with 0 or median
                     logger.warning(f"Numerical imputer ({imputer_key}) not available. Filling NaNs with 0 for columns: {numerical_cols}")
                     df[numerical_cols] = df[numerical_cols].fillna(0)


        # For categorical variables, fill with mode (most frequent value) or 'Unknown'
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        logger.debug(f"Categorical columns identified for mode imputation: {categorical_cols}")

        for col in categorical_cols:
             # Ensure column exists and has missing values
            if col not in df.columns or not df[col].isna().any(): continue

            # Calculate mode, handle empty mode case
            mode_series = df[col].mode()
            if not mode_series.empty:
                mode_value = mode_series.iloc[0]
            else:
                mode_value = 'Unknown' # Use 'Unknown' if mode is empty or column is all NaN

            df[col] = df[col].fillna(mode_value)
            logger.debug(f"Imputed missing values in categorical column '{col}' with '{mode_value}'.")

        return df

    def _encode_categorical_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical variables using one-hot encoding.

        Args:
            df: DataFrame with categorical variables

        Returns:
            DataFrame with encoded categorical variables
        """
        # Identify categorical columns (object or category type)
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        # Exclude ID columns
        categorical_cols = [col for col in categorical_cols if not col.endswith('_id')]
        logger.debug(f"Categorical columns identified for encoding: {categorical_cols}")


        df_encoded = df.copy() # Work on a copy
        newly_fitted_cols = []

        for col in categorical_cols:
            if col not in df_encoded.columns: continue # Skip if column was somehow removed

            # Check if column has more than 1 unique value (excluding NaN, which should be filled)
            if df_encoded[col].nunique() <= 1:
                 logger.warning(f"Skipping encoding for categorical column '{col}' as it has <= 1 unique value.")
                 continue

            # Initialize encoder if not already done (FIT)
            if col not in self.categorical_encoders:
                logger.info(f"Fitting OneHotEncoder for column '{col}'.")
                # handle_unknown='ignore' is crucial for applying to new data later
                # sparse_output=False makes it easier to work with DataFrames
                self.categorical_encoders[col] = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                try:
                    # Ensure fit data is 2D and string type to handle potential mixed types
                    fit_data = df_encoded[[col]].astype(str)
                    self.categorical_encoders[col].fit(fit_data)
                    newly_fitted_cols.append(col)
                    logger.info(f"OneHotEncoder fitted for '{col}'.")
                except Exception as e:
                     logger.error(f"OneHotEncoder fitting failed for '{col}': {e}. Removing encoder.")
                     self.categorical_encoders.pop(col, None)
                     continue # Skip transformation if fit failed

            # Apply encoding (TRANSFORM)
            if col in self.categorical_encoders:
                logger.debug(f"Applying OneHotEncoder transform for column '{col}'.")
                try:
                    # Ensure transform data is 2D and string
                    transform_data = df_encoded[[col]].astype(str)
                    encoded_array = self.categorical_encoders[col].transform(transform_data)

                    # Get feature names for the new columns
                    # Use get_feature_names_out for modern scikit-learn versions
                    if hasattr(self.categorical_encoders[col], 'get_feature_names_out'):
                        feature_names = self.categorical_encoders[col].get_feature_names_out([col])
                    else: # Fallback for older versions
                        feature_names = [f"{col}_{cat}" for cat in self.categorical_encoders[col].categories_[0]]

                    # Create a DataFrame with the encoded features
                    encoded_df = pd.DataFrame(
                        encoded_array,
                        columns=feature_names,
                        index=df_encoded.index
                    ).astype(int) # One-hot features should be integers (0 or 1)

                    # Concatenate with original dataframe and remove original column
                    df_encoded = pd.concat([df_encoded.drop(col, axis=1), encoded_df], axis=1)
                    logger.debug(f"Encoded '{col}' into columns: {feature_names}")

                except Exception as e:
                    logger.error(f"OneHotEncoder transform failed for '{col}': {e}. Original column kept.")

        # Store the list of columns that were successfully encoded in this run
        self.fitted_cat_cols_encode = newly_fitted_cols
        logger.info(f"Columns encoded in this run: {self.fitted_cat_cols_encode}")

        return df_encoded

    def _scale_numerical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Scale numerical features using StandardScaler. Excludes binary/one-hot encoded features.

        Args:
            df: DataFrame with numerical features (including newly encoded ones)

        Returns:
            DataFrame with scaled numerical features
        """
        # Identify numerical columns
        numerical_cols = df.select_dtypes(include=np.number).columns.tolist()

        # --- Refine column selection for scaling ---
        # Exclude ID columns, timepoint, and binary/one-hot encoded columns
        # Heuristic: Exclude columns that only contain 0 and 1, and ID columns.
        cols_to_exclude = set()
        for col in numerical_cols:
            if col.endswith('_id') or 'timepoint' in col.lower():
                cols_to_exclude.add(col)
                continue
            # Check if column looks binary (only 0s and 1s, ignoring NaNs)
            unique_vals = df[col].dropna().unique()
            if set(unique_vals).issubset({0, 1}):
                cols_to_exclude.add(col)
                logger.debug(f"Excluding potential binary/one-hot column from scaling: '{col}'")

        numerical_cols_to_scale = [col for col in numerical_cols if col not in cols_to_exclude]
        logger.debug(f"Numerical columns identified for scaling: {numerical_cols_to_scale}")


        if numerical_cols_to_scale: # Use the filtered list
            # Ensure columns actually exist in the dataframe before proceeding
            numerical_cols_to_scale = [col for col in numerical_cols_to_scale if col in df.columns]
            if not numerical_cols_to_scale:
                 logger.warning("No numerical columns left to scale after filtering.")
                 return df

            # Convert columns to numeric, coercing errors, before fitting/transforming
            # This should ideally be handled earlier, but double-check
            df_scaled = df.copy() # Work on a copy
            for col in numerical_cols_to_scale:
                df_scaled[col] = pd.to_numeric(df_scaled[col], errors='coerce')


            scaler_key = 'numerical_clinical'
            # Initialize and fit scaler (FIT)
            if scaler_key not in self.numerical_scalers:
                logger.info(f"Fitting StandardScaler ({scaler_key}) for numerical columns.")
                self.numerical_scalers[scaler_key] = StandardScaler()
                # Fit only on non-empty, non-all-NaN data
                fit_df = df_scaled[numerical_cols_to_scale].copy()
                # Check for columns with zero variance before fitting scaler
                zero_var_cols = fit_df.columns[fit_df.var() == 0]
                if not zero_var_cols.empty:
                    logger.warning(f"Columns with zero variance found before scaling: {zero_var_cols.tolist()}. Scaling might result in NaNs or errors for these columns.")
                    # Scaler should handle this (output 0), but good to be aware.

                if not fit_df.empty and not fit_df.isnull().all().all():
                    try:
                        # StandardScaler handles NaNs during fit by ignoring them for mean/std calculation
                        self.numerical_scalers[scaler_key].fit(fit_df)
                        self.fitted_num_cols_scale = numerical_cols_to_scale # Store fitted columns
                        logger.info(f"StandardScaler ({scaler_key}) fitted on columns: {self.fitted_num_cols_scale}")
                    except ValueError as e:
                         logger.error(f"StandardScaler ({scaler_key}) fitting failed: {e}. Removing scaler.")
                         self.numerical_scalers.pop(scaler_key, None)
                         self.fitted_num_cols_scale = []
                else:
                    logger.warning(f"Cannot fit StandardScaler ({scaler_key}): Data is empty or all NaN.")
                    self.numerical_scalers.pop(scaler_key, None) # Remove unusable scaler safely
                    self.fitted_num_cols_scale = []


            # Apply scaling (TRANSFORM)
            if scaler_key in self.numerical_scalers:
                # Ensure transform uses the same columns as fit
                cols_to_transform = [col for col in self.fitted_num_cols_scale if col in df_scaled.columns]
                if cols_to_transform:
                    logger.info(f"Applying StandardScaler ({scaler_key}) transform on columns: {cols_to_transform}")
                    # Impute NaNs *before* scaling, e.g., with 0 or mean/median
                    # Using 0 here, as KNN imputation should have run before
                    transform_subset = df_scaled[cols_to_transform].fillna(0)

                    try:
                        # Transform using the DataFrame directly
                        scaled_values = self.numerical_scalers[scaler_key].transform(transform_subset)

                        # Assign back using DataFrame constructor to maintain float type
                        scaled_df = pd.DataFrame(
                            scaled_values,
                            index=df_scaled.index,
                            columns=cols_to_transform
                        ).astype(float) # Explicitly cast to float

                        # Update original DataFrame
                        df_scaled.update(scaled_df)
                        logger.info(f"StandardScaler applied successfully to {len(cols_to_transform)} columns.")

                    except Exception as e:
                        logger.error(f"StandardScaler ({scaler_key}) transform failed: {e}. Skipping scaling for these columns.")
                        # Keep original (but coerced to numeric and imputed) values if scaling fails
                        # df_scaled[cols_to_transform] = transform_subset # Or keep df_scaled as is

                else:
                     logger.warning(f"No columns found to apply StandardScaler ({scaler_key}) transform.")

            else:
                 # If scaler couldn't be fit, maybe just return df or fillna
                 logger.warning(f"Numerical scaler ({scaler_key}) not available. Skipping scaling.")
                 # Ensure NaNs are filled if scaling is skipped (should be done by imputation)
                 # df_scaled[numerical_cols_to_scale] = df_scaled[numerical_cols_to_scale].fillna(0)

            return df_scaled # Return the modified copy

        logger.info("No numerical columns suitable for scaling were found.")
        return df # Return original if no numerical columns to scale

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer additional features from existing data.
        This should ideally happen *before* scaling if the engineered features
        themselves need scaling, or *after* if they are categorical flags.
        Let's assume they are flags created after scaling.

        Args:
            df: DataFrame with preprocessed features

        Returns:
            DataFrame with additional engineered features
        """
        df_engineered = df.copy() # Work on a copy
        logger.info("Engineering additional features...")

        # Example: Create age groups (if 'age' exists and is interpretable)
        # Note: If 'age' was scaled, these groups might be less meaningful unless based on original age.
        # We might need to pass the original unscaled df or specific columns.
        # For now, assume 'age' column exists and is somewhat interpretable (e.g., imputed but not scaled).
        age_col_name = 'age' # Assume original name before potential scaling prefix/suffix
        if age_col_name in df_engineered.columns:
            try:
                # Ensure age is numeric before comparison
                age_col = pd.to_numeric(df_engineered[age_col_name], errors='coerce')
                # Create binary flags based on thresholds
                df_engineered['is_age_lt_40'] = (age_col < 40).astype(int)
                df_engineered['is_age_40_to_60'] = ((age_col >= 40) & (age_col < 60)).astype(int)
                df_engineered['is_age_gte_60'] = (age_col >= 60).astype(int)
                logger.info("Engineered age group flags (is_age_lt_40, is_age_40_to_60, is_age_gte_60).")
            except Exception as e:
                 logger.warning(f"Could not engineer age features: {e}")

        # Example: Create performance status groups (based on Karnofsky score)
        kps_col_name = 'karnofsky_score' # Assume original name
        if kps_col_name in df_engineered.columns:
             try:
                # Ensure KPS is numeric
                kps_col = pd.to_numeric(df_engineered[kps_col_name], errors='coerce')
                df_engineered['is_kps_gte_80'] = (kps_col >= 80).astype(int)
                logger.info("Engineered high performance flag (is_kps_gte_80).")
             except Exception as e:
                 logger.warning(f"Could not engineer KPS features: {e}")


        # Example: Interaction term (Age * Grade - if grade_numeric exists)
        if 'age' in df_engineered.columns and 'grade_numeric' in df_engineered.columns:
             try:
                 df_engineered['age_x_grade'] = df_engineered['age'] * df_engineered['grade_numeric']
                 logger.info("Engineered interaction term 'age_x_grade'.")
                 # This new feature might need scaling if created before the scaling step
             except Exception as e:
                 logger.warning(f"Could not engineer age*grade interaction: {e}")


        return df_engineered

    def process_treatment_data(self, treatments_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process treatment data including encoding and feature creation.
        (This might be called separately or integrated into the main pipeline)

        Args:
            treatments_df: DataFrame with treatment information

        Returns:
            Processed treatment DataFrame
        """
        logger.info(f"Processing treatment data. Initial shape: {treatments_df.shape}")
        # Create a copy to avoid modifying the original
        df = treatments_df.copy()

        # Standardize column names
        df.columns = [col.strip().lower().replace(' ', '_').replace('-', '_') for col in df.columns]

        # Encode treatment categories (if 'category' column exists)
        if 'category' in df.columns:
            col = 'category'
            encoder_key = 'treatment_category'
            if df[col].nunique() > 1:
                if encoder_key not in self.categorical_encoders:
                    logger.info(f"Fitting OneHotEncoder for treatment '{col}'.")
                    self.categorical_encoders[encoder_key] = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                    try:
                        self.categorical_encoders[encoder_key].fit(df[[col]].astype(str))
                        logger.info(f"OneHotEncoder fitted for treatment '{col}'.")
                    except Exception as e:
                        logger.error(f"OneHotEncoder fitting failed for treatment '{col}': {e}")
                        self.categorical_encoders.pop(encoder_key, None)

                if encoder_key in self.categorical_encoders:
                    logger.debug(f"Applying OneHotEncoder transform for treatment '{col}'.")
                    try:
                        encoded_array = self.categorical_encoders[encoder_key].transform(df[[col]].astype(str))
                        if hasattr(self.categorical_encoders[encoder_key], 'get_feature_names_out'):
                            feature_names = self.categorical_encoders[encoder_key].get_feature_names_out([col])
                        else:
                            feature_names = [f"{col}_{cat}" for cat in self.categorical_encoders[encoder_key].categories_[0]]

                        encoded_df = pd.DataFrame(encoded_array, columns=feature_names, index=df.index).astype(int)
                        # Concatenate - avoid dropping original 'category' if needed elsewhere
                        df = pd.concat([df, encoded_df], axis=1)
                        logger.debug(f"Encoded treatment '{col}' into columns: {feature_names}")
                    except Exception as e:
                        logger.error(f"OneHotEncoder transform failed for treatment '{col}': {e}")
            else:
                 logger.warning(f"Skipping encoding for treatment '{col}' as it has <= 1 unique value.")


        # Compute treatment duration if start and end days are available
        if 'duration_days' not in df.columns and 'start_day' in df.columns and 'end_day' in df.columns:
            start_day = pd.to_numeric(df['start_day'], errors='coerce')
            end_day = pd.to_numeric(df['end_day'], errors='coerce')
            # Calculate duration, handle NaNs, ensure non-negative
            df['duration_days'] = (end_day - start_day).fillna(0).clip(lower=0)
            logger.info("Calculated 'duration_days' from start/end days.")
        elif 'duration_days' in df.columns:
             # Ensure existing duration_days is numeric and non-negative, fill NaNs
             df['duration_days'] = pd.to_numeric(df['duration_days'], errors='coerce').fillna(0).clip(lower=0)
             logger.debug("Cleaned existing 'duration_days'.")


        # Compute cumulative dose if dose and duration are available
        if 'dose' in df.columns and 'duration_days' in df.columns:
            # Ensure columns are numeric, fill NaNs with 0
            dose_values = pd.to_numeric(df['dose'], errors='coerce').fillna(0)
            duration_values = df['duration_days'] # Already cleaned above
            df['cumulative_dose'] = dose_values * duration_values
            logger.info("Calculated 'cumulative_dose'.")

        # Scale numerical treatment features (e.g., dose, duration) if needed
        treatment_num_cols = ['dose', 'duration_days', 'cumulative_dose']
        treatment_num_cols = [col for col in treatment_num_cols if col in df.columns]

        if treatment_num_cols:
             scaler_key = 'treatment_numerical'
             if scaler_key not in self.numerical_scalers:
                 logger.info(f"Fitting StandardScaler ({scaler_key}) for treatment numerical columns.")
                 self.numerical_scalers[scaler_key] = StandardScaler()
                 fit_df = df[treatment_num_cols].copy()
                 if not fit_df.empty and not fit_df.isnull().all().all():
                     try:
                         self.numerical_scalers[scaler_key].fit(fit_df)
                         logger.info(f"StandardScaler ({scaler_key}) fitted for treatments.")
                     except ValueError as e:
                         logger.error(f"StandardScaler ({scaler_key}) fitting failed for treatments: {e}")
                         self.numerical_scalers.pop(scaler_key, None)
                 else:
                     logger.warning(f"Cannot fit StandardScaler ({scaler_key}) for treatments: Data empty or all NaN.")
                     self.numerical_scalers.pop(scaler_key, None)

             if scaler_key in self.numerical_scalers:
                 logger.info(f"Applying StandardScaler ({scaler_key}) transform for treatments.")
                 transform_subset = df[treatment_num_cols].fillna(0)
                 try:
                     scaled_values = self.numerical_scalers[scaler_key].transform(transform_subset)
                     scaled_df = pd.DataFrame(scaled_values, index=df.index, columns=treatment_num_cols).astype(float)
                     # Add suffix to scaled columns to distinguish from original
                     scaled_df = scaled_df.add_suffix('_scaled')
                     df = pd.concat([df, scaled_df], axis=1)
                     logger.info(f"Scaled treatment columns: {scaled_df.columns.tolist()}")
                 except Exception as e:
                     logger.error(f"StandardScaler ({scaler_key}) transform failed for treatments: {e}")

        logger.info(f"Finished processing treatment data. Final shape: {df.shape}")
        return df

    def integrate_imaging_features(self,
                                clinical_df: pd.DataFrame,
                                imaging_features: Dict[Union[int, str], Dict[str, Dict[str, float]]]) -> pd.DataFrame:
        """
        Integrate imaging features (from HDF5 or dict) into the clinical DataFrame.

        Args:
            clinical_df: Preprocessed clinical DataFrame (must have 'patient_id')
            imaging_features: Dictionary mapping patient IDs (int or str) to imaging features
                              Format: {patient_id: {sequence_name: {feature_name: value}}}

        Returns:
            Integrated DataFrame with clinical and imaging features
        """
        logger.info("Integrating imaging features...")
        if 'patient_id' not in clinical_df.columns:
             logger.error("Clinical DataFrame must contain a 'patient_id' column for imaging integration.")
             raise ValueError("Clinical DataFrame must contain a 'patient_id' column.")

        integrated_df = clinical_df.copy()
        # Ensure patient_id in df is string for matching keys (which might be int or str)
        integrated_df['patient_id'] = integrated_df['patient_id'].astype(str)
        patient_ids_in_df = set(integrated_df['patient_id'])
        logger.debug(f"Integrating imaging features for {len(patient_ids_in_df)} patients in the DataFrame.")

        # --- Flatten imaging features and prepare for merge ---
        feature_rows = []
        all_feature_columns = set()

        for patient_id_key, sequences in imaging_features.items():
            patient_id_str = str(patient_id_key) # Ensure key is string for matching
            if patient_id_str in patient_ids_in_df:
                row_data = {'patient_id': patient_id_str}
                for sequence, features in sequences.items():
                    for feature_name, feature_value in features.items():
                        # Create a unique column name, e.g., t1c_mean, tumor_volume_mm3
                        col_name = f"{sequence}_{feature_name}"
                        # Replace potentially problematic characters in column names
                        col_name = col_name.replace('.', '_').replace('-', '_')
                        row_data[col_name] = feature_value
                        all_feature_columns.add(col_name)
                feature_rows.append(row_data)

        if not feature_rows:
             logger.warning("No matching imaging features found for patients in the clinical DataFrame.")
             return integrated_df # Return original df if no features to add

        # Create DataFrame from imaging features
        imaging_df = pd.DataFrame(feature_rows)
        imaging_df['patient_id'] = imaging_df['patient_id'].astype(str) # Ensure string ID for merge
        logger.info(f"Created DataFrame with imaging features for {len(imaging_df)} patients. Found {len(all_feature_columns)} unique imaging features.")

        # Merge imaging features into the main DataFrame
        integrated_df = pd.merge(integrated_df, imaging_df, on='patient_id', how='left')
        logger.info(f"Shape after merging imaging features: {integrated_df.shape}")

        # --- Impute missing imaging features ---
        imaging_cols_to_impute = list(all_feature_columns.intersection(integrated_df.columns))
        if imaging_cols_to_impute:
            logger.info(f"Imputing missing values in {len(imaging_cols_to_impute)} imaging feature columns...")
            # Ensure data is numeric before imputation
            integrated_df[imaging_cols_to_impute] = integrated_df[imaging_cols_to_impute].apply(pd.to_numeric, errors='coerce')

            imputer_key = 'imaging_features'
            # Fit imputer if not done
            if imputer_key not in self.imputers:
                logger.info(f"Fitting KNNImputer ({imputer_key}) for imaging features.")
                self.imputers[imputer_key] = KNNImputer(n_neighbors=3) # Use fewer neighbors for potentially sparse features
                fit_df = integrated_df[imaging_cols_to_impute].copy()
                if not fit_df.empty and not fit_df.isnull().all().all():
                    try:
                        self.imputers[imputer_key].fit(fit_df)
                        self.fitted_imaging_cols_impute = imaging_cols_to_impute
                        logger.info(f"KNNImputer ({imputer_key}) fitted on imaging columns: {self.fitted_imaging_cols_impute}")
                    except ValueError as e:
                         logger.error(f"KNNImputer ({imputer_key}) fitting failed for imaging: {e}")
                         self.imputers.pop(imputer_key, None)
                         self.fitted_imaging_cols_impute = []
                else:
                    logger.warning(f"Cannot fit KNNImputer ({imputer_key}) for imaging: Data empty or all NaN.")
                    self.imputers.pop(imputer_key, None)
                    self.fitted_imaging_cols_impute = []

            # Apply imputation
            if imputer_key in self.imputers:
                cols_to_transform = [col for col in self.fitted_imaging_cols_impute if col in integrated_df.columns]
                if cols_to_transform:
                    logger.info(f"Applying KNNImputer ({imputer_key}) transform for imaging.")
                    try:
                        imputed_values = self.imputers[imputer_key].transform(integrated_df[cols_to_transform])
                        imputed_df_img = pd.DataFrame(imputed_values, index=integrated_df.index, columns=cols_to_transform)
                        integrated_df.update(imputed_df_img)
                        logger.info(f"KNN imputation applied successfully to {len(cols_to_transform)} imaging columns.")
                    except Exception as e:
                        logger.error(f"KNNImputer ({imputer_key}) transform failed for imaging: {e}. Filling NaNs with 0.")
                        integrated_df[cols_to_transform] = integrated_df[cols_to_transform].fillna(0)
                else:
                     logger.warning(f"No columns found to apply KNNImputer ({imputer_key}) transform for imaging.")
            else:
                 logger.warning(f"Imaging imputer ({imputer_key}) not available. Filling NaNs with 0.")
                 integrated_df[imaging_cols_to_impute] = integrated_df[imaging_cols_to_impute].fillna(0)

        # Optionally scale imaging features (could be done here or with other numerical features)
        # If scaling here, use a separate scaler instance, e.g., self.numerical_scalers['imaging']

        logger.info("Finished integrating imaging features.")
        return integrated_df
