# susruta/tests/test_data.py
"""Tests for the data processing module."""

import os
import pytest
import numpy as np
import pandas as pd
import SimpleITK as sitk
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path # Import Path

# Import classes directly (fixtures are now in conftest.py)
from susruta.data import ClinicalDataProcessor, EfficientMRIProcessor
from susruta.utils import MemoryTracker

# Fixtures like synthetic_clinical_data, synthetic_imaging_features, etc.,
# are automatically injected by pytest from conftest.py


class TestClinicalDataProcessor:
    """Test suite for the ClinicalDataProcessor class."""

    def test_initialization(self, clinical_processor): # Use fixture
        """Test initialization of the ClinicalDataProcessor."""
        assert hasattr(clinical_processor, 'categorical_encoders')
        assert hasattr(clinical_processor, 'numerical_scalers')
        assert hasattr(clinical_processor, 'imputers')
        assert isinstance(clinical_processor.categorical_encoders, dict)
        assert isinstance(clinical_processor.numerical_scalers, dict)
        assert isinstance(clinical_processor.imputers, dict)

    def test_impute_missing_values(self, clinical_processor, synthetic_clinical_data):
        """Test missing value imputation."""
        data = synthetic_clinical_data.copy()
        # Use a fresh processor instance for this specific test if state matters
        processor = ClinicalDataProcessor()
        imputed_data = processor._impute_missing_values(data)

        # Check that numerical columns used for imputation are filled
        numerical_cols = data.select_dtypes(include=np.number).columns
        numerical_cols = [col for col in numerical_cols if not col.endswith('_id') and 'date' not in col.lower()]
        assert not imputed_data[numerical_cols].isna().any().any()

        # Check that categorical columns are filled
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        assert not imputed_data[categorical_cols].isna().any().any()

        assert imputed_data.shape == data.shape
        assert 'numerical' in processor.imputers

    def test_encode_categorical_variables(self, clinical_processor, synthetic_clinical_data):
        """Test categorical variable encoding."""
        data = synthetic_clinical_data[['patient_id', 'sex', 'grade']].copy()
        processor = ClinicalDataProcessor() # Fresh instance
        # Need to impute first if there are NaNs
        data_imputed = processor._impute_missing_values(data)
        encoded_data = processor._encode_categorical_variables(data_imputed)

        assert 'sex' not in encoded_data.columns
        assert 'grade' not in encoded_data.columns
        assert any(col.startswith('sex_') for col in encoded_data.columns)
        assert any(col.startswith('grade_') for col in encoded_data.columns)
        assert 'sex' in processor.categorical_encoders
        assert 'grade' in processor.categorical_encoders

    def test_scale_numerical_features(self, clinical_processor, synthetic_clinical_data):
        """Test numerical feature scaling."""
        data = synthetic_clinical_data[['patient_id', 'age', 'karnofsky_score']].copy()
        processor = ClinicalDataProcessor() # Fresh instance
        # Need to impute first if there are NaNs
        data_imputed = processor._impute_missing_values(data)
        scaled_data = processor._scale_numerical_features(data_imputed)

        assert scaled_data.shape == data.shape

        # Check scaled columns exist and have near-zero mean and unit variance
        scaled_cols = ['age', 'karnofsky_score']
        for col in scaled_cols:
             if col in scaled_data.columns: # Check if column was actually scaled
                 assert col in scaled_data.columns
                 # Use pytest.approx for floating point comparisons
                 assert scaled_data[col].mean() == pytest.approx(0.0, abs=1e-6)
                 # Increase tolerance further for std dev with small samples
                 assert scaled_data[col].std() == pytest.approx(1.0, abs=0.3) # Increased tolerance to 0.3

        assert 'numerical' in processor.numerical_scalers


    def test_engineer_features(self, clinical_processor, synthetic_clinical_data):
        """Test feature engineering."""
        # Use a copy of the original data before preprocessing for clarity
        data = synthetic_clinical_data[['patient_id', 'age', 'karnofsky_score']].copy()
        # Impute NaNs as engineer_features might expect non-null values
        processor = ClinicalDataProcessor()
        data_imputed = processor._impute_missing_values(data)
        engineered_data = processor._engineer_features(data_imputed) # Use fixture processor

        assert 'age_group_young' in engineered_data.columns
        assert 'age_group_middle' in engineered_data.columns
        assert 'age_group_elder' in engineered_data.columns
        assert 'high_performance' in engineered_data.columns
        assert set(engineered_data['high_performance'].unique()).issubset({0, 1})

    def test_preprocess_clinical_data_end_to_end(self, clinical_processor, synthetic_clinical_data):
        """Test the entire preprocessing pipeline."""
        processor = ClinicalDataProcessor() # Fresh instance for full pipeline
        result = processor.preprocess_clinical_data(synthetic_clinical_data.copy()) # Pass copy

        # Check original categorical columns are removed
        assert 'sex' not in result.columns
        assert 'grade' not in result.columns
        assert 'histology' not in result.columns
        assert 'location' not in result.columns

        # Check engineered features are present
        assert any(col.startswith('age_group_') for col in result.columns)
        assert 'high_performance' in result.columns

        # Check one-hot encoded features are present
        assert any(col.startswith('sex_') for col in result.columns)
        assert any(col.startswith('grade_') for col in result.columns)
        assert any(col.startswith('histology_') for col in result.columns)
        assert any(col.startswith('location_') for col in result.columns)

        # Check numerical features are scaled (near zero mean)
        original_numerical = ['age', 'karnofsky_score'] # Add others if present
        for col in original_numerical:
             if col in result.columns: # Check if it wasn't removed (e.g., if it was an ID)
                 assert result[col].mean() == pytest.approx(0.0, abs=1e-6)
                 assert result[col].std() == pytest.approx(1.0, abs=0.3) # Increased tolerance to 0.3


    def test_process_treatment_data(self, clinical_processor, synthetic_treatment_data):
        """Test treatment data processing."""
        processor = ClinicalDataProcessor() # Fresh instance
        processed = processor.process_treatment_data(synthetic_treatment_data.copy()) # Pass copy

        assert any(col.startswith('category_') for col in processed.columns)
        if 'dose' in processed.columns and 'duration_days' in processed.columns:
            # Check cumulative dose calculation (handle potential NaNs)
            assert 'cumulative_dose' in processed.columns
            assert not processed['cumulative_dose'].isna().any()


    def test_integrate_multimodal_data(self, clinical_processor, synthetic_clinical_data, synthetic_imaging_features):
        """Test multimodal data integration."""
        processor = ClinicalDataProcessor() # Fresh instance
        # Preprocess clinical data first, as integrate expects it
        clinical_processed = processor.preprocess_clinical_data(synthetic_clinical_data.copy())
        integrated = processor.integrate_multimodal_data(clinical_processed, synthetic_imaging_features)

        # Check that imaging features were added
        expected_imaging_cols = []
        for pid, sequences in synthetic_imaging_features.items():
            for seq, features in sequences.items():
                for feat_name in features.keys():
                    expected_imaging_cols.append(f"{seq}_{feat_name}")
        expected_imaging_cols = sorted(list(set(expected_imaging_cols)))

        for col in expected_imaging_cols:
            assert col in integrated.columns
            # Check that imputed imaging columns have no NaNs
            assert not integrated[col].isna().any()


        # Check that original (processed) clinical columns are still present
        for col in clinical_processed.columns:
            assert col in integrated.columns

        assert len(integrated) == len(clinical_processed)
        assert 'imaging' in processor.imputers


# --- EfficientMRIProcessor Tests ---

class TestEfficientMRIProcessor:
    """Test suite for the EfficientMRIProcessor class."""

    def test_initialization(self, mri_processor): # Use fixture
        """Test initialization of EfficientMRIProcessor."""
        assert mri_processor.memory_limit_mb == 1000
        assert hasattr(mri_processor, 'memory_tracker')

    # --- FIX: Correct Path patching ---
    @patch('susruta.data.mri.Path') # Patch Path where it's used
    @patch('susruta.data.mri.sitk')
    def test_load_nifti_metadata(self, mock_sitk, mock_Path, mri_processor): # Renamed mock_exists to mock_Path
        """Test loading of NIfTI metadata."""
        # Configure the mock Path instance's exists method
        mock_path_instance = mock_Path.return_value
        mock_path_instance.exists.return_value = True

        mock_reader = MagicMock()
        mock_sitk.ImageFileReader.return_value = mock_reader
        mock_reader.GetSize.return_value = (256, 256, 155) # X, Y, Z
        mock_reader.GetSpacing.return_value = (1.0, 1.0, 1.0)
        mock_reader.GetOrigin.return_value = (0.0, 0.0, 0.0)
        mock_reader.GetDirection.return_value = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

        metadata = mri_processor.load_nifti_metadata('dummy_path.nii.gz')

        # Check Path was called with the correct argument
        mock_Path.assert_called_once_with('dummy_path.nii.gz')
        # Check the mock instance's exists was called
        mock_path_instance.exists.assert_called_once()

        mock_sitk.ImageFileReader.assert_called_once()
        mock_reader.SetFileName.assert_called_once_with('dummy_path.nii.gz')
        mock_reader.ReadImageInformation.assert_called_once()
        assert metadata['size'] == (256, 256, 155) # Check XYZ order

    # --- FIX: Correct Path patching ---
    @patch('susruta.data.mri.Path') # Patch Path where it's used
    @patch('susruta.data.mri.sitk')
    def test_compute_bounding_box(self, mock_sitk, mock_Path, mri_processor): # Renamed mock_exists to mock_Path
        """Test bounding box computation."""
        # Configure the mock Path instance's exists method
        mock_path_instance = mock_Path.return_value
        mock_path_instance.exists.return_value = True

        mock_mask = MagicMock(spec=sitk.Image)
        mock_casted_mask = MagicMock(spec=sitk.Image) # Mock for casted mask
        mock_stats = MagicMock(spec=sitk.LabelStatisticsImageFilter)
        mock_stats_basic = MagicMock(spec=sitk.StatisticsImageFilter) # Mock for initial sum check

        # --- Mock Setup ---
        mock_sitk.ReadImage.return_value = mock_mask
        mock_sitk.StatisticsImageFilter.return_value = mock_stats_basic
        mock_sitk.Cast.return_value = mock_casted_mask # Mock Cast
        mock_sitk.LabelStatisticsImageFilter.return_value = mock_stats
        mock_stats.Execute.side_effect = lambda img, lbl: None # Mock execute
        mock_stats_basic.Execute.side_effect = lambda img: None # Mock execute for basic stats

        # --- Test Case 1: Mask needs casting (assume label 1 exists) ---
        mock_stats_basic.GetSum.return_value = 100 # Non-empty
        mock_mask.GetPixelIDValue.return_value = sitk.sitkInt16 # Not UInt8
        mock_sitk.sitkUInt8 = sitk.sitkUInt8 # Use actual sitk constant
        mock_stats.HasLabel.return_value = True
        # Bounding box format: [xStart, yStart, zStart, xSize, ySize, zSize]
        mock_stats.GetBoundingBox.return_value = (10, 20, 30, 100, 110, 120)

        bbox_min, bbox_max = mri_processor.compute_bounding_box('dummy_mask.nii.gz')

        # Check Path was called
        mock_Path.assert_called_once_with('dummy_mask.nii.gz')
        # Check the mock instance's exists was called
        mock_path_instance.exists.assert_called_once()

        mock_sitk.ReadImage.assert_called_once_with('dummy_mask.nii.gz')
        mock_stats_basic.Execute.assert_called_once_with(mock_mask) # Check initial sum check
        mock_sitk.Cast.assert_called_once_with(mock_mask, sitk.sitkUInt8) # Cast applied
        # Execute should be called with the casted mask for LabelStats
        mock_stats.Execute.assert_called_once_with(mock_casted_mask, mock_casted_mask)
        mock_stats.HasLabel.assert_called_once_with(1)
        mock_stats.GetBoundingBox.assert_called_once_with(1)
        # Check expected bbox_min/max based on SITK format and mri.py logic
        assert bbox_min == [10, 20, 30] # [xStart, yStart, zStart]
        assert bbox_max == [10+100-1, 20+110-1, 30+120-1] # [start + size - 1]
        assert bbox_max == [109, 129, 149]

        # --- Reset mocks for Case 2 ---
        mock_Path.reset_mock()
        mock_path_instance.reset_mock()
        mock_sitk.ReadImage.reset_mock()
        mock_sitk.Cast.reset_mock()
        mock_stats.reset_mock()
        mock_stats_basic.reset_mock()
        mock_stats.Execute.reset_mock()
        mock_stats_basic.Execute.reset_mock()
        mock_stats.HasLabel.reset_mock()
        mock_stats.GetBoundingBox.reset_mock()

        # --- Test Case 2: Mask is already integer type (UInt8) ---
        mock_stats_basic.GetSum.return_value = 100 # Non-empty
        mock_mask.GetPixelIDValue.return_value = sitk.sitkUInt8 # Already UInt8
        mock_stats.HasLabel.return_value = True
        mock_stats.GetBoundingBox.return_value = (5, 15, 25, 50, 60, 70)

        bbox_min, bbox_max = mri_processor.compute_bounding_box('binary_mask.nii.gz')

        mock_Path.assert_called_once_with('binary_mask.nii.gz')
        mock_path_instance.exists.assert_called_once()
        mock_sitk.ReadImage.assert_called_once_with('binary_mask.nii.gz')
        mock_stats_basic.Execute.assert_called_once_with(mock_mask)
        mock_sitk.Cast.assert_not_called() # Should not be called
        # Execute should be called with the original mask for LabelStats
        mock_stats.Execute.assert_called_once_with(mock_mask, mock_mask)
        mock_stats.HasLabel.assert_called_once_with(1)
        mock_stats.GetBoundingBox.assert_called_once_with(1)
        assert bbox_min == [5, 15, 25]
        assert bbox_max == [5+50-1, 15+60-1, 25+70-1]
        assert bbox_max == [54, 74, 94]

        # --- Reset mocks for Case 3 ---
        mock_Path.reset_mock()
        mock_path_instance.reset_mock()
        mock_sitk.ReadImage.reset_mock()
        mock_stats.reset_mock()
        mock_stats_basic.reset_mock()
        mock_stats.Execute.reset_mock()
        mock_stats_basic.Execute.reset_mock()
        mock_stats.HasLabel.reset_mock()
        mock_stats.GetBoundingBox.reset_mock()

        # --- Test Case 3: Empty mask ---
        mock_stats_basic.GetSum.return_value = 0 # Empty mask
        mock_mask.GetSize.return_value = (100, 110, 120) # Need size for empty case

        bbox_min, bbox_max = mri_processor.compute_bounding_box('empty_mask.nii.gz')

        mock_Path.assert_called_once_with('empty_mask.nii.gz')
        mock_path_instance.exists.assert_called_once()
        mock_sitk.ReadImage.assert_called_once_with('empty_mask.nii.gz')
        mock_stats_basic.Execute.assert_called_once_with(mock_mask)
        mock_sitk.Cast.assert_not_called() # Not called if sum is 0
        mock_stats.Execute.assert_not_called() # LabelStats not called if sum is 0
        assert bbox_min == [0, 0, 0]
        assert bbox_max == [99, 109, 119] # Full image bounds

        # --- Reset mocks for Case 4 ---
        mock_Path.reset_mock()
        mock_path_instance.reset_mock()
        mock_sitk.ReadImage.reset_mock()
        mock_stats.reset_mock()
        mock_stats_basic.reset_mock()
        mock_stats.Execute.reset_mock()
        mock_stats_basic.Execute.reset_mock()
        mock_stats.HasLabel.reset_mock()
        mock_stats.GetBoundingBox.reset_mock()

        # --- Test Case 4: Label 1 not found, but label 2 exists ---
        mock_stats_basic.GetSum.return_value = 100 # Non-empty
        mock_mask.GetPixelIDValue.return_value = sitk.sitkUInt8
        mock_stats.HasLabel.side_effect = lambda label: label == 2 # Only label 2 exists
        mock_stats.GetLabels.return_value = [0, 2] # Labels present
        mock_stats.GetBoundingBox.return_value = (7, 17, 27, 70, 80, 90) # BBox for label 2

        bbox_min, bbox_max = mri_processor.compute_bounding_box('label2_mask.nii.gz')

        mock_Path.assert_called_once_with('label2_mask.nii.gz')
        mock_path_instance.exists.assert_called_once()
        mock_sitk.ReadImage.assert_called_once_with('label2_mask.nii.gz')
        mock_stats_basic.Execute.assert_called_once_with(mock_mask)
        mock_sitk.Cast.assert_not_called()
        mock_stats.Execute.assert_called_once_with(mock_mask, mock_mask)
        mock_stats.HasLabel.assert_any_call(1) # Check for label 1 first
        mock_stats.GetBoundingBox.assert_called_once_with(2) # Should use label 2
        assert bbox_min == [7, 17, 27]
        assert bbox_max == [7+70-1, 17+80-1, 27+90-1]
        assert bbox_max == [76, 96, 116]


    @patch('susruta.data.mri.sitk')
    def test_compute_roi_features(self, mock_sitk, mri_processor):
        """Test ROI feature computation."""
        # Use realistic dimensions
        img_roi_array = np.random.rand(20, 30, 40) * 200 # Z, Y, X
        mask_roi_array = np.zeros((20, 30, 40), dtype=np.uint8)
        mask_roi_array[5:15, 10:20, 15:25] = 1 # Create a non-empty mask region

        mock_img_roi = MagicMock(spec=sitk.Image)
        mock_mask_roi = MagicMock(spec=sitk.Image)
        mock_sitk.GetArrayFromImage.side_effect = [img_roi_array, mask_roi_array]
        # Mock GetPixelIDValue for the mask
        mock_mask_roi.GetPixelIDValue.return_value = sitk.sitkUInt8

        features = mri_processor._compute_roi_features(mock_img_roi, mock_mask_roi)

        assert mock_sitk.GetArrayFromImage.call_count == 2
        assert 'mean' in features and 'volume_voxels' in features

        masked_voxels = img_roi_array[mask_roi_array > 0]
        assert features['volume_voxels'] == masked_voxels.size
        assert features['mean'] == pytest.approx(np.mean(masked_voxels))
        assert features['std'] == pytest.approx(np.std(masked_voxels))
        assert features['min'] == pytest.approx(np.min(masked_voxels))
        assert features['max'] == pytest.approx(np.max(masked_voxels))
        assert features['p50'] == pytest.approx(np.median(masked_voxels))

        # Test empty mask case
        mock_sitk.GetArrayFromImage.reset_mock()
        empty_mask_array = np.zeros((20, 30, 40), dtype=np.uint8)
        mock_sitk.GetArrayFromImage.side_effect = [img_roi_array, empty_mask_array]
        features_empty = mri_processor._compute_roi_features(mock_img_roi, mock_mask_roi)
        assert features_empty['volume_voxels'] == 0
        assert features_empty['mean'] == 0 # Check default values for empty mask


    def test_combine_chunk_features(self, mri_processor):
        """Test combining features from multiple chunks."""
        chunks = [
            {'mean': 10.0, 'std': 2.0, 'min': 5.0, 'max': 15.0, 'p25': 8.0, 'p50': 10.0, 'p75': 12.0, 'volume_voxels': 100},
            {'mean': 20.0, 'std': 3.0, 'min': 10.0, 'max': 25.0, 'p25': 18.0, 'p50': 20.0, 'p75': 22.0, 'volume_voxels': 200},
            {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'p25': 0.0, 'p50': 0.0, 'p75': 0.0, 'volume_voxels': 0} # Empty chunk
        ]
        combined = mri_processor._combine_chunk_features(chunks)
        assert combined['volume_voxels'] == 300
        assert combined['min'] == 5.0 # Min of non-empty chunks
        assert combined['max'] == 25.0 # Max of non-empty chunks

        # Calculate expected combined mean and std carefully
        total_volume = 300
        weighted_mean_sum = (10.0 * 100) + (20.0 * 200) + (0.0 * 0)
        expected_mean = weighted_mean_sum / total_volume

        e_x2_1 = chunks[0]['std']**2 + chunks[0]['mean']**2
        e_x2_2 = chunks[1]['std']**2 + chunks[1]['mean']**2
        e_x2_3 = chunks[2]['std']**2 + chunks[2]['mean']**2 # Will be 0

        weighted_e_x2_sum = (chunks[0]['volume_voxels'] * e_x2_1) + \
                            (chunks[1]['volume_voxels'] * e_x2_2) + \
                            (chunks[2]['volume_voxels'] * e_x2_3)
        combined_e_x2 = weighted_e_x2_sum / total_volume
        combined_variance = combined_e_x2 - expected_mean**2
        expected_std = np.sqrt(max(0, combined_variance)) # Ensure non-negative

        assert combined['mean'] == pytest.approx(expected_mean)
        assert combined['std'] == pytest.approx(expected_std)

        # Test combining only empty chunks
        empty_chunks = [
             {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'p25': 0.0, 'p50': 0.0, 'p75': 0.0, 'volume_voxels': 0},
             {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'p25': 0.0, 'p50': 0.0, 'p75': 0.0, 'volume_voxels': 0}
        ]
        combined_empty = mri_processor._combine_chunk_features(empty_chunks)
        assert combined_empty['volume_voxels'] == 0
        assert combined_empty['mean'] == 0
        assert combined_empty['min'] == 0.0
        assert combined_empty['max'] == 0.0


    # --- FIX: Correct Path patching ---
    @patch('susruta.data.mri.Path') # Patch Path where it's used
    @patch('susruta.data.mri.sitk')
    def test_extract_roi_features_simplified(self, mock_sitk, mock_Path, mri_processor): # Renamed mock_exists to mock_Path
        """Test feature extraction from ROI (simplified, no chunking)."""
        # Configure the mock Path instance's exists method
        mock_path_instance = mock_Path.return_value
        mock_path_instance.exists.return_value = True

        # --- Mock Setup ---
        mock_img = MagicMock(spec=sitk.Image)
        mock_mask = MagicMock(spec=sitk.Image)
        mock_sitk.ReadImage.side_effect = lambda p: mock_img if 'img' in p else mock_mask

        bbox_min, bbox_max = ([10, 20, 30], [100, 110, 140]) # Example ROI

        # Mock compute_bounding_box directly on the processor instance
        mri_processor.compute_bounding_box = MagicMock(return_value=(bbox_min, bbox_max))

        metadata = {'size': (256, 256, 155), 'spacing': (1.0, 1.0, 1.0), 'direction': np.eye(3).flatten(), 'origin': (0,0,0)}
        # Mock load_nifti_metadata directly on the processor instance
        mri_processor.load_nifti_metadata = MagicMock(return_value=metadata)

        # Mock RegionOfInterest to return a dummy image
        mock_roi_img_instance = MagicMock(spec=sitk.Image)
        mock_roi_mask_instance = MagicMock(spec=sitk.Image)
        def mock_roi_func(image, size, index):
            # Return the appropriate mock based on the input image
            return mock_roi_mask_instance if image == mock_mask else mock_roi_img_instance
        mock_sitk.RegionOfInterest.side_effect = mock_roi_func

        # Mock feature computation results
        roi_features_result = {'mean': 150.0, 'std': 30.0, 'min': 50.0, 'max': 250.0, 'p25': 120.0, 'p50': 150.0, 'p75': 180.0, 'volume_voxels': 5000}

        # Mock _compute_roi_features directly on the processor instance
        mri_processor._compute_roi_features = MagicMock(return_value=roi_features_result)
        mri_processor.memory_tracker.log_memory = MagicMock() # Mock logging

        # --- Test ROI case ---
        print("Testing ROI extraction...")
        features = mri_processor.extract_roi_features('img.nii.gz', 'mask.nii.gz')

        # Check Path was called for both files
        mock_Path.assert_any_call('img.nii.gz')
        mock_Path.assert_any_call('mask.nii.gz')
        # Check the mock instance's exists was called (at least twice)
        assert mock_path_instance.exists.call_count >= 2

        mri_processor.compute_bounding_box.assert_called_with('mask.nii.gz')
        mri_processor.load_nifti_metadata.assert_called_with('img.nii.gz')
        # Check RegionOfInterest calls
        assert mock_sitk.RegionOfInterest.call_count == 2
        # Check that _compute_roi_features was called with the correct mock ROI images
        mri_processor._compute_roi_features.assert_called_once_with(mock_roi_img_instance, mock_roi_mask_instance)
        assert features == roi_features_result


    @patch('susruta.data.mri.EfficientMRIProcessor._extract_tumor_features')
    @patch('susruta.data.mri.EfficientMRIProcessor.extract_roi_features')
    @patch('glob.glob')
    @patch('os.path.exists') # Keep os.path.exists patch if needed by other parts
    def test_extract_features_for_patient(self, mock_os_exists, mock_glob, mock_extract_roi, mock_extract_tumor, mri_processor):
        """Test feature extraction for all sequences of a patient (updated)."""
        # Use Patient 3, Timepoint 1 from the fixture setup
        patient_num = 3
        tp_num = 1
        # Construct the specific timepoint directory path expected by the function
        base_dir = '/path/to/dummy/base' # Base dir isn't used directly by the function anymore
        timepoint_dir = os.path.join(base_dir, f'PatientID_{patient_num:04d}', f'Timepoint_{tp_num}')

        # Define expected file paths within the timepoint_dir
        mask_file = os.path.join(timepoint_dir, 'tumorMask_patient3_tp1.nii.gz')
        t1c_file = os.path.join(timepoint_dir, 't1c_patient3_tp1.nii.gz')
        t2w_file = os.path.join(timepoint_dir, 't2w_patient3_tp1.nii.gz')
        # Add paths for other default sequences if they exist in the test setup
        t1n_file = os.path.join(timepoint_dir, 't1n_patient3_tp1.nii.gz')
        t2f_file = os.path.join(timepoint_dir, 't2f_patient3_tp1.nii.gz')


        # Define glob patterns relative to the timepoint_dir
        mask_pattern_glob = os.path.join(timepoint_dir, '*tumorMask*.nii*')
        t1c_pattern_glob = os.path.join(timepoint_dir, '*t1c*.nii*')
        t2w_pattern_glob = os.path.join(timepoint_dir, '*t2w*.nii*')
        t1n_pattern_glob = os.path.join(timepoint_dir, '*t1n*.nii*')
        t2f_pattern_glob = os.path.join(timepoint_dir, '*t2f*.nii*')

        # Configure glob mock
        mock_glob.side_effect = lambda pattern: {
            mask_pattern_glob: [mask_file],
            t1c_pattern_glob: [t1c_file],
            t2w_pattern_glob: [t2w_file],
            t1n_pattern_glob: [t1n_file], # Assume t1n exists
            t2f_pattern_glob: [], # Simulate missing t2f
        }.get(pattern, []) # Return empty list for non-matching patterns

        # Configure ROI extraction mock
        def mock_extract_roi_side_effect(img_path, mask_path_arg):
            assert mask_path_arg == mask_file # Check correct mask path is used
            if img_path == t1c_file: return {'mean': 150.0, 'volume_voxels': 5000}
            elif img_path == t2w_file: return {'mean': 180.0, 'volume_voxels': 5200}
            elif img_path == t1n_file: return {'mean': 120.0, 'volume_voxels': 4800}
            else: pytest.fail(f"Unexpected img_path: {img_path}")
        mock_extract_roi.side_effect = mock_extract_roi_side_effect
        mock_extract_tumor.return_value = {'volume_mm3': 15000.0, 'roundness': 0.7}

        # Mock Path(data_dir).is_dir() to return True
        with patch('pathlib.Path.is_dir', return_value=True):
            # --- Test successful case with default sequences ---
            features_default = mri_processor.extract_features_for_patient(
                patient_id=patient_num,
                data_dir=timepoint_dir, # Pass the specific timepoint directory
                timepoint=tp_num,
                sequences=None # Use default
            )

        # Glob called for defaults (t1c, t1n, t2f, t2w) + mask = 5 times
        assert mock_glob.call_count == 5
        mock_glob.assert_any_call(mask_pattern_glob)
        mock_glob.assert_any_call(t1c_pattern_glob)
        mock_glob.assert_any_call(t2w_pattern_glob)
        mock_glob.assert_any_call(t1n_pattern_glob)
        mock_glob.assert_any_call(t2f_pattern_glob)
        assert mock_extract_roi.call_count == 3 # Called for t1c, t2w, t1n
        mock_extract_tumor.assert_called_once_with(mask_file)
        assert 't1c' in features_default and 't2w' in features_default and 't1n' in features_default and 'tumor' in features_default
        assert 't2f' not in features_default # Check t2f wasn't found

        # --- Reset mocks for specific sequence test ---
        mock_glob.reset_mock(); mock_glob.side_effect = lambda pattern: {
            mask_pattern_glob: [mask_file], t1c_pattern_glob: [t1c_file] # Only provide t1c
        }.get(pattern, [])
        mock_extract_roi.reset_mock(); mock_extract_roi.side_effect = mock_extract_roi_side_effect
        mock_extract_tumor.reset_mock(); mock_extract_tumor.return_value = {'volume_mm3': 15000.0, 'roundness': 0.7}

        # --- Test successful case with specific sequences ---
        with patch('pathlib.Path.is_dir', return_value=True):
            features_specific = mri_processor.extract_features_for_patient(
                patient_id=patient_num,
                data_dir=timepoint_dir, # Pass the specific timepoint directory
                timepoint=tp_num,
                sequences=['t1c'] # Request only t1c
            )

        # Glob called for t1c + mask = 2 times
        assert mock_glob.call_count == 2
        mock_glob.assert_any_call(mask_pattern_glob)
        mock_glob.assert_any_call(t1c_pattern_glob)
        assert mock_extract_roi.call_count == 1 # Called only for t1c
        mock_extract_tumor.assert_called_once_with(mask_file)
        assert 't1c' in features_specific and 'tumor' in features_specific
        assert 't2w' not in features_specific # Check t2w wasn't processed


        # --- Test error case - no data directory ---
        # Mock Path(data_dir).is_dir() to return False
        with patch('pathlib.Path.is_dir', return_value=False):
            features_no_dir = mri_processor.extract_features_for_patient(
                patient_id=999,
                data_dir='/nonexistent/PatientID_0999/Timepoint_1',
                timepoint=1
            )
        assert features_no_dir == {} # Should return empty dict

        # --- Test error case - no tumor mask ---
        mock_glob.reset_mock()
        # Simulate glob finding sequence files but not mask file
        mock_glob.side_effect = lambda pattern: [t1c_file] if pattern == t1c_pattern_glob else []
        with patch('pathlib.Path.is_dir', return_value=True):
            features_no_mask = mri_processor.extract_features_for_patient(
                patient_id=patient_num,
                data_dir=timepoint_dir,
                timepoint=tp_num,
                sequences=['t1c']
            )
        # Check that glob was called for the mask pattern
        mock_glob.assert_any_call(mask_pattern_glob)
        assert features_no_mask == {} # Should return empty if mask missing


    # --- FIX: Correct Path patching ---
    @patch('susruta.data.mri.Path') # Patch Path where it's used
    @patch('susruta.data.mri.sitk')
    def test_extract_tumor_features(self, mock_sitk, mock_Path, mri_processor): # Renamed mock_exists to mock_Path
        """Test tumor shape and volume feature extraction."""
        # Configure the mock Path instance's exists method
        mock_path_instance = mock_Path.return_value
        mock_path_instance.exists.return_value = True

        mock_mask = MagicMock(spec=sitk.Image)
        mock_sitk.ReadImage.return_value = mock_mask
        mock_shape_stats = MagicMock(spec=sitk.LabelShapeStatisticsImageFilter)
        mock_stats_basic = MagicMock(spec=sitk.StatisticsImageFilter) # Mock for initial sum check

        mock_sitk.StatisticsImageFilter.return_value = mock_stats_basic
        mock_sitk.LabelShapeStatisticsImageFilter.return_value = mock_shape_stats

        # Test case with valid label 1
        mock_stats_basic.GetSum.return_value = 100 # Non-empty
        mock_mask.GetPixelIDValue.return_value = sitk.sitkUInt8 # Correct type
        mock_shape_stats.HasLabel.return_value = True
        mock_shape_stats.GetPhysicalSize.return_value = 15000.0
        mock_shape_stats.GetPerimeter.return_value = 2000.0
        mock_shape_stats.GetElongation.return_value = 0.5
        mock_shape_stats.GetRoundness.return_value = 0.7
        mock_shape_stats.GetFeretDiameter.return_value = 50.0

        features = mri_processor._extract_tumor_features('dummy_mask.nii.gz')

        # Check Path was called
        mock_Path.assert_called_once_with('dummy_mask.nii.gz')
        # Check the mock instance's exists was called
        mock_path_instance.exists.assert_called_once()

        assert features['volume_mm3'] == 15000.0
        assert features['surface_area_mm2'] == 2000.0 # Check corrected name
        assert features['elongation'] == 0.5
        assert features['roundness'] == 0.7
        assert features['feret_diameter_mm'] == 50.0

        # Test case with no label 1, but label 2 exists
        mock_Path.reset_mock()
        mock_path_instance.reset_mock()
        mock_stats_basic.reset_mock()
        mock_shape_stats.reset_mock()
        mock_stats_basic.GetSum.return_value = 100 # Non-empty
        mock_mask.GetPixelIDValue.return_value = sitk.sitkUInt8
        mock_shape_stats.HasLabel.side_effect = lambda label: label == 2 # Only label 2 exists
        mock_shape_stats.GetLabels.return_value = [0, 2]
        # Return different values for label 2
        mock_shape_stats.GetPhysicalSize.return_value = 16000.0
        mock_shape_stats.GetPerimeter.return_value = 2100.0
        mock_shape_stats.GetElongation.return_value = 0.6
        mock_shape_stats.GetRoundness.return_value = 0.8
        mock_shape_stats.GetFeretDiameter.return_value = 55.0

        features_l2 = mri_processor._extract_tumor_features('label2_mask.nii.gz')

        mock_Path.assert_called_once_with('label2_mask.nii.gz')
        mock_path_instance.exists.assert_called_once()
        mock_shape_stats.GetPhysicalSize.assert_called_with(2) # Check called with label 2
        assert features_l2['volume_mm3'] == 16000.0
        assert features_l2['surface_area_mm2'] == 2100.0
        assert features_l2['elongation'] == 0.6
        assert features_l2['roundness'] == 0.8
        assert features_l2['feret_diameter_mm'] == 55.0

        # Test case with empty mask
        mock_Path.reset_mock()
        mock_path_instance.reset_mock()
        mock_stats_basic.reset_mock()
        mock_shape_stats.reset_mock()
        mock_stats_basic.GetSum.return_value = 0 # Empty mask
        features_no_label = mri_processor._extract_tumor_features('empty_mask.nii.gz')

        mock_Path.assert_called_once_with('empty_mask.nii.gz')
        mock_path_instance.exists.assert_called_once()
        assert features_no_label['volume_mm3'] == 0
        assert features_no_label['surface_area_mm2'] == 0
        assert features_no_label['elongation'] == 0
        assert features_no_label['roundness'] == 0
        assert features_no_label['feret_diameter_mm'] == 0
        mock_shape_stats.Execute.assert_not_called() # LabelShapeStats shouldn't be executed
