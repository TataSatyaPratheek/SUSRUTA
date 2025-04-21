"""Tests for the data processing module."""

import os
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

from susruta.data import ClinicalDataProcessor, EfficientMRIProcessor
from susruta.utils import MemoryTracker


class TestClinicalDataProcessor:
    """Test suite for the ClinicalDataProcessor class."""
    
    def test_initialization(self):
        """Test initialization of the ClinicalDataProcessor."""
        processor = ClinicalDataProcessor()
        assert hasattr(processor, 'categorical_encoders')
        assert hasattr(processor, 'numerical_scalers')
        assert hasattr(processor, 'imputers')
        assert isinstance(processor.categorical_encoders, dict)
        assert isinstance(processor.numerical_scalers, dict)
        assert isinstance(processor.imputers, dict)
    
    def test_impute_missing_values(self, synthetic_clinical_data):
        """Test missing value imputation."""
        # Create a copy with some missing values
        data = synthetic_clinical_data.copy()
        data.loc[0, 'age'] = np.nan
        data.loc[1, 'sex'] = np.nan
        data.loc[2, 'karnofsky_score'] = np.nan
        
        processor = ClinicalDataProcessor()
        imputed_data = processor._impute_missing_values(data)
        
        # Check that missing values are filled
        assert not imputed_data.isna().any().any()
        assert imputed_data.shape == data.shape
        
        # Check that imputer was created for numerical features
        assert 'numerical' in processor.imputers
    
    def test_encode_categorical_variables(self, synthetic_clinical_data):
        """Test categorical variable encoding."""
        processor = ClinicalDataProcessor()
        
        # Focus on a subset of columns to make testing easier
        data = synthetic_clinical_data[['patient_id', 'sex', 'grade']].copy()
        
        encoded_data = processor._encode_categorical_variables(data)
        
        # Check that categorical columns are removed
        assert 'sex' not in encoded_data.columns
        assert 'grade' not in encoded_data.columns
        
        # Check that one-hot encoded columns are created
        assert any(col.startswith('sex_') for col in encoded_data.columns)
        assert any(col.startswith('grade_') for col in encoded_data.columns)
        
        # Check that encoders were created
        assert 'sex' in processor.categorical_encoders
        assert 'grade' in processor.categorical_encoders
    
    def test_scale_numerical_features(self, synthetic_clinical_data):
        """Test numerical feature scaling."""
        processor = ClinicalDataProcessor()
        
        # Focus on a subset of columns
        data = synthetic_clinical_data[['patient_id', 'age', 'karnofsky_score']].copy()
        
        # Record original means and stds
        original_age_mean = data['age'].mean()
        original_age_std = data['age'].std()
        original_kps_mean = data['karnofsky_score'].mean()
        original_kps_std = data['karnofsky_score'].std()
        
        scaled_data = processor._scale_numerical_features(data)
        
        # Check that data shape is preserved
        assert scaled_data.shape == data.shape
        
        # Check that means are close to 0 and stds close to 1 after scaling
        assert abs(scaled_data['age'].mean()) < 1e-10
        assert abs(scaled_data['karnofsky_score'].mean()) < 1e-10
        assert abs(scaled_data['age'].std() - 1.0) < 1e-10
        assert abs(scaled_data['karnofsky_score'].std() - 1.0) < 1e-10
        
        # Check that scaler was created
        assert 'numerical' in processor.numerical_scalers
    
    def test_engineer_features(self, synthetic_clinical_data):
        """Test feature engineering."""
        processor = ClinicalDataProcessor()
        
        # Focus on a subset of columns
        data = synthetic_clinical_data[['patient_id', 'age', 'karnofsky_score']].copy()
        
        engineered_data = processor._engineer_features(data)
        
        # Check that new features were created
        assert 'age_group_young' in engineered_data.columns
        assert 'age_group_middle' in engineered_data.columns
        assert 'age_group_elder' in engineered_data.columns
        assert 'high_performance' in engineered_data.columns
        
        # Check binary nature of engineered features
        assert set(engineered_data['age_group_young'].unique()).issubset({0, 1})
        assert set(engineered_data['age_group_middle'].unique()).issubset({0, 1})
        assert set(engineered_data['age_group_elder'].unique()).issubset({0, 1})
        assert set(engineered_data['high_performance'].unique()).issubset({0, 1})
    
    def test_preprocess_clinical_data_end_to_end(self, synthetic_clinical_data):
        """Test the entire preprocessing pipeline."""
        processor = ClinicalDataProcessor()
        result = processor.preprocess_clinical_data(synthetic_clinical_data)
        
        # Check that all original categorical columns are removed
        assert 'sex' not in result.columns
        assert 'grade' not in result.columns
        assert 'histology' not in result.columns
        assert 'location' not in result.columns
        
        # Check that engineered features are present
        assert any(col.startswith('age_group_') for col in result.columns)
        assert 'high_performance' in result.columns
        
        # Check one-hot encoded columns exist
        assert any(col.startswith('sex_') for col in result.columns)
        assert any(col.startswith('grade_') for col in result.columns)
        assert any(col.startswith('histology_') for col in result.columns)
        assert any(col.startswith('location_') for col in result.columns)
    
    def test_process_treatment_data(self, synthetic_treatment_data):
        """Test treatment data processing."""
        processor = ClinicalDataProcessor()
        processed = processor.process_treatment_data(synthetic_treatment_data)
        
        # Check one-hot encoding of treatment category
        assert any(col.startswith('category_') for col in processed.columns)
        
        # If we have both dose and duration_days, we should have cumulative_dose
        if 'dose' in processed.columns and 'duration_days' in processed.columns:
            has_dose_and_duration = (~processed['dose'].isna() & ~processed['duration_days'].isna())
            if has_dose_and_duration.any():
                assert 'cumulative_dose' in processed.columns
    
    def test_integrate_multimodal_data(self, synthetic_clinical_data, synthetic_imaging_features):
        """Test multimodal data integration."""
        processor = ClinicalDataProcessor()
        
        # Use a subset for testing
        clinical_subset = synthetic_clinical_data.iloc[:5].copy()
        
        integrated = processor.integrate_multimodal_data(clinical_subset, synthetic_imaging_features)
        
        # Check that imaging features were added
        assert any(col.startswith('t1c_') for col in integrated.columns)
        assert any(col.startswith('t2w_') for col in integrated.columns)
        assert any(col.startswith('tumor_') for col in integrated.columns)
        
        # Check that original clinical data is preserved
        for col in clinical_subset.columns:
            assert col in integrated.columns
        
        # Check row count is preserved
        assert len(integrated) == len(clinical_subset)


class TestEfficientMRIProcessor:
    """Test suite for the EfficientMRIProcessor class."""
    
    def test_initialization(self):
        """Test initialization of EfficientMRIProcessor."""
        processor = EfficientMRIProcessor(memory_limit_mb=2000)
        assert processor.memory_limit_mb == 2000
        assert hasattr(processor, 'memory_tracker')
    
    @patch('susruta.data.mri.sitk')
    def test_load_nifti_metadata(self, mock_sitk):
        """Test loading of NIfTI metadata."""
        # Setup mock
        mock_reader = MagicMock()
        mock_sitk.ImageFileReader.return_value = mock_reader
        mock_reader.GetSize.return_value = (256, 256, 155)
        mock_reader.GetSpacing.return_value = (1.0, 1.0, 1.0)
        mock_reader.GetOrigin.return_value = (0.0, 0.0, 0.0)
        mock_reader.GetDirection.return_value = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        
        processor = EfficientMRIProcessor()
        metadata = processor.load_nifti_metadata('dummy_path.nii.gz')
        
        # Check that ImageFileReader was called
        mock_sitk.ImageFileReader.assert_called_once()
        mock_reader.SetFileName.assert_called_once_with('dummy_path.nii.gz')
        mock_reader.ReadImageInformation.assert_called_once()
        
        # Check returned metadata
        assert 'size' in metadata
        assert 'spacing' in metadata
        assert 'origin' in metadata
        assert 'direction' in metadata
        assert metadata['size'] == (256, 256, 155)
    
    @patch('susruta.data.mri.sitk')
    def test_compute_bounding_box(self, mock_sitk):
        """Test bounding box computation."""
        # Setup mock
        mock_mask = MagicMock()
        mock_binary_mask = MagicMock()
        mock_stats = MagicMock()
        
        mock_sitk.ReadImage.return_value = mock_mask
        mock_mask.GetPixelID.return_value = 1  # Not sitkUInt8
        mock_sitk.sitkUInt8 = 2  # Different from mask.GetPixelID()
        mock_mask > 0
        mock_sitk.LabelStatisticsImageFilter.return_value = mock_stats
        mock_stats.HasLabel.return_value = True
        mock_stats.GetBoundingBox.return_value = (10, 100, 20, 110, 30, 120)
        
        processor = EfficientMRIProcessor()
        bbox_min, bbox_max = processor.compute_bounding_box('dummy_mask.nii.gz')
        
        # Check that the function works as expected
        mock_sitk.ReadImage.assert_called_once_with('dummy_mask.nii.gz')
        mock_stats.HasLabel.assert_called_once_with(1)
        mock_stats.GetBoundingBox.assert_called_once_with(1)
        
        # Check returned bounding box
        assert bbox_min == [10, 20, 30]
        assert bbox_max == [100, 110, 120]
    
    @patch('susruta.data.mri.sitk')
    def test_compute_roi_features(self, mock_sitk):
        """Test ROI feature computation."""
        # Setup mock
        mock_img_array = np.random.rand(10, 10, 10)
        mock_mask_array = np.zeros((10, 10, 10))
        mock_mask_array[3:7, 3:7, 3:7] = 1  # Simple cube mask
        
        mock_sitk.GetArrayFromImage.side_effect = [mock_img_array, mock_mask_array]
        
        processor = EfficientMRIProcessor()
        features = processor._compute_roi_features(MagicMock(), MagicMock())
        
        # Check that arrays were retrieved
        assert mock_sitk.GetArrayFromImage.call_count == 2
        
        # Check returned features
        assert 'mean' in features
        assert 'std' in features
        assert 'min' in features
        assert 'max' in features
        assert 'p25' in features
        assert 'p50' in features
        assert 'p75' in features
        assert 'volume_voxels' in features
    
    def test_combine_chunk_features(self):
        """Test combining features from multiple chunks."""
        processor = EfficientMRIProcessor()
        
        # Create test chunks
        chunks = [
            {
                'mean': 10.0,
                'std': 2.0,
                'min': 5.0,
                'max': 15.0,
                'p25': 8.0,
                'p50': 10.0,
                'p75': 12.0,
                'volume_voxels': 100
            },
            {
                'mean': 20.0,
                'std': 3.0,
                'min': 10.0,
                'max': 25.0,
                'p25': 18.0,
                'p50': 20.0,
                'p75': 22.0,
                'volume_voxels': 200
            }
        ]
        
        combined = processor._combine_chunk_features(chunks)
        
        # Check combined features
        assert combined['volume_voxels'] == 300
        assert combined['min'] == 5.0
        assert combined['max'] == 25.0
        
        # Check weighted averaging
        assert combined['mean'] == (10.0*100 + 20.0*200) / 300
        assert combined['p25'] == (8.0*100 + 18.0*200) / 300
        assert combined['p50'] == (10.0*100 + 20.0*200) / 300
        assert combined['p75'] == (12.0*100 + 22.0*200) / 300
        
        # Check std calculation (approximate)
        expected_std = np.sqrt((2.0**2*100 + 3.0**2*200) / 300)
        assert abs(combined['std'] - expected_std) < 1e-10