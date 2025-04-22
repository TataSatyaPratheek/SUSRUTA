# susruta/tests/test_excel.py
"""Tests for Excel data loading and integration."""

import os
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock, mock_open

from susruta.data import ExcelDataLoader, MultimodalDataIntegrator
from susruta.data import ClinicalDataProcessor
from susruta.utils import MemoryTracker


class TestExcelDataLoader:
    """Test suite for the ExcelDataLoader class."""

    def test_initialization(self):
        """Test initialization of ExcelDataLoader."""
        loader = ExcelDataLoader(memory_limit_mb=1500)
        
        assert loader.memory_limit_mb == 1500
        assert hasattr(loader, 'memory_tracker')
        assert loader._scanner_data_cache is None
        assert loader._clinical_data_cache is None
        assert loader._segmentation_data_cache is None

    @patch('pandas.read_excel')
    def test_load_scanner_data(self, mock_read_excel):
        """Test loading scanner metadata."""
        # Create mock data
        mock_data = pd.DataFrame({
            'PatientID': ['1', '2', '3'],
            'Timepoint': [1, 1, 2],
            'ScannerManufacturer': ['GE', 'Siemens', 'Philips'],
            'ScannerModel': ['Model1', 'Model2', 'Model3'],
            'FieldStrength': [1.5, 3.0, 1.5],
            'SequenceType': ['T1', 'T2', 'FLAIR']
        })
        mock_read_excel.return_value = mock_data
        
        # Create loader and load data
        loader = ExcelDataLoader(memory_limit_mb=1000)
        result = loader.load_scanner_data('dummy_path.xlsx')
        
        # Check that read_excel was called correctly
        mock_read_excel.assert_called_once()
        
        # Check results
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert list(result.columns) == ['PatientID', 'Timepoint', 'ScannerManufacturer', 
                                       'ScannerModel', 'FieldStrength', 'SequenceType']
        
        # Test caching
        result2 = loader.load_scanner_data('dummy_path.xlsx')
        # read_excel should not be called again
        assert mock_read_excel.call_count == 1
        
        # Force reload should call read_excel again
        result3 = loader.load_scanner_data('dummy_path.xlsx', force_reload=True)
        assert mock_read_excel.call_count == 2

    def test_clean_scanner_data(self):
        """Test cleaning and standardizing scanner data."""
        # Create test data
        raw_data = pd.DataFrame({
            'PatientID ': ['P1', 'P2', 'P3'],  # Note space in column name
            'Timepoint': [1, 1, 2],
            'ScannerManufacturer': ['GE', 'Siemens', None],
            'FieldStrength': [1.5, '3.0', np.nan],
            'ExtraCol': ['a', 'b', 'c']
        })
        
        loader = ExcelDataLoader()
        cleaned = loader._clean_scanner_data(raw_data)
        
        # Check column names are standardized
        assert 'PatientID' in cleaned.columns
        
        # Check patient ID is standardized
        assert cleaned['PatientID'].iloc[0] == 'P1'
        
        # Check field strength is converted to float
        assert isinstance(cleaned['FieldStrength'].iloc[0], float)
        assert cleaned['FieldStrength'].iloc[0] == 1.5
        
        # Check string columns are categories
        assert cleaned['ScannerManufacturer'].dtype.name == 'category'
        assert cleaned['ExtraCol'].dtype.name == 'category'

    @patch('pandas.read_excel')
    def test_load_clinical_data(self, mock_read_excel):
        """Test loading clinical data."""
        # Create mock data
        mock_data = pd.DataFrame({
            'patient_id': [1, 2, 3],
            'age': [45, 62, 38],
            'sex': ['M', 'F', 'M'],
            'grade': ['II', 'IV', 'III'],
            'idh_mutation': [1, 0, 1]
        })
        mock_read_excel.return_value = mock_data
        
        # Create loader and load data
        loader = ExcelDataLoader()
        result = loader.load_clinical_data('dummy_path.xlsx')
        
        # Check results
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert 'patient_id' in result.columns
        assert 'grade' in result.columns
        
        # Test caching
        loader.load_clinical_data('dummy_path.xlsx')
        assert mock_read_excel.call_count == 1

    @patch('pandas.read_excel')
    def test_load_segmentation_volumes(self, mock_read_excel):
        """Test loading segmentation volume data."""
        # Create mock data
        mock_data = pd.DataFrame({
            'PatientID': [1, 2, 3],
            'Timepoint': [1, 1, 2],
            'TumorVolume_mm3': [15000.0, 8000.0, 20000.0],
            'EnhancingVolume_mm3': [5000.0, 3000.0, 8000.0],
            'EdemaVolume_mm3': [25000.0, 12000.0, 30000.0]
        })
        mock_read_excel.return_value = mock_data
        
        # Create loader and load data
        loader = ExcelDataLoader()
        result = loader.load_segmentation_volumes('dummy_path.xlsx')
        
        # Check results
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        # Check derived metrics were calculated
        assert 'EnhancingRatio' in result.columns
        assert 'EdemaRatio' in result.columns
        
        # Check ratio calculation is correct
        assert result['EnhancingRatio'].iloc[0] == 5000.0 / 15000.0

    def test_merge_data_sources(self):
        """Test merging multiple data sources."""
        # Create test data
        clinical = pd.DataFrame({
            'patient_id': [1, 2, 3],
            'age': [45, 62, 38],
            'sex': ['M', 'F', 'M']
        })
        
        scanner = pd.DataFrame({
            'PatientID': [1, 2, 4],
            'Timepoint': [1, 1, 1],
            'ScannerManufacturer': ['GE', 'Siemens', 'Philips']
        })
        
        segmentation = pd.DataFrame({
            'PatientID': [1, 2, 5],
            'Timepoint': [1, 1, 1],
            'TumorVolume_mm3': [15000.0, 8000.0, 20000.0]
        })
        
        loader = ExcelDataLoader()
        result = loader.merge_data_sources(clinical, scanner, segmentation)
        
        # Check results
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3  # Only patients in clinical data
        assert 'scanner_scannermanufacturer' in result.columns
        assert 'seg_tumorvolume_mm3' in result.columns
        
        # Check correct merging (Use string comparison for patient_id)
        # --- START FIX ---
        assert result.loc[result['patient_id'] == '1', 'scanner_scannermanufacturer'].iloc[0] == 'GE'
        assert result.loc[result['patient_id'] == '1', 'seg_tumorvolume_mm3'].iloc[0] == 15000.0

        # Patient 3 has no scanner or segmentation data
        assert pd.isna(result.loc[result['patient_id'] == '3', 'scanner_scannermanufacturer'].iloc[0])
        assert pd.isna(result.loc[result['patient_id'] == '3', 'seg_tumorvolume_mm3'].iloc[0])
        # --- END FIX ---


    def test_process_for_graph(self):
        """Test graph preparation processing."""
        # Create merged data
        merged_data = pd.DataFrame({
            'patient_id': [1, 2, 3],
            'age': [45, 62, np.nan],
            'karnofsky_score': [90, 70, 80],
            'grade': ['II', 'IV', 'III'],
            'idh_mutation': [1, 0, 1],
            'mgmt_methylation': [1, 0, np.nan],
            'seg_tumorvolume_mm3': [15000.0, 8000.0, np.nan],
            'seg_enhancingvolume_mm3': [5000.0, 3000.0, np.nan],
            'seg_edemavolume_mm3': [25000.0, 12000.0, np.nan]
        })
        
        loader = ExcelDataLoader()
        result = loader.process_for_graph(merged_data)
        
        # Check results
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        
        # Check missing values are handled
        assert not pd.isna(result['age'].iloc[2])  # Should be imputed
        assert not pd.isna(result['mgmt_methylation'].iloc[2])  # Should be imputed
        
        # Check derived features
        assert 'age_group' in result.columns
        assert 'risk_score' in result.columns
        assert 'risk_category' in result.columns
        
        # Check risk scoring
        # Higher grade (IV) and negative molecular markers should have higher risk
        assert result.loc[result['patient_id'] == 2, 'risk_score'].iloc[0] > \
               result.loc[result['patient_id'] == 1, 'risk_score'].iloc[0]


class TestMultimodalDataIntegrator:
    """Test suite for the MultimodalDataIntegrator class."""

    def test_initialization(self):
        """Test initialization of MultimodalDataIntegrator."""
        integrator = MultimodalDataIntegrator(memory_limit_mb=3000)
        
        assert integrator.memory_limit_mb == 3000
        assert hasattr(integrator, 'memory_tracker')
        assert hasattr(integrator, 'excel_loader')
        assert hasattr(integrator, 'clinical_processor')
        assert integrator._integrated_data_cache is None

    @patch.object(ExcelDataLoader, 'load_scanner_data')
    @patch.object(ExcelDataLoader, 'load_clinical_data')
    @patch.object(ExcelDataLoader, 'load_segmentation_volumes')
    def test_load_all_excel_data(self, mock_load_seg, mock_load_clinical, mock_load_scanner):
        """Test loading all Excel data sources."""
        # Create mock data
        mock_scanner_data = pd.DataFrame({'PatientID': [1, 2, 3]})
        mock_clinical_data = pd.DataFrame({'patient_id': [1, 2, 3]})
        mock_seg_data = pd.DataFrame({'PatientID': [1, 2, 3]})
        
        mock_load_scanner.return_value = mock_scanner_data
        mock_load_clinical.return_value = mock_clinical_data
        mock_load_seg_data = mock_seg_data
        
        # Patch os.path.exists to return True
        with patch('os.path.exists', return_value=True):
            integrator = MultimodalDataIntegrator()
            result = integrator.load_all_excel_data(
                scanner_path='scanner.xlsx',
                clinical_path='clinical.xlsx',
                segmentation_path='segmentation.xlsx'
            )
        
        # Check results
        assert isinstance(result, dict)
        assert 'scanner' in result
        assert 'clinical' in result
        assert 'segmentation' in result
        
        # Check each loader was called with correct path
        mock_load_scanner.assert_called_once_with('scanner.xlsx', force_reload=False)
        mock_load_clinical.assert_called_once_with('clinical.xlsx', force_reload=False)
        mock_load_seg.assert_called_once_with('segmentation.xlsx', force_reload=False)

    def test_merge_clinical_sources(self):
        """Test merging primary and Excel clinical data."""
        # Create primary clinical data
        primary = pd.DataFrame({
            'patient_id': [1, 2, 3],
            'age': [45, 62, np.nan],
            'karnofsky_score': [90, 70, 80],
            'unique_primary': ['a', 'b', 'c']
        })
        
        # Create Excel clinical data
        excel = pd.DataFrame({
            'patient_id': [1, 3, 4],
            'age': [46, 39, 55],
            'grade': ['II', 'III', 'IV'],
            'unique_excel': ['x', 'y', 'z']
        })
        
        integrator = MultimodalDataIntegrator()
        result = integrator._merge_clinical_sources(primary, excel)
        
        # Check results
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 4  # Should include all patients from both sources
        
        # Check unique columns from both sources
        assert 'unique_primary' in result.columns
        assert 'unique_excel' in result.columns
        
        # Check filling NaNs from Excel
        assert result.loc[result['patient_id'] == '3', 'age'].iloc[0] == 39  # NEW - Use string '3'
        
        # Check patient 4 (only in Excel)
        assert result.loc[result['patient_id'] == '4', 'age'].iloc[0] == 55  # NEW - Use string '4'
        assert pd.isna(result.loc[result['patient_id'] == '4', 'karnofsky_score'].iloc[0]) # NEW - Use string '4'


    @patch.object(ClinicalDataProcessor, 'preprocess_clinical_data')
    def test_prepare_for_graph_construction(self, mock_preprocess):
        """Test preparing data for graph construction."""
        # Create mock integrated data
        integrated_data = pd.DataFrame({
            'patient_id': [1, 2, 3],
            'age': [45, 62, 38],
            'seg_tumorvolume_mm3': [15000.0, 8000.0, 20000.0]
        })
        
        # Create mock image features
        imaging_features = {
            1: {'t1c': {'mean': 150.0}, 'tumor': {'volume_mm3': 14000.0}},
            2: {'t1c': {'mean': 180.0}, 'tumor': {'volume_mm3': 7500.0}},
            3: {'t1c': {'mean': 130.0}, 'tumor': {'volume_mm3': 19000.0}}
        }
        
        # Mock preprocess to return a simple DataFrame
        mock_preprocess.return_value = pd.DataFrame({
            'patient_id': [1, 2, 3],
            'age_processed': [0.5, 0.8, 0.3]
        })
        
        integrator = MultimodalDataIntegrator()
        processed_df, enhanced_imaging = integrator.prepare_for_graph_construction(
            integrated_data, imaging_features
        )
        
        # Check preprocessing was called
        mock_preprocess.assert_called_once()
        
        # Check enhanced imaging features
        assert enhanced_imaging[1]['segmentation']['tumorvolume_mm3'] == 15000.0
        assert enhanced_imaging[2]['segmentation']['tumorvolume_mm3'] == 8000.0
        assert enhanced_imaging[3]['segmentation']['tumorvolume_mm3'] == 20000.0
        
        # Original features should be preserved
        assert enhanced_imaging[1]['t1c']['mean'] == 150.0
        assert enhanced_imaging[2]['tumor']['volume_mm3'] == 7500.0

    def test_enhance_imaging_features(self):
        """Test enhancing imaging features with segmentation data."""
        # Create imaging features
        imaging_features = {
            1: {'t1c': {'mean': 150.0}, 'tumor': {'volume_mm3': 14000.0}},
            2: {'t1c': {'mean': 180.0}, 'tumor': {'volume_mm3': 7500.0}}
        }
        
        # Create integrated data with segmentation
        integrated_data = pd.DataFrame({
            'patient_id': [1, 2, 3],
            'seg_tumorvolume_mm3': [15000.0, 8000.0, 20000.0],
            'seg_enhancingvolume_mm3': [5000.0, 3000.0, 8000.0],
            'seg_edema_ratio': [1.5, 1.2, 1.8]
        })
        
        integrator = MultimodalDataIntegrator()
        result = integrator._enhance_imaging_features(imaging_features, integrated_data)
        
        # Check results
        assert isinstance(result, dict)
        assert result != imaging_features  # Should be a different object
        
        # Check segmentation features were added
        assert 'segmentation' in result[1]
        assert result[1]['segmentation']['tumorvolume_mm3'] == 15000.0
        assert result[1]['segmentation']['enhancingvolume_mm3'] == 5000.0
        assert result[1]['segmentation']['edema_ratio'] == 1.5
        
        # Check original features are preserved
        assert result[1]['t1c']['mean'] == 150.0
        assert result[2]['tumor']['volume_mm3'] == 7500.0
        
        # Patient 3 is not in original imaging_features
        assert 3 not in result

    def test_generate_patient_summaries(self):
        """Test generating patient summaries from integrated data."""
        # Create integrated data
        integrated_data = pd.DataFrame({
            'patient_id': [1, 2, 3],
            'age': [45, 62, 38],
            'sex': ['M', 'F', 'M'],
            'karnofsky_score': [90, 70, 80],
            'grade': ['II', 'IV', 'III'],
            'idh_mutation': [1, 0, 1],
            'mgmt_methylation': [1, 0, 0],
            'seg_tumorvolume_mm3': [15000.0, 8000.0, 20000.0],
            'seg_enhancingvolume_mm3': [5000.0, 3000.0, 8000.0],
            'scanner_manufacturer': ['GE', 'Siemens', 'Philips'],
            'scanner_fieldstrength': [1.5, 3.0, 1.5],
            'risk_score': [0.3, 0.8, 0.5],
            'risk_category': ['low', 'high', 'medium']
        })
        
        integrator = MultimodalDataIntegrator()
        result = integrator.generate_patient_summaries(integrated_data)
        
        # Check results
        assert isinstance(result, dict)
        assert len(result) == 3
        assert 1 in result
        
        # Check patient 1 summary structure
        patient1 = result[1]
        assert 'demographics' in patient1
        assert 'tumor' in patient1
        assert 'scanner' in patient1
        assert 'risk_assessment' in patient1
        
        # Check specific values
        assert patient1['demographics']['age'] == 45
        assert patient1['demographics']['sex'] == 'M'
        assert patient1['tumor']['grade'] == 'II'
        assert patient1['tumor']['idh_mutation'] == 1
        assert patient1['tumor']['volumes']['tumorvolume_mm3'] == 15000.0
        assert patient1['scanner']['manufacturer'] == 'GE'
        assert patient1['risk_assessment']['risk_score'] == 0.3
        assert patient1['risk_assessment']['risk_category'] == 'low'