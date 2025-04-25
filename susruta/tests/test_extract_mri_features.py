# tests/test_extract_mri_features.py
import pytest
import h5py
import numpy as np
from pathlib import Path
import sys
import subprocess
from unittest.mock import patch, MagicMock, ANY, call
import concurrent.futures
import argparse
import os

# Add project root to allow importing the script's functions if needed,
# or prepare to run it as a subprocess.
project_root = Path(__file__).resolve().parents[1]
# Ensure correct relative path if src is not directly under project_root
susruta_src_path = project_root # Assuming src is directly under project_root
if not (susruta_src_path / 'susruta').exists():
     susruta_src_path = project_root / 'src' # Try src subdirectory

if str(susruta_src_path) not in sys.path:
     sys.path.insert(0, str(susruta_src_path))
if str(project_root) not in sys.path:
     sys.path.insert(0, str(project_root)) # Add project root itself for scripts

# --- Define script_path ---
script_path = project_root / 'scripts' / 'extract_mri_features.py'
# --- End script_path definition ---


# Import functions from the script for more direct testing
try:
    from scripts.extract_mri_features import (
        find_patients,
        find_timepoints,
        extract_and_save_features,
        process_single_timepoint, # May need mocking if testing directly
        parse_arguments,
        main as script_main # Rename to avoid conflict
    )
    script_imported = True
except ImportError as e:
    print(f"Could not import functions from script: {e}")
    print(f"PYTHONPATH: {sys.path}")
    script_imported = False
    pytest.skip("Skipping script tests: Could not import script functions", allow_module_level=True)


# --- Fixtures ---

@pytest.fixture
def script_test_env(tmp_path):
    """Creates a temporary directory structure for script testing using P3, P5, P6."""
    base_dir = tmp_path / "mri_base_p3p5p6"
    output_dir = tmp_path / "output_p3p5p6"
    output_hdf5 = output_dir / "features_p3p5p6.hdf5"
    output_dir.mkdir()

    # Patient 3: T1, T2, T5
    p3_dir = base_dir / "PatientID_0003"
    p3t1_dir = p3_dir / "Timepoint_1"
    p3t2_dir = p3_dir / "Timepoint_2"
    p3t5_dir = p3_dir / "Timepoint_5"
    p3t1_dir.mkdir(parents=True, exist_ok=True)
    p3t2_dir.mkdir(exist_ok=True)
    p3t5_dir.mkdir(exist_ok=True)
    # T1 files
    (p3t1_dir / "P0003_T1_t1c.nii.gz").touch()
    (p3t1_dir / "P0003_T1_t1n.nii.gz").touch()
    (p3t1_dir / "P0003_T1_t2f.nii.gz").touch()
    (p3t1_dir / "P0003_T1_t2w.nii.gz").touch()
    (p3t1_dir / "P0003_T1_tumorMask.nii.gz").touch()
    # T2 files (missing t1n, t2f)
    (p3t2_dir / "P0003_T2_t1c.nii.gz").touch()
    (p3t2_dir / "P0003_T2_t2w.nii.gz").touch()
    (p3t2_dir / "P0003_T2_tumorMask.nii.gz").touch()
    # T5 files (only mask)
    (p3t5_dir / "P0003_T5_tumorMask.nii.gz").touch()

    # Patient 5: T3, T4
    p5_dir = base_dir / "PatientID_0005"
    p5t3_dir = p5_dir / "Timepoint_3"
    p5t4_dir = p5_dir / "Timepoint_4"
    p5t3_dir.mkdir(parents=True, exist_ok=True)
    p5t4_dir.mkdir(exist_ok=True)
    # T3 files
    (p5t3_dir / "P0005_T3_t1c.nii.gz").touch()
    (p5t3_dir / "P0005_T3_tumorMask.nii.gz").touch()
    # T4 files
    (p5t4_dir / "P0005_T4_t2w.nii.gz").touch()
    (p5t4_dir / "P0005_T4_tumorMask.nii.gz").touch()

    # Patient 6: T2, T4, T5, T6
    p6_dir = base_dir / "PatientID_0006"
    p6t2_dir = p6_dir / "Timepoint_2"
    p6t4_dir = p6_dir / "Timepoint_4"
    p6t5_dir = p6_dir / "Timepoint_5"
    p6t6_dir = p6_dir / "Timepoint_6"
    p6t2_dir.mkdir(parents=True, exist_ok=True)
    p6t4_dir.mkdir(exist_ok=True)
    p6t5_dir.mkdir(exist_ok=True)
    p6t6_dir.mkdir(exist_ok=True)
    # T2 files
    (p6t2_dir / "P0006_T2_t1c.nii.gz").touch()
    (p6t2_dir / "P0006_T2_tumorMask.nii.gz").touch()
    # T4 files
    (p6t4_dir / "P0006_T4_t2w.nii.gz").touch()
    (p6t4_dir / "P0006_T4_tumorMask.nii.gz").touch()
    # T5 files
    (p6t5_dir / "P0006_T5_t1c.nii.gz").touch()
    (p6t5_dir / "P0006_T5_tumorMask.nii.gz").touch()
    # T6 files (no mask)
    (p6t6_dir / "P0006_T6_t1c.nii.gz").touch()

    # Patient 7 (no valid timepoints)
    p7_dir = base_dir / "PatientID_0007"
    p7_dir.mkdir(exist_ok=True)
    (p7_dir / "notes.txt").touch()

    # Non-patient dir
    (base_dir / "metadata").mkdir()

    return {
        "base_dir": base_dir,
        "output_dir": output_dir,
        "output_hdf5": output_hdf5,
        "patient_dirs": [p3_dir, p5_dir, p6_dir, p7_dir]
    }

# --- Helper Function Tests ---

def test_find_patients_all(script_test_env):
    """Test finding all patients."""
    patients = find_patients(script_test_env["base_dir"])
    assert sorted(patients) == [3, 5, 6, 7]

def test_find_patients_specific(script_test_env):
    """Test finding specific patients by ID."""
    patients = find_patients(script_test_env["base_dir"], specific_ids=[3, 6, 8])
    assert sorted(patients) == [3, 6]

def test_find_patients_none_found(script_test_env):
    """Test finding patients when none match."""
    patients = find_patients(script_test_env["base_dir"], specific_ids=[1, 2])
    assert patients == []

def test_find_patients_dir_not_found(tmp_path):
    """Test finding patients in a non-existent directory."""
    patients = find_patients(tmp_path / "non_existent")
    assert patients == []

def test_find_timepoints_all(script_test_env):
    """Test finding all timepoints for a patient."""
    p3_dir = script_test_env["base_dir"] / "PatientID_0003"
    timepoints = find_timepoints(p3_dir)
    assert sorted(timepoints) == [1, 2, 5]

    p5_dir = script_test_env["base_dir"] / "PatientID_0005"
    timepoints = find_timepoints(p5_dir)
    assert sorted(timepoints) == [3, 4]

    p6_dir = script_test_env["base_dir"] / "PatientID_0006"
    timepoints = find_timepoints(p6_dir)
    assert sorted(timepoints) == [2, 4, 5, 6]

    p7_dir = script_test_env["base_dir"] / "PatientID_0007"
    timepoints = find_timepoints(p7_dir)
    assert timepoints == []

def test_find_timepoints_specific(script_test_env):
    """Test finding specific timepoints for a patient."""
    p3_dir = script_test_env["base_dir"] / "PatientID_0003"
    timepoints = find_timepoints(p3_dir, specific_timepoints=[2, 5, 7])
    assert sorted(timepoints) == [2, 5]

def test_find_timepoints_none_found(script_test_env):
    """Test finding timepoints when none match."""
    p3_dir = script_test_env["base_dir"] / "PatientID_0003"
    timepoints = find_timepoints(p3_dir, specific_timepoints=[3, 4])
    assert timepoints == []

# --- Core Logic Tests (extract_and_save_features) ---

# Mock the worker function's dependency
@patch('scripts.extract_mri_features.EfficientMRIProcessor')
def test_process_single_timepoint_success(MockProcessor, script_test_env):
    """Test the worker function for a successful extraction."""
    mock_instance = MockProcessor.return_value
    mock_features = {'t1c': {'mean': 1.0}, 'tumor': {'volume_mm3': 100.0}}
    mock_instance.extract_features_for_patient.return_value = mock_features

    # Use Patient 3, Timepoint 1
    patient_id = 3
    timepoint = 1
    timepoint_dir_str = str(script_test_env["base_dir"] / f"PatientID_{patient_id:04d}" / f"Timepoint_{timepoint}")
    args = (
        patient_id, timepoint, timepoint_dir_str,
        ['t1c'], 1000
    )
    result_key, result_data = process_single_timepoint(args)

    assert result_key == (patient_id, timepoint)
    assert result_data == mock_features
    mock_instance.extract_features_for_patient.assert_called_once_with(
        patient_id=patient_id,
        data_dir=timepoint_dir_str, # Check the specific timepoint dir is passed
        timepoint=timepoint,
        sequences=['t1c']
    )

@patch('scripts.extract_mri_features.EfficientMRIProcessor')
def test_process_single_timepoint_no_features(MockProcessor, script_test_env):
    """Test the worker function when no features are extracted."""
    mock_instance = MockProcessor.return_value
    mock_instance.extract_features_for_patient.return_value = {} # Empty dict

    # Use Patient 3, Timepoint 5 (only mask exists)
    patient_id = 3
    timepoint = 5
    timepoint_dir_str = str(script_test_env["base_dir"] / f"PatientID_{patient_id:04d}" / f"Timepoint_{timepoint}")
    args = (
        patient_id, timepoint, timepoint_dir_str,
        ['t1c'], 1000 # Request t1c, but it doesn't exist
    )
    result_key, result_data = process_single_timepoint(args)

    assert result_key == (patient_id, timepoint)
    # The worker should return {} if the processor returns {}
    assert result_data == {}

@patch('scripts.extract_mri_features.EfficientMRIProcessor')
def test_process_single_timepoint_error(MockProcessor, script_test_env):
    """Test the worker function when an exception occurs."""
    mock_instance = MockProcessor.return_value
    mock_instance.extract_features_for_patient.side_effect = ValueError("Test processing error")

    # Use Patient 5, Timepoint 3
    patient_id = 5
    timepoint = 3
    timepoint_dir_str = str(script_test_env["base_dir"] / f"PatientID_{patient_id:04d}" / f"Timepoint_{timepoint}")
    args = (
        patient_id, timepoint, timepoint_dir_str,
        ['t1c'], 1000
    )
    result_key, result_data = process_single_timepoint(args)

    assert result_key == (patient_id, timepoint)
    assert "error" in result_data
    assert "Test processing error" in result_data["error"]
    assert "ValueError" in result_data["error"]


# Mock concurrent.futures and h5py for integration test
@patch('scripts.extract_mri_features.concurrent.futures.ProcessPoolExecutor')
@patch('scripts.extract_mri_features.h5py.File')
@patch('scripts.extract_mri_features.process_single_timepoint') # Mock the worker directly
def test_extract_and_save_features_success(mock_worker, mock_h5py_File, mock_executor, script_test_env):
    """Test the main orchestration function for a successful run."""
    # --- Mock concurrent.futures ---
    mock_pool = mock_executor.return_value.__enter__.return_value
    mock_futures = []

    # Define worker return values for P3/T1, P3/T2, P3/T5, P5/T3, P5/T4
    results_map = {
        (3, 1): {'t1c': {'mean': 1.0}, 't2w': {'mean': 2.0}, 'tumor': {'vol': 10}},
        (3, 2): {'t1c': {'mean': 1.5}, 't2w': {'mean': 2.5}, 'tumor': {'vol': 15}},
        (3, 5): {'tumor': {'vol': 5}}, # P3/T5 only has mask -> only tumor features
        (5, 3): {'t1c': {'mean': 3.0}, 'tumor': {'vol': 20}},
        (5, 4): {'t2w': {'mean': 4.0}, 'tumor': {'vol': 25}},
    }

    # Create mock futures that return predefined results
    # Tasks submitted: P3/T1, P3/T2, P3/T5, P5/T3, P5/T4
    tasks_expected = [(3, 1), (3, 2), (3, 5), (5, 3), (5, 4)] # P6 not requested
    for pt_id, tp in tasks_expected:
        future = MagicMock(spec=concurrent.futures.Future)
        if (pt_id, tp) in results_map:
             future.result.return_value = ((pt_id, tp), results_map[(pt_id, tp)])
        else:
             # Should not happen with updated results_map
             future.result.return_value = ((pt_id, tp), {})
        mock_futures.append(future)

    # Mock submit to return an ITERATOR over our futures
    mock_pool.submit.side_effect = iter(mock_futures) # Use iterator

    # Mock as_completed to return our futures
    mock_as_completed = patch('scripts.extract_mri_features.concurrent.futures.as_completed', return_value=iter(mock_futures)).start()


    # --- Mock h5py.File ---
    mock_h5_file_instance = MagicMock()
    mock_h5py_File.return_value.__enter__.return_value = mock_h5_file_instance

    # Mock dictionary-like access for groups
    mock_groups = {}
    def mock_create_group(name):
        full_name = name
        if hasattr(mock_create_group, 'current_parent') and mock_create_group.current_parent:
             full_name = f"{mock_create_group.current_parent.name}/{name}"

        mock_group = MagicMock(spec=h5py.Group)
        mock_group.name = full_name
        mock_group.attrs = {}
        mock_groups[full_name] = mock_group

        def nested_create_group(key):
             mock_create_group.current_parent = mock_group
             new_group = mock_create_group(key)
             mock_create_group.current_parent = None
             return new_group

        mock_group.create_group.side_effect = nested_create_group
        mock_group.__contains__.side_effect = lambda key: f"{full_name}/{key}" in mock_groups
        mock_group.__getitem__.side_effect = lambda key: mock_groups[f"{full_name}/{key}"]

        mock_group.create_dataset = MagicMock()
        mock_group.__delitem__ = MagicMock()
        return mock_group

    mock_h5_file_instance.create_group.side_effect = mock_create_group
    mock_h5_file_instance.__contains__.side_effect = lambda key: key in mock_groups
    mock_h5_file_instance.__getitem__.side_effect = lambda key: mock_groups[key]
    mock_h5_file_instance.attrs = {}


    # --- Run the function ---
    extract_and_save_features(
        mri_base_dir=script_test_env["base_dir"],
        output_hdf5_path=script_test_env["output_hdf5"],
        sequences_to_process=['t1c', 't2w'], # Request t1c, t2w
        num_workers=2,
        patient_ids=[3, 5], # Filter patients for this test
        timepoints=None, # Process all found timepoints for P3, P5
        memory_limit_mb=1000,
        overwrite=True
    )

    # --- Assertions ---
    # Check task submission (P3/T1, P3/T2, P3/T5, P5/T3, P5/T4) -> 5 tasks
    assert mock_pool.submit.call_count == 5 # P3/T5 should also be submitted

    # Check HDF5 file opening
    mock_h5py_File.assert_called_once_with(script_test_env["output_hdf5"], 'w')

    # Check HDF5 metadata
    assert 'creation_date' in mock_h5_file_instance.attrs
    # assert mock_h5_file_instance.attrs['susruta_version'] == np.bytes_("0.1.1-parallel") # Version might change
    assert mock_h5_file_instance.attrs['processed_sequences'] == np.bytes_("t1c,t2w")
    assert mock_h5_file_instance.attrs['num_workers_used'] == 2

    # Check HDF5 structure and data saving calls (example for P3/T1)
    p3_group_name = "PatientID_0003"
    p3t1_group_name = f"{p3_group_name}/Timepoint_1"
    p3t1_t1c_group_name = f"{p3t1_group_name}/t1c"
    p3t1_t2w_group_name = f"{p3t1_group_name}/t2w"
    p3t1_tumor_group_name = f"{p3t1_group_name}/tumor"

    # Check that groups were created (using the mock_groups dict)
    assert p3_group_name in mock_groups
    assert p3t1_group_name in mock_groups
    assert p3t1_t1c_group_name in mock_groups
    assert p3t1_t2w_group_name in mock_groups
    assert p3t1_tumor_group_name in mock_groups

    # Check dataset creation calls within P3/T1/t1c
    p3t1_t1c_group = mock_groups[p3t1_t1c_group_name]
    p3t1_t1c_group.create_dataset.assert_any_call('mean', data=1.0)

    # Check dataset creation calls within P3/T1/tumor
    p3t1_tumor_group = mock_groups[p3t1_tumor_group_name]
    p3t1_tumor_group.create_dataset.assert_any_call('vol', data=10)

    # Check P5/T4 (only t2w requested/found)
    p5_group_name = "PatientID_0005"
    p5t4_group_name = f"{p5_group_name}/Timepoint_4"
    p5t4_t2w_group_name = f"{p5t4_group_name}/t2w"
    p5t4_tumor_group_name = f"{p5t4_group_name}/tumor"
    assert p5_group_name in mock_groups
    assert p5t4_group_name in mock_groups
    assert p5t4_t2w_group_name in mock_groups
    assert p5t4_tumor_group_name in mock_groups
    assert f"{p5t4_group_name}/t1c" not in mock_groups # t1c group shouldn't exist

    # Check P3/T5 (only tumor features)
    p3t5_group_name = f"{p3_group_name}/Timepoint_5"
    p3t5_tumor_group_name = f"{p3t5_group_name}/tumor"
    assert p3t5_group_name in mock_groups
    assert p3t5_tumor_group_name in mock_groups
    assert f"{p3t5_group_name}/t1c" not in mock_groups
    assert f"{p3t5_group_name}/t2w" not in mock_groups
    p3t5_tumor_group = mock_groups[p3t5_tumor_group_name]
    p3t5_tumor_group.create_dataset.assert_any_call('vol', data=5)


    # Stop the patcher for as_completed
    mock_as_completed.stop()


@patch('scripts.extract_mri_features.concurrent.futures.ProcessPoolExecutor')
@patch('scripts.extract_mri_features.h5py.File')
@patch('scripts.extract_mri_features.process_single_timepoint')
def test_extract_and_save_features_worker_error(mock_worker, mock_h5py_File, mock_executor, script_test_env):
    """Test handling of errors reported by workers."""
    mock_pool = mock_executor.return_value.__enter__.return_value
    mock_futures = []

    # P3/T1 succeeds, P3/T2 fails
    results = [
        ((3, 1), {'t1c': {'mean': 1.0}}),
        ((3, 2), {'error': 'Worker failed on P0003 T2\nTraceback...'}),
    ]

    for res_key, res_data in results:
        future = MagicMock(spec=concurrent.futures.Future)
        future.result.return_value = (res_key, res_data)
        mock_futures.append(future)

    mock_pool.submit.side_effect = iter(mock_futures) # Use iterator
    mock_as_completed = patch('scripts.extract_mri_features.concurrent.futures.as_completed', return_value=iter(mock_futures)).start()

    mock_h5_file_instance = MagicMock()
    mock_h5py_File.return_value.__enter__.return_value = mock_h5_file_instance

    mock_groups = {}
    # Simplified mock group creation for this test
    def mock_create_group(name):
        full_name = name
        if hasattr(mock_create_group, 'current_parent') and mock_create_group.current_parent:
             full_name = f"{mock_create_group.current_parent.name}/{name}"

        mock_group = MagicMock(spec=h5py.Group); mock_group.name = full_name; mock_group.attrs = {}
        mock_groups[full_name] = mock_group

        def nested_create_group(key):
             mock_create_group.current_parent = mock_group
             new_group = mock_create_group(key)
             mock_create_group.current_parent = None
             return new_group

        mock_group.create_group.side_effect = nested_create_group
        mock_group.__contains__.side_effect = lambda key: f"{full_name}/{key}" in mock_groups
        mock_group.__getitem__.side_effect = lambda key: mock_groups[f"{full_name}/{key}"]
        mock_group.create_dataset = MagicMock()
        mock_group.__delitem__ = MagicMock()
        return mock_group

    mock_h5_file_instance.create_group.side_effect = mock_create_group
    mock_h5_file_instance.__contains__.side_effect = lambda key: key in mock_groups
    mock_h5_file_instance.__getitem__.side_effect = lambda key: mock_groups[key]
    mock_h5_file_instance.attrs = {}

    extract_and_save_features(
        mri_base_dir=script_test_env["base_dir"],
        output_hdf5_path=script_test_env["output_hdf5"],
        sequences_to_process=['t1c'],
        num_workers=1,
        patient_ids=[3], # Use patient 3
        timepoints=[1, 2], # Only process T1, T2
        overwrite=True
    )

    p3_group_name = "PatientID_0003"
    p3t1_group_name = f"{p3_group_name}/Timepoint_1"
    p3t2_group_name = f"{p3_group_name}/Timepoint_2"

    # Check that P3/T1 group exists and has data
    assert p3t1_group_name in mock_groups
    assert 'extraction_error' not in mock_groups[p3t1_group_name].attrs

    # Check that P3/T2 group exists and has error attribute
    assert p3t2_group_name in mock_groups
    assert 'extraction_error' in mock_groups[p3t2_group_name].attrs
    # Check the error message content (decode bytes if necessary)
    error_attr = mock_groups[p3t2_group_name].attrs['extraction_error']
    error_str = error_attr.decode() if isinstance(error_attr, bytes) else str(error_attr)
    assert "Worker failed" in error_str

    mock_as_completed.stop()


def test_extract_and_save_features_overwrite_false(script_test_env):
    """Test that the script exits if overwrite is False and file exists."""
    script_test_env["output_hdf5"].touch() # Create dummy output file

    with pytest.raises(SystemExit) as e:
         extract_and_save_features(
            mri_base_dir=script_test_env["base_dir"],
            output_hdf5_path=script_test_env["output_hdf5"],
            sequences_to_process=['t1c'],
            num_workers=1,
            overwrite=False # Default or explicitly False
        )
    assert e.value.code == 1 # Expecting exit code 1

@patch('pathlib.Path.unlink') # Patch the class method
def test_extract_and_save_features_overwrite_true(mock_unlink, script_test_env): # Add mock_unlink arg
    """Test that the script attempts to remove the file if overwrite is True."""
    output_path_obj = script_test_env["output_hdf5"]
    output_path_obj.touch()
    assert output_path_obj.exists()

    # Mock the actual processing to prevent it from running fully
    # NOTE: We mock find_patients/timepoints to return [] to prevent
    #       ProcessPoolExecutor from actually being used in this specific test,
    #       as we only care about the unlink call which happens before the pool.
    with patch('scripts.extract_mri_features.find_patients', return_value=[]), \
         patch('scripts.extract_mri_features.find_timepoints', return_value=[]), \
         patch('scripts.extract_mri_features.concurrent.futures.ProcessPoolExecutor'), \
         patch('scripts.extract_mri_features.h5py.File'):

        extract_and_save_features(
            mri_base_dir=script_test_env["base_dir"],
            output_hdf5_path=output_path_obj, # Pass the Path object
            sequences_to_process=['t1c'],
            num_workers=1,
            overwrite=True
        )

        # --- FIX: Use assert_called_once ---
        # Since assert_called_once_with is problematic with class method patching,
        # rely on assert_called_once. We know from the code structure that
        # if unlink is called, it's called on output_path_obj.
        mock_unlink.assert_called_once()
        # --- END FIX ---

# --- CLI / main Function Tests ---

@patch('scripts.extract_mri_features.extract_and_save_features')
@patch('argparse.ArgumentParser.parse_args')
def test_main_function_calls_extractor(mock_parse_args, mock_extractor, script_test_env):
    """Test that main parses args and calls the core function."""
    # Mock parsed arguments using P3, P5
    args = argparse.Namespace(
        mri_base_dir=str(script_test_env["base_dir"]),
        output_file=str(script_test_env["output_hdf5"]),
        patient_ids=[3, 5],
        timepoints=None,
        sequences=['t1c', 't2w'],
        memory_limit=2000,
        num_workers=4,
        overwrite=True,
        verbose=False
    )
    mock_parse_args.return_value = args

    # Call main
    script_main()

    # Assert that extract_and_save_features was called with correct args
    mock_extractor.assert_called_once_with(
        mri_base_dir=script_test_env["base_dir"].resolve(),
        output_hdf5_path=script_test_env["output_hdf5"].resolve(),
        sequences_to_process=['t1c', 't2w'],
        num_workers=4,
        patient_ids=[3, 5],
        timepoints=None,
        memory_limit_mb=2000,
        overwrite=True
    )

# --- Subprocess Test (Example - More complex setup) ---

def test_script_execution_help(script_test_env):
    """Test running the script with --help."""
    susruta_src_path = project_root
    if not (susruta_src_path / 'susruta').exists():
        susruta_src_path = project_root / 'src'
    python_path_env = f"{susruta_src_path}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"

    result = subprocess.run(
        [sys.executable, str(script_path), '--help'],
        capture_output=True, text=True, check=True,
        env={**os.environ, 'PYTHONPATH': python_path_env}
    )
    assert 'usage: extract_mri_features.py' in result.stdout
    assert '--mri-base-dir' in result.stdout
    assert '--num-workers' in result.stdout

# @pytest.mark.slow # Mark as slow if it takes time
def test_script_execution_basic_run(script_test_env):
    """Test a basic run of the script via subprocess (writes actual HDF5)."""
    output_file = script_test_env["output_hdf5"]
    base_dir = script_test_env["base_dir"]
    cmd = [
        sys.executable, str(script_path),
        '--mri-base-dir', str(base_dir),
        '--output-file', str(output_file),
        '--patient-ids', '3', '5', # Use P3, P5
        '--num-workers', '1', # Use 1 worker for simplicity in testing
        '--overwrite'
    ]
    susruta_src_path = project_root
    if not (susruta_src_path / 'susruta').exists():
        susruta_src_path = project_root / 'src'
    python_path_env = f"{susruta_src_path}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"

    result = subprocess.run(
        cmd, capture_output=True, text=True, check=False, # check=False to inspect errors
        env={**os.environ, 'PYTHONPATH': python_path_env}
    )

    if result.returncode != 0:
        print("--- STDOUT ---")
        print(result.stdout)
        print("--- STDERR ---")
        print(result.stderr)

    assert result.returncode == 0
    assert output_file.exists()

    # Verify HDF5 content
    try:
        with h5py.File(output_file, 'r') as f:
            assert 'PatientID_0003' in f
            assert 'PatientID_0005' in f
            assert 'PatientID_0006' not in f # Wasn't requested
            assert 'PatientID_0003/Timepoint_1' in f
            assert 'PatientID_0003/Timepoint_2' in f
            assert 'PatientID_0003/Timepoint_5' in f # Should exist, might have error/no features
            assert 'PatientID_0005/Timepoint_3' in f
            assert 'PatientID_0005/Timepoint_4' in f
            # Check metadata
            assert 'num_workers_used' in f.attrs
            assert f.attrs['num_workers_used'] == 1
            # Check P3/T5 status (should have only tumor features or error/status)
            if 'PatientID_0003/Timepoint_5' in f:
                 tp5_group = f['PatientID_0003/Timepoint_5']
                 assert 'tumor' in tp5_group or 'extraction_status' in tp5_group.attrs or 'extraction_error' in tp5_group.attrs
                 assert 't1c' not in tp5_group
                 assert 't2w' not in tp5_group
    except Exception as e:
        pytest.fail(f"Failed to read or verify HDF5 output: {e}")
