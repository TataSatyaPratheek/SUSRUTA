#!/usr/bin/env python
# scripts/extract_mri_features.py
"""
Script for Phase 1: MRI Feature Extraction and Persistence (Parallelized).

Scans a base directory for patient MRI data (.nii.gz files),
extracts features using EfficientMRIProcessor in parallel across timepoints,
and saves them to a structured HDF5 file.
"""

import os
import re
import argparse
import logging
import h5py
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm # For progress bar
import time
import concurrent.futures # For parallelization
import traceback # For detailed error logging in workers

# --- Color definitions for terminal output ---
class TermColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def color_text(text, color):
    """Applies ANSI color codes to text."""
    # Disable color if not a TTY (e.g., redirecting output to file)
    if sys.stdout.isatty():
        return f"{color}{text}{TermColors.ENDC}"
    else:
        return text

# Add susruta to path if not installed
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    print(color_text(f"Adding project root to PYTHONPATH: {project_root}", TermColors.OKBLUE))
    sys.path.insert(0, str(project_root))

try:
    from susruta.data.mri import EfficientMRIProcessor
    # MemoryTracker might be less useful across processes, but keep if needed
    # from susruta.utils import MemoryTracker
except ImportError as e:
    print(color_text(f"Error importing susruta modules: {e}", TermColors.FAIL))
    print(color_text("Ensure the susruta package is installed or the project root is in PYTHONPATH.", TermColors.FAIL))
    sys.exit(1)

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
PATIENT_ID_PATTERN = re.compile(r"PatientID_(\d{4})")
TIMEPOINT_PATTERN = re.compile(r"Timepoint_(\d+)")
DEFAULT_SEQUENCES = ['t1c', 't1n', 't2f', 't2w', 'tumorMask']

# --- Helper Functions (Unchanged from original) ---

def find_patients(base_dir: Path, specific_ids: list[int] = None) -> list[int]:
    """Scans the base directory for patient folders."""
    patient_ids = []
    print(color_text(f"Scanning for patients in: {base_dir}", TermColors.OKBLUE))
    if not base_dir.is_dir():
        print(color_text(f"Error: Base directory not found: {base_dir}", TermColors.FAIL))
        return []

    for item in base_dir.iterdir():
        if item.is_dir():
            match = PATIENT_ID_PATTERN.match(item.name)
            if match:
                patient_id = int(match.group(1))
                if specific_ids is None or patient_id in specific_ids:
                    patient_ids.append(patient_id)

    if not patient_ids:
        print(color_text(f"Warning: No patient directories found matching pattern 'PatientID_XXXX' in {base_dir}", TermColors.WARNING))
    else:
        found_ids_str = ", ".join(map(str, sorted(patient_ids)))
        print(color_text(f"Found {len(patient_ids)} patients.", TermColors.OKGREEN))
        if len(patient_ids) <= 20:
             print(color_text(f"  IDs: {found_ids_str}", TermColors.OKCYAN))
        else:
             print(color_text(f"  IDs: {found_ids_str[:100]}...", TermColors.OKCYAN))

    return sorted(patient_ids)

def find_timepoints(patient_dir: Path, specific_timepoints: list[int] = None) -> list[int]:
    """Scans a patient directory for timepoint folders."""
    timepoints = []
    if not patient_dir.is_dir():
        return []

    for item in patient_dir.iterdir():
        if item.is_dir():
            match = TIMEPOINT_PATTERN.match(item.name)
            if match:
                tp = int(match.group(1))
                if specific_timepoints is None or tp in specific_timepoints:
                    timepoints.append(tp)

    # Simplified: Assume Timepoint_X folders exist based on previous logs
    # if not timepoints and specific_timepoints is None:
    #      # Inferring logic removed for clarity, assuming standard structure
    #      pass

    # Log found timepoints within the main processing loop for better context

    return sorted(timepoints)

# --- Parallel Worker Function ---

def process_single_timepoint(args_tuple):
    """
    Worker function to process features for a single patient timepoint.
    Designed to be used with ProcessPoolExecutor.map or submit.

    Args:
        args_tuple (tuple): A tuple containing:
            (patient_id, timepoint, current_data_dir_str, sequences_to_process, memory_limit_mb)

    Returns:
        tuple: A tuple containing:
            ((patient_id, timepoint), result_dict)
            where result_dict contains extracted features or an error message.
    """
    patient_id, timepoint, current_data_dir_str, sequences_to_process, memory_limit_mb = args_tuple
    current_data_dir = Path(current_data_dir_str) # Recreate Path object in worker

    try:
        # Instantiate processor *within* the worker process
        mri_processor = EfficientMRIProcessor(memory_limit_mb=memory_limit_mb)

        # The core extraction logic (assuming mri.py is corrected)
        # Note: This function now handles finding files internally via glob
        extracted_features = mri_processor.extract_features_for_patient(
            patient_id=patient_id,
            data_dir=str(current_data_dir), # Pass the specific directory string
            timepoint=timepoint,
            sequences=sequences_to_process # Pass the list of sequences
        )

        if not extracted_features:
             # Return None or an empty dict to indicate no features, not necessarily an error
             return ((patient_id, timepoint), {})
        else:
             return ((patient_id, timepoint), extracted_features)

    except Exception as e:
        # Capture errors and return them for reporting in the main process
        error_msg = f"Error in worker for P{patient_id:04d} T{timepoint}: {e}\n{traceback.format_exc()}"
        return ((patient_id, timepoint), {"error": error_msg})


# --- Main Processing Function (Refactored for Parallelism) ---

def extract_and_save_features(
    mri_base_dir: Path,
    output_hdf5_path: Path,
    sequences_to_process: list[str],
    num_workers: int, # Added for parallel control
    patient_ids: list[int] | None = None,
    timepoints: list[int] | None = None,
    memory_limit_mb: int = 4000,
    overwrite: bool = False
):
    """
    Extracts features in parallel and saves results sequentially to HDF5.
    """
    start_time = time.time()
    print(color_text("="*60, TermColors.HEADER))
    print(color_text(" SUSRUTA MRI Feature Extraction Pipeline (Parallel)", TermColors.HEADER + TermColors.BOLD))
    print(color_text("="*60, TermColors.HEADER))

    print(color_text("\nConfiguration:", TermColors.OKBLUE))
    print(f"  MRI Base Directory: {mri_base_dir}")
    print(f"  Output HDF5 File: {output_hdf5_path}")
    print(f"  Sequences to Process: {', '.join(sequences_to_process)}")
    print(f"  Specific Patient IDs: {patient_ids if patient_ids else 'All Found'}")
    print(f"  Specific Timepoints: {timepoints if timepoints else 'All Found'}")
    print(f"  Memory Limit (MB) per worker: {memory_limit_mb}")
    print(f"  Number of Parallel Workers: {num_workers}") # Display worker count
    print(f"  Overwrite Output: {overwrite}\n")

    if num_workers > os.cpu_count():
        print(color_text(f"Warning: Number of workers ({num_workers}) exceeds CPU count ({os.cpu_count()}).", TermColors.WARNING))
    if num_workers > 4 and memory_limit_mb * num_workers > 6000: # Heuristic for 8GB RAM
         print(color_text(f"Warning: High number of workers ({num_workers}) with 8GB RAM might lead to excessive memory usage and slow performance. Monitor system resources.", TermColors.WARNING))


    if not mri_base_dir.exists():
        print(color_text(f"Error: MRI Base Directory '{mri_base_dir}' does not exist.", TermColors.FAIL))
        sys.exit(1)

    if output_hdf5_path.exists() and not overwrite:
        print(color_text(f"Error: Output file {output_hdf5_path} already exists. Use --overwrite to replace it.", TermColors.FAIL))
        sys.exit(1)
    elif output_hdf5_path.exists() and overwrite:
        print(color_text(f"Warning: Output file {output_hdf5_path} exists and will be overwritten.", TermColors.WARNING))
        try:
            output_hdf5_path.unlink() # Delete before starting
            print(color_text(f"Removed existing file: {output_hdf5_path}", TermColors.OKBLUE))
        except OSError as e:
            print(color_text(f"Error removing existing file {output_hdf5_path}: {e}", TermColors.FAIL))
            sys.exit(1)


    # --- Task Gathering ---
    print(color_text("Gathering tasks (Patient-Timepoints)...", TermColors.OKBLUE))
    tasks_to_run = []
    all_found_patients = find_patients(mri_base_dir, patient_ids)
    if not all_found_patients:
        print(color_text("No patients found matching criteria. Exiting.", TermColors.FAIL))
        return

    total_timepoints_found = 0
    for patient_id in all_found_patients:
        patient_dir = mri_base_dir / f"PatientID_{patient_id:04d}"
        patient_timepoints = find_timepoints(patient_dir, timepoints)
        if not patient_timepoints:
            print(color_text(f"  Skipping Patient {patient_id:04d}: No timepoints found/specified.", TermColors.WARNING))
            continue

        print(color_text(f"  Patient {patient_id:04d}: Found timepoints {patient_timepoints}", TermColors.OKCYAN))
        total_timepoints_found += len(patient_timepoints)

        for timepoint in patient_timepoints:
            timepoint_dir_path = patient_dir / f"Timepoint_{timepoint}"
            # Determine the correct data_dir for the processor
            current_data_dir = timepoint_dir_path if timepoint_dir_path.is_dir() else patient_dir
            # Add task arguments as a tuple
            tasks_to_run.append((
                patient_id,
                timepoint,
                str(current_data_dir), # Pass path as string for pickling
                sequences_to_process,
                memory_limit_mb
            ))

    if not tasks_to_run:
        print(color_text("No patient timepoints to process after filtering. Exiting.", TermColors.FAIL))
        return

    print(color_text(f"\nFound {total_timepoints_found} total patient-timepoints to process.", TermColors.OKGREEN))
    print(color_text(f"Starting parallel feature extraction with {num_workers} workers...", TermColors.OKBLUE))

    # --- Parallel Execution ---
    results = {} # Dictionary to store results: {(patient_id, timepoint): features_dict}
    processed_count = 0
    error_count = 0
    futures = []

    # Use ProcessPoolExecutor for CPU-bound tasks
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        for task_args in tasks_to_run:
            futures.append(executor.submit(process_single_timepoint, task_args))

        # Process results as they complete with tqdm progress bar
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(tasks_to_run), desc=color_text("Extracting Features", TermColors.BOLD), unit="timepoint"):
            try:
                # Get the result from the future
                (pt_id, tp), result_data = future.result()

                if isinstance(result_data, dict) and "error" in result_data:
                    # Log the error message from the worker
                    print(color_text(f"\nError reported for P{pt_id:04d} T{tp}:\n{result_data['error']}", TermColors.FAIL))
                    error_count += 1
                    results[(pt_id, tp)] = result_data # Store error marker
                elif isinstance(result_data, dict) and not result_data:
                     # No features extracted, but not an error (e.g., mask empty)
                     print(color_text(f"  No features extracted for P{pt_id:04d} T{tp} (handled by worker).", TermColors.WARNING))
                     results[(pt_id, tp)] = {} # Store empty dict
                     # Consider if this should count as processed or skipped
                elif isinstance(result_data, dict):
                    results[(pt_id, tp)] = result_data # Store successful results
                    processed_count += 1
                else:
                     # Should not happen if worker returns correctly
                     print(color_text(f"\nUnexpected result type for P{pt_id:04d} T{tp}: {type(result_data)}", TermColors.FAIL))
                     error_count += 1
                     results[(pt_id, tp)] = {"error": "Unexpected result type from worker"}


            except Exception as exc:
                # Catch errors during future.result() itself (less likely)
                print(color_text(f"\nException retrieving result: {exc}", TermColors.FAIL))
                # Try to associate error with a task if possible (difficult here)
                error_count += 1

    print(color_text(f"\nParallel extraction finished. Processed={processed_count}, Errors={error_count}.", TermColors.OKBLUE))

    # --- Sequential HDF5 Writing ---
    print(color_text(f"Writing {len(results)} results to HDF5 file: {output_hdf5_path}...", TermColors.OKBLUE))
    feature_save_count_total = 0
    patients_written = set()

    if not results:
        print(color_text("No results collected to write to HDF5.", TermColors.WARNING))
    else:
        try:
            with h5py.File(output_hdf5_path, 'w') as f_out: # Open in write mode ('w')
                # Add metadata
                f_out.attrs['creation_date'] = np.bytes_(np.datetime64('now', 's'))
                f_out.attrs['susruta_version'] = np.bytes_("0.1.1-parallel") # Indicate parallel version
                f_out.attrs['processed_sequences'] = np.bytes_(",".join(sequences_to_process))
                f_out.attrs['source_directory'] = np.bytes_(str(mri_base_dir))
                f_out.attrs['memory_limit_mb_per_worker'] = memory_limit_mb
                f_out.attrs['num_workers_used'] = num_workers

                # Sort results by patient ID, then timepoint for structured writing
                sorted_keys = sorted(results.keys())

                for patient_id, timepoint in tqdm(sorted_keys, desc=color_text("Saving to HDF5", TermColors.BOLD), unit="timepoint"):
                    extracted_features = results[(patient_id, timepoint)]

                    # Get or create patient group
                    patient_group_name = f"PatientID_{patient_id:04d}"
                    if patient_group_name not in f_out:
                        patient_group = f_out.create_group(patient_group_name)
                        if patient_id not in patients_written:
                             # print(color_text(f"  Created HDF5 group: '{patient_group_name}'", TermColors.OKGREEN)) # Too verbose
                             patients_written.add(patient_id)
                    else:
                        patient_group = f_out[patient_group_name]

                    # Get or create timepoint group
                    tp_group_name = f"Timepoint_{timepoint}"
                    if tp_group_name not in patient_group:
                        tp_group = patient_group.create_group(tp_group_name)
                    else:
                        tp_group = patient_group[tp_group_name]


                    # Handle cases where extraction failed or yielded no features
                    if "error" in extracted_features:
                        # Optionally save error message to HDF5
                        try:
                             error_str = extracted_features["error"]
                             # Truncate long tracebacks if necessary
                             if len(error_str) > 2048: error_str = error_str[:2045] + "..."
                             tp_group.attrs['extraction_error'] = np.bytes_(error_str)
                             # print(color_text(f"    Recorded error for P{patient_id:04d} T{timepoint} in HDF5.", TermColors.WARNING))
                        except Exception as attr_err:
                             print(color_text(f"    Could not save error attribute for P{patient_id:04d} T{timepoint}: {attr_err}", TermColors.FAIL))
                        continue # Skip saving features if there was an error

                    if not extracted_features:
                         # print(color_text(f"    No features to save for P{patient_id:04d} T{timepoint}.", TermColors.OKCYAN))
                         tp_group.attrs['extraction_status'] = np.bytes_("No features extracted")
                         continue


                    # Save features for each sequence within the timepoint group
                    feature_save_count_tp = 0
                    for sequence_name, features in extracted_features.items():
                        if not features:
                            # print(color_text(f"      No features for sequence '{sequence_name}', skipping save.", TermColors.OKCYAN))
                            continue

                        if sequence_name in tp_group:
                             seq_group = tp_group[sequence_name]
                        else:
                             seq_group = tp_group.create_group(sequence_name)

                        # print(color_text(f"      Saving {len(features)} features for sequence '{sequence_name}'...", TermColors.OKCYAN))
                        for feature_name, value in features.items():
                            try:
                                if value is None: continue # Skip None
                                if feature_name in seq_group: del seq_group[feature_name]
                                seq_group.create_dataset(feature_name, data=value)
                                feature_save_count_tp += 1
                            except TypeError as h5_err:
                                 print(color_text(f"        Warning: Could not save P{patient_id:04d} T{timepoint} Seq '{sequence_name}' Feat '{feature_name}' (type: {type(value)}): {h5_err}. Skipping.", TermColors.WARNING))
                    feature_save_count_total += feature_save_count_tp
                    # print(color_text(f"      Saved {feature_save_count_tp} features for T{timepoint}.", TermColors.OKGREEN))


            print(color_text(f"\nSuccessfully saved {feature_save_count_total} features to {output_hdf5_path}", TermColors.OKGREEN))

        except Exception as e:
            print(color_text(f"\nCritical Error: Failed during HDF5 file writing: {e}", TermColors.FAIL))
            logging.error(f"Full traceback for HDF5 file writing:", exc_info=True)
            # File might be corrupt or partially written
            return # Exit after critical error

    # --- Final Summary ---
    total_duration = time.time() - start_time
    print(color_text("\n" + "="*60, TermColors.HEADER))
    print(color_text(" Feature Extraction Summary", TermColors.HEADER + TermColors.BOLD))
    print(color_text("="*60, TermColors.HEADER))
    print(color_text(f"  Successfully processed: {processed_count} patient-timepoints", TermColors.OKGREEN))
    skipped_tp_count = total_timepoints_found - processed_count - error_count
    if skipped_tp_count > 0:
        print(color_text(f"  Skipped/No Features: {skipped_tp_count} patient-timepoints", TermColors.WARNING))
    if error_count > 0:
        print(color_text(f"  Encountered errors: {error_count} patient-timepoints", TermColors.FAIL))
    else:
        print(color_text(f"  Encountered errors: 0", TermColors.OKGREEN))
    print(color_text(f"  Total features saved: {feature_save_count_total}", TermColors.OKCYAN))
    print(color_text(f"  Features saved to: {output_hdf5_path}", TermColors.OKCYAN))
    print(color_text(f"  Total execution time: {total_duration:.2f} seconds", TermColors.OKBLUE))
    print(color_text("="*60, TermColors.HEADER))


# --- Command Line Interface (Added --num-workers) ---

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='SUSRUTA MRI Feature Extraction Pipeline (Parallel)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Default number of workers - conservative for 8GB RAM
    default_workers = max(1, min(4, os.cpu_count() // 2 if os.cpu_count() else 4))

    parser.add_argument('--mri-base-dir', type=str, required=True,
                        help='Base directory containing patient MRI data (e.g., ./PKG-MU-Glioma-Post/MU-Glioma-Post/)')
    parser.add_argument('--output-file', type=str, required=True,
                        help='Path to save the output HDF5 file (e.g., ./output/mri_features.hdf5)')
    parser.add_argument('--patient-ids', type=int, nargs='+', default=None,
                        help='Optional list of specific Patient IDs (integers) to process (e.g., 3 10 25)')
    parser.add_argument('--timepoints', type=int, nargs='+', default=None,
                        help='Optional list of specific Timepoints (integers) to process for each patient (e.g., 1 2)')
    parser.add_argument('--sequences', type=str, nargs='+', default=DEFAULT_SEQUENCES,
                        help=f'MRI sequences to process')
    parser.add_argument('--memory-limit', type=int, default=4000,
                        help='Approx. memory limit in MB *per worker* for EfficientMRIProcessor')
    parser.add_argument('--num-workers', type=int, default=default_workers,
                        help='Number of parallel processes to use for feature extraction.')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite the output file if it already exists.')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable debug logging (more detailed output).')


    return parser.parse_args()

def main():
    """Main execution function."""
    args = parse_arguments()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        print(color_text("Verbose logging enabled.", TermColors.OKBLUE))

    mri_base_path = Path(args.mri_base_dir).resolve()
    output_path = Path(args.output_file).resolve()

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(color_text(f"Ensured output directory exists: {output_path.parent}", TermColors.OKGREEN))
    except OSError as e:
        print(color_text(f"Error creating output directory {output_path.parent}: {e}", TermColors.FAIL))
        sys.exit(1)

    extract_and_save_features(
        mri_base_dir=mri_base_path,
        output_hdf5_path=output_path,
        sequences_to_process=args.sequences,
        num_workers=args.num_workers, # Pass num_workers
        patient_ids=args.patient_ids,
        timepoints=args.timepoints,
        memory_limit_mb=args.memory_limit,
        overwrite=args.overwrite
    )

if __name__ == "__main__":
    main()
