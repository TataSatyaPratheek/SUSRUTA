#!/usr/bin/env python
# scripts/process_excel_data.py
"""
Script for Phase 2: Excel Data Loading, Processing, and Integration.

Loads data from clinical, segmentation, and scanner Excel files for ALL patients
and timepoints, processes them using ClinicalDataProcessor, integrates them,
and saves a SINGLE final processed DataFrame.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import logging
import gc

# --- Setup Project Path ---
project_root = Path(__file__).resolve().parents[1]
src_path = project_root / 'src'
if str(src_path) not in sys.path:
    print(f"Adding src directory to PYTHONPATH: {src_path}")
    sys.path.insert(0, str(src_path))

try:
    from susruta.data.excel_integration import MultimodalDataIntegrator
    from susruta.data.clinical import ClinicalDataProcessor
    from susruta.utils.memory import MemoryTracker
except ImportError as e:
    print(f"Error importing susruta modules: {e}")
    print("Ensure the susruta package is installed or the project structure is correct.")
    sys.exit(1)

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Optional: Color definitions ---
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
    if sys.stdout.isatty():
        return f"{color}{text}{TermColors.ENDC}"
    return text

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=color_text('SUSRUTA Phase 2: Excel Data Processing & Integration (All Timepoints)', TermColors.HEADER),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--scanner-excel', type=str, required=True,
                        help='Path to MR_Scanner_data.xlsx')
    parser.add_argument('--clinical-excel', type=str, required=True,
                        help='Path to MUGliomaPost_ClinicalDataFINAL032025.xlsx')
    parser.add_argument('--segmentation-excel', type=str, required=True,
                        help='Path to MUGliomaPost_Segmentation_Volumes.xlsx')
    parser.add_argument('--output-dir', type=str, default='./output/processed_tabular_data',
                        help='Directory to save the processed data file')
    # --- MODIFIED: Output filename is now fixed, no timepoint ---
    parser.add_argument('--output-filename', type=str, default='integrated_processed_data_all_timepoints',
                        help='Base name for the output file (without extension)')
    # --- END MODIFIED ---
    parser.add_argument('--output-format', type=str, default='feather', choices=['feather', 'csv', 'parquet'],
                        help='Format to save the processed DataFrame (feather is fast).')
    # --- REMOVED: --timepoint argument ---
    parser.add_argument('--memory-limit-mb', type=float, default=6000, # Increased default slightly
                        help='Approximate memory limit in MB for the integration process.')

    return parser.parse_args()

def main():
    """Main execution function for Excel processing."""
    args = parse_arguments()
    memory_tracker = MemoryTracker(threshold_mb=args.memory_limit_mb * 0.9)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(color_text(f"Output directory set to: {output_dir}", TermColors.OKBLUE))

    excel_paths = {
        'scanner': Path(args.scanner_excel),
        'clinical': Path(args.clinical_excel),
        'segmentation': Path(args.segmentation_excel),
    }
    excel_paths = {k: v for k, v in excel_paths.items() if v is not None}

    # --- 1. Initialize Integrator ---
    logger.info(color_text("\n--- Initializing Data Integrator ---", TermColors.HEADER))
    integrator = MultimodalDataIntegrator(memory_limit_mb=args.memory_limit_mb)
    memory_tracker.log_memory("Initialized Integrator")

    # --- 2. Load Data ---
    logger.info(color_text("\n--- Loading Excel Files ---", TermColors.HEADER))
    raw_data_dict = integrator.load_all_excel_data(
        scanner_path=excel_paths.get('scanner'),
        clinical_path=excel_paths.get('clinical'),
        segmentation_path=excel_paths.get('segmentation'),
        force_reload=True
    )
    memory_tracker.log_memory("Loaded Raw Excel Data")

    if raw_data_dict.get('clinical', pd.DataFrame()).empty:
        logger.error(color_text("Critical Error: Clinical data failed to load or is empty. Exiting.", TermColors.FAIL))
        sys.exit(1)

    # --- 3. Integrate Excel Sources (All Timepoints) ---
    logger.info(color_text("\n--- Integrating Excel Sources (All Timepoints) ---", TermColors.HEADER))
    # --- MODIFIED: Removed timepoint argument ---
    integrated_df = integrator.integrate_excel_sources(
        excel_data_sources=raw_data_dict
        # No timepoint filter applied here
    )
    # --- END MODIFIED ---
    memory_tracker.log_memory("Integrated Excel Sources")
    del raw_data_dict
    gc.collect()

    # --- REMOVED: Check for empty df related to single timepoint ---
    # Now we expect a potentially large df with all data

    if integrated_df.empty:
         logger.error(color_text("Data integration resulted in an empty DataFrame. Check input files and merge logic.", TermColors.FAIL))
         sys.exit(1)

    logger.info(f"Integrated DataFrame shape (All Timepoints): {integrated_df.shape}")
    logger.debug(f"Integrated DataFrame columns: {integrated_df.columns.tolist()}")


    # --- 4. Detailed Preprocessing ---
    logger.info(color_text("\n--- Applying Detailed Preprocessing ---", TermColors.HEADER))
    processor = ClinicalDataProcessor()
    processed_df = processor.preprocess_clinical_data(integrated_df)
    memory_tracker.log_memory("Applied Detailed Preprocessing")
    del integrated_df
    gc.collect()

    logger.info(f"Fully processed DataFrame shape: {processed_df.shape}")
    logger.debug(f"Fully processed DataFrame columns: {processed_df.columns.tolist()}")
    memory_tracker.log_memory("After final processing")


    # --- 5. Save Output ---
    logger.info(color_text("\n--- Saving Processed Data ---", TermColors.HEADER))
    # --- MODIFIED: Use the fixed output filename ---
    output_filename = f"{args.output_filename}.{args.output_format}"
    # --- END MODIFIED ---
    output_path = output_dir / output_filename

    try:
        if args.output_format == 'feather':
            # Feather format might have issues with complex index, reset it.
            processed_df.reset_index(drop=True).to_feather(output_path)
        elif args.output_format == 'csv':
            processed_df.to_csv(output_path, index=False)
        elif args.output_format == 'parquet':
            processed_df.to_parquet(output_path, index=False)
        logger.info(color_text(f"Successfully saved processed data (all timepoints) to: {output_path}", TermColors.OKGREEN))
    except Exception as e:
        logger.error(color_text(f"Error saving processed data to {output_path}: {e}", TermColors.FAIL), exc_info=True)
        sys.exit(1)

    logger.info(color_text("\nPhase 2 (Excel Data Processing & Integration - All Timepoints) completed successfully.", TermColors.BOLD))

if __name__ == "__main__":
    main()
