#!/bin/bash

echo "--- SUSRUTA: Phase 3 Parallel Graph Construction ---"

# --- Configuration ---
MRI_HDF5="/Users/vi/Documents/brain/susruta/output/mri_features_parallel.hdf5"
TABULAR_DIR="/Users/vi/Documents/brain/susruta/output/processed_tabular_data"
OUTPUT_DIR="./output/knowledge_graph"
OUTPUT_PREFIX="patient_kg"
GRAPH_FORMAT="gpickle"
BUILD_MODE="temporal" # or 'latest_static'

# --- Resource Limits for M1 Air 8GB RAM ---
# Start VERY conservatively. Monitor Activity Monitor closely!
# If memory usage spikes too high or system becomes unresponsive, STOP and reduce jobs/memory.
NUM_JOBS=2       # Try 2 concurrent jobs first. Maybe 3 if memory allows.
MEM_PER_JOB=2500 # MB per job. Total ~ NUM_JOBS * MEM_PER_JOB.

# --- 1. Discover Patient IDs ---
echo "Discovering patient IDs from $TABULAR_DIR..."
# Use python to reliably extract unique patient IDs from any feather file
PATIENT_IDS=$(python -c "
import pandas as pd
from pathlib import Path
import sys
import re

tabular_dir = Path('$TABULAR_DIR')
patient_ids = set()
pattern = re.compile(r'integrated_processed_data_tp(\d+)\.feather')

# Read IDs from just one file to get the list
found_file = False
for f in tabular_dir.glob('*.feather'):
    if pattern.search(f.name):
        try:
            df_ids = pd.read_feather(f, columns=['patient_id'])
            patient_ids.update(df_ids['patient_id'].astype(str).unique())
            found_file = True
            # break # Read only one file is enough to get all patient IDs if clinical data was common
        except Exception as e:
            print(f'Error reading {f}: {e}', file=sys.stderr)
            continue # Try next file if one fails

if not found_file:
     print(f'No matching Feather files found in {tabular_dir}', file=sys.stderr)
     exit(1)
if not patient_ids:
     print(f'No patient IDs found in Feather files.', file=sys.stderr)
     exit(1)

# Print space-separated list
print(' '.join(sorted(list(patient_ids))))
")

if [ -z "$PATIENT_IDS" ]; then
    echo "Error: Failed to extract patient IDs. Exiting."
    exit 1
fi
NUM_PATIENTS=$(echo $PATIENT_IDS | wc -w | xargs) # Count patients
echo "Found $NUM_PATIENTS patients."

# --- 2. Create Output Directory ---
mkdir -p "$OUTPUT_DIR"
echo "Output directory: $OUTPUT_DIR"

# --- 3. Run Parallel Graph Construction ---
echo "Starting parallel graph construction with $NUM_JOBS jobs..."

# GNU Parallel command
parallel --jobs $NUM_JOBS --eta --bar --halt soon,fail=1 \
    python scripts/build_single_patient_graph.py \
        --patient-id {} \
        --mri-features-hdf5 "$MRI_HDF5" \
        --processed-tabular-dir "$TABULAR_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --output-filename-prefix "$OUTPUT_PREFIX" \
        --graph-format "$GRAPH_FORMAT" \
        --build-mode "$BUILD_MODE" \
        --memory-limit-mb $MEM_PER_JOB \
    ::: $PATIENT_IDS

# Check exit status of parallel
if [ $? -eq 0 ]; then
    echo "--- Parallel graph construction completed successfully. ---"
else
    echo "--- Parallel graph construction failed for one or more patients. Check logs above. ---"
    exit 1
fi

exit 0
