# Define the MAXIMUM potential timepoint range to check
TIMEPOINTS=$(seq 1 6) # Try timepoints 1 through 6

# Run the script in parallel for each potential timepoint
# Adjust --jobs and --memory-limit-mb based on your M1 Air's resources (e.g., 8GB or 16GB RAM)
# Start conservatively, e.g., --jobs 3 or 4, --memory-limit-mb 2000 or 2500
parallel --jobs 4 --eta --bar \
    python scripts/process_excel_data.py \
        --scanner-excel /Users/vi/Documents/brain/susruta/example_data/MR_Scanner_data.xlsx \
        --clinical-excel /Users/vi/Documents/brain/susruta/example_data/MUGliomaPost_ClinicalDataFINAL032025.xlsx \
        --segmentation-excel /Users/vi/Documents/brain/susruta/example_data/MUGliomaPost_Segmentation_Volumes.xlsx \
        --output-dir ./output/processed_tabular_data \
        --output-filename integrated_processed_data_tp{} \
        --output-format feather \
        --timepoint {} \
        --memory-limit-mb 3000 \
    ::: $TIMEPOINTS
