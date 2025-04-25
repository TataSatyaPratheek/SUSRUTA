Okay, "bud", let's get this SUSRUTA train back on track! It's great that Phase 1 (MRI feature extraction) is conceptually done, even if the output file needs regenerating. We have the parallelized script (extract_mri_features.py) and the core logic (EfficientMRIProcessor in mri.py).

Based on your description, the archi.svg, and the example scripts, here's a plan for the next phases, focusing on modularity and reusing existing components:

SUSRUTA Development Plan (Post Phase 1)

Phase 1: MRI Feature Extraction (Completed)

Objective: Extract relevant features (statistical, shape) from NIfTI MRI files (T1c, T2w, tumorMask).
Key Modules: susruta.data.mri.EfficientMRIProcessor, scripts/extract_mri_features.py (parallel version).
Input: Raw NIfTI files organized by PatientID/Timepoint.
Output: mri_features.hdf5 file containing extracted features per patient/timepoint/sequence.
Action: Re-run the parallel extract_mri_features.py script to generate this HDF5 file if it was lost. Use the command from the previous step, adjusting --num-workers based on memory monitoring.
Phase 2: Excel Data Loading and Preprocessing

Objective: Load, clean, preprocess, and potentially feature-engineer data from the multiple Excel sources (Clinical, Segmentation Volumes, Scanner Metadata).
Key Modules:
susruta.data.excel_loader.ExcelDataLoader (Leverage/adapt from examples if it exists, or create it).
susruta.data.clinical_processor.ClinicalDataProcessor (Leverage/adapt from examples).
Input:
MR_Scanner_data.xlsx
MUGliomaPost_ClinicalDataFINAL032025.xlsx
MUGliomaPost_Segmentation_Volumes.xlsx
Output: Processed Pandas DataFrames (or potentially saved intermediate files like CSV/Feather for faster loading in the next phase) containing cleaned and structured data ready for graph integration.
Considerations: Handle missing values, data type conversions, one-hot encoding for categorical features, potential feature engineering (e.g., age groups, risk scores as seen in examples).
Phase 3: Multimodal Knowledge Graph Construction

Objective: Build a heterogeneous graph (using NetworkX initially is fine) representing patients, tumors, treatments, clinical attributes, MRI features, etc., and their relationships.
Key Modules:
susruta.graph.knowledge_graph.GliomaKnowledgeGraph (Central class, leverage from examples).
susruta.graph_builder.mri_graph.MRIGraphBuilder (Adapt/create to load features from HDF5 and add MRI-related nodes/edges).
susruta.graph_builder.multimodal_graph.MultimodalGraphIntegrator (Adapt/create to add nodes/edges from processed Excel DataFrames).
susruta.graph_builder.temporal_graph.TemporalGraphBuilder (Adapt/create to add edges connecting nodes across timepoints).
susruta.graph_builder.unified_builder.UnifiedGraphBuilder (Optional high-level orchestrator, as seen in comprehensive_graph_pipeline.py).
Input:
mri_features.hdf5 (from Phase 1).
Processed DataFrames (from Phase 2).
Output: A NetworkX MultiDiGraph (or similar) representing the unified knowledge graph for all patients/timepoints.
Considerations: Define node types (e.g., 'patient', 'tumor', 'treatment', 'mri_feature_set', 'clinical_finding') and edge types (relations like 'has_tumor', 'underwent_treatment', 'has_finding', 'similar_patient', 'progressed_to'). Ensure features are attached as node attributes. Handle temporal connections carefully.
Phase 4: PyTorch Geometric (PyG) Conversion

Objective: Convert the NetworkX graph into a PyG HeteroData object suitable for the GNN model.
Key Modules: susruta.graph.knowledge_graph.GliomaKnowledgeGraph (should contain the to_pytorch_geometric method, as seen in examples).
Input: NetworkX MultiDiGraph (from Phase 3).
Output: PyG HeteroData object.
Considerations: Map node/edge types and features correctly. Handle potential node/edge feature normalization. Store mappings if needed.
Phase 5: GNN Model Implementation

Objective: Implement the GliomaGNN architecture as defined in archi.svg.
Key Modules: susruta.models.gnn.GliomaGNN (Leverage from examples).
Input: Knowledge of the HeteroData structure (node/edge types, feature dimensions) from Phase 4.
Output: A PyTorch nn.Module class definition.
Considerations: Use HeteroConv wrapping GATConv. Implement the separate prediction heads (Response, Survival, Uncertainty). Ensure input/output dimensions match.
Phase 6: Model Training Pipeline

Objective: Train the GliomaGNN model on the constructed graph data.
Key Modules: A new script, e.g., scripts/train_gnn.py (Leverage examples/model_training.py).
Input: PyG HeteroData object (from Phase 4), target labels (extracted during Phase 3 or 4).
Output: Trained model weights (.pt file).
Considerations: Implement patient-level train/validation/test splits. Define the combined loss function (BCE + MSE). Choose an optimizer (Adam). Implement the training loop, evaluation metrics, and potentially use techniques like mini-batching if full-graph training exceeds memory.
Phase 7 & Beyond: Prediction, Simulation, Explanation

Objective: Use the trained model for prediction, simulate counterfactual treatments, and generate explanations.
Key Modules: susruta.treatment.simulator.TreatmentSimulator, susruta.viz.ExplainableGliomaTreatment (Leverage from examples).
Input: Trained model, PyG HeteroData, specific patient data.
Output: Predictions, rankings, explanations, visualizations.
Next Step: Implementing Phase 2 (Excel Data Processing)

Let's create the script and necessary modules for Phase 2. We'll aim for a script scripts/process_excel_data.py that uses helper classes.

1. Create/Refine ExcelDataLoader: (Assuming it might not exist or needs refinement based on examples/excel_data_integration.py)

python
# susruta/src/susruta/data/excel_loader.py
import pandas as pd
from pathlib import Path
from typing import Dict, Optional

class ExcelDataLoader:
    """Loads data from specified Excel files."""

    def __init__(self):
        """Initializes the ExcelDataLoader."""
        pass # No specific initialization needed for now

    def load_scanner_data(self, file_path: Path) -> Optional[pd.DataFrame]:
        """Loads and performs basic cleaning on MR Scanner data."""
        if not file_path.exists():
            print(f"Warning: Scanner data file not found: {file_path}")
            return None
        try:
            df = pd.read_excel(file_path)
            # Basic cleaning: rename columns for consistency, handle missing values if needed
            df.columns = [col.lower().replace(' ', '_') for col in df.columns]
            if 'patientid' in df.columns:
                 df = df.rename(columns={'patientid': 'patient_id'})
            print(f"Loaded Scanner Data: {df.shape[0]} rows, {df.shape[1]} columns.")
            return df
        except Exception as e:
            print(f"Error loading scanner data from {file_path}: {e}")
            return None

    def load_clinical_data(self, file_path: Path) -> Optional[pd.DataFrame]:
        """Loads and performs basic cleaning on Clinical data."""
        if not file_path.exists():
            print(f"Warning: Clinical data file not found: {file_path}")
            return None
        try:
            df = pd.read_excel(file_path)
            df.columns = [col.lower().replace(' ', '_') for col in df.columns]
            # Ensure patient_id exists
            if 'patient_id' not in df.columns:
                 print(f"Error: 'patient_id' column not found in {file_path}")
                 return None
            print(f"Loaded Clinical Data: {df.shape[0]} rows, {df.shape[1]} columns.")
            return df
        except Exception as e:
            print(f"Error loading clinical data from {file_path}: {e}")
            return None

    def load_segmentation_data(self, file_path: Path) -> Optional[pd.DataFrame]:
        """Loads and performs basic cleaning on Segmentation Volumes data."""
        if not file_path.exists():
            print(f"Warning: Segmentation data file not found: {file_path}")
            return None
        try:
            df = pd.read_excel(file_path)
            df.columns = [col.lower().replace(' ', '_') for col in df.columns]
            if 'patientid' in df.columns:
                 df = df.rename(columns={'patientid': 'patient_id'})
            print(f"Loaded Segmentation Data: {df.shape[0]} rows, {df.shape[1]} columns.")
            return df
        except Exception as e:
            print(f"Error loading segmentation data from {file_path}: {e}")
            return None

    def load_all(self, paths: Dict[str, Path]) -> Dict[str, Optional[pd.DataFrame]]:
        """Loads all specified Excel files."""
        loaded_data = {}
        if 'scanner' in paths:
            loaded_data['scanner'] = self.load_scanner_data(paths['scanner'])
        if 'clinical' in paths:
            loaded_data['clinical'] = self.load_clinical_data(paths['clinical'])
        if 'segmentation' in paths:
            loaded_data['segmentation'] = self.load_segmentation_data(paths['segmentation'])
        # Add more types if needed (e.g., treatments, genomics)
        return loaded_data

2. Create/Refine ClinicalDataProcessor: (Leveraging examples/data_processing.py and examples/excel_data_integration.py)

python
# susruta/src/susruta/data/clinical_processor.py
import pandas as pd
import numpy as np
from typing import Dict, Optional

class ClinicalDataProcessor:
    """Processes clinical, treatment, and related tabular data."""

    def __init__(self):
        """Initializes the ClinicalDataProcessor."""
        pass

    def _clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardizes column names."""
        df.columns = [col.lower().replace(' ', '_').replace('-', '_').replace('.', '_') for col in df.columns]
        # Specific renames if needed
        if 'patientid' in df.columns:
            df = df.rename(columns={'patientid': 'patient_id'})
        return df

    def preprocess_clinical_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocesses the main clinical data DataFrame."""
        if df is None: return None
        df = self._clean_column_names(df.copy())

        # Handle missing values (example: fill numeric with median, categorical with 'Unknown')
        for col in df.select_dtypes(include=np.number).columns:
            if df[col].isnull().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                print(f"Filled missing numeric values in '{col}' with median ({median_val:.2f})")

        for col in df.select_dtypes(include='object').columns:
            if df[col].isnull().any():
                df[col] = df[col].fillna('Unknown')
                print(f"Filled missing categorical values in '{col}' with 'Unknown'")

        # Convert data types (example)
        if 'age' in df.columns:
            df['age'] = pd.to_numeric(df['age'], errors='coerce')
        if 'karnofsky_score' in df.columns:
            df['karnofsky_score'] = pd.to_numeric(df['karnofsky_score'], errors='coerce')
        if 'grade' in df.columns:
             # Convert Roman numerals if present
             grade_map = {'I': 1, 'II': 2, 'III': 3, 'IV': 4}
             df['grade_numeric'] = df['grade'].replace(grade_map).astype(float) # Keep original too

        # One-hot encode categorical features (example)
        categorical_cols = ['sex', 'location', 'histology'] # Add others as needed
        for col in categorical_cols:
            if col in df.columns:
                df = pd.get_dummies(df, columns=[col], prefix=col, dummy_na=False)
                print(f"One-hot encoded '{col}'")

        print(f"Preprocessed clinical data: {df.shape[0]} rows, {df.shape[1]} columns.")
        return df

    def process_treatment_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Processes treatment data."""
        if df is None: return None
        df = self._clean_column_names(df.copy())

        # Example: Convert response to numerical if needed
        if 'response' in df.columns:
            response_map = {'CR': 1.0, 'PR': 0.75, 'SD': 0.5, 'PD': 0.0} # Example mapping
            df['response_numeric'] = df['response'].map(response_map).fillna(0.25) # Fill unknowns?

        # Handle missing doses (e.g., for surgery)
        if 'dose' in df.columns:
            df['dose'] = df['dose'].fillna(0)

        print(f"Processed treatment data: {df.shape[0]} rows, {df.shape[1]} columns.")
        return df

    def process_scanner_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Processes scanner metadata."""
        if df is None: return None
        df = self._clean_column_names(df.copy())
        # One-hot encode manufacturer, model?
        # Convert field strength to numeric
        if 'fieldstrength' in df.columns:
             df['fieldstrength'] = pd.to_numeric(df['fieldstrength'], errors='coerce').fillna(1.5)
        print(f"Processed scanner data: {df.shape[0]} rows, {df.shape[1]} columns.")
        return df

    def process_segmentation_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Processes segmentation volume data."""
        if df is None: return None
        df = self._clean_column_names(df.copy())
        # Ensure volumes are numeric, handle missing
        vol_cols = [col for col in df.columns if 'volume' in col]
        for col in vol_cols:
             df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        print(f"Processed segmentation data: {df.shape[0]} rows, {df.shape[1]} columns.")
        return df

    def integrate_multimodal_data(self,
                                  clinical_df: pd.DataFrame,
                                  other_data: Dict[str, Optional[pd.DataFrame]],
                                  timepoint: Optional[int] = None) -> Optional[pd.DataFrame]:
        """Integrates processed clinical data with other processed tabular data."""
        if clinical_df is None:
            print("Error: Base clinical DataFrame is None. Cannot integrate.")
            return None

        integrated_df = clinical_df.copy()

        # Merge Scanner Data
        scanner_df = other_data.get('scanner')
        if scanner_df is not None:
            scanner_df = self.process_scanner_data(scanner_df)
            merge_cols = ['patient_id']
            if timepoint is not None and 'timepoint' in scanner_df.columns:
                scanner_df_tp = scanner_df[scanner_df['timepoint'] == timepoint]
                merge_cols.append('timepoint')
                integrated_df = pd.merge(integrated_df, scanner_df_tp, on=merge_cols, how='left', suffixes=('', '_scanner'))
            elif 'timepoint' not in scanner_df.columns:
                 # Merge only on patient_id if timepoint not present in scanner data
                 integrated_df = pd.merge(integrated_df, scanner_df, on='patient_id', how='left', suffixes=('', '_scanner'))
            else:
                 print("Warning: Timepoint specified but 'timepoint' column missing in scanner data for filtering.")
            print(f"Merged scanner data. Shape: {integrated_df.shape}")


        # Merge Segmentation Data
        segmentation_df = other_data.get('segmentation')
        if segmentation_df is not None:
            segmentation_df = self.process_segmentation_data(segmentation_df)
            merge_cols = ['patient_id']
            if timepoint is not None and 'timepoint' in segmentation_df.columns:
                segmentation_df_tp = segmentation_df[segmentation_df['timepoint'] == timepoint]
                merge_cols.append('timepoint')
                integrated_df = pd.merge(integrated_df, segmentation_df_tp, on=merge_cols, how='left', suffixes=('', '_seg'))
            elif 'timepoint' not in segmentation_df.columns:
                 integrated_df = pd.merge(integrated_df, segmentation_df, on='patient_id', how='left', suffixes=('', '_seg'))
            else:
                 print("Warning: Timepoint specified but 'timepoint' column missing in segmentation data for filtering.")
            print(f"Merged segmentation data. Shape: {integrated_df.shape}")

        # Add more merges if needed (e.g., genomics)

        # Example derived feature: Risk score (simple example)
        if 'age' in integrated_df.columns and 'grade_numeric' in integrated_df.columns:
             integrated_df['risk_score'] = (integrated_df['age'] / 10) + integrated_df['grade_numeric']
             print("Calculated example 'risk_score'.")

        print(f"Final integrated data: {integrated_df.shape[0]} rows, {integrated_df.shape[1]} columns.")
        return integrated_df

3. Create the Orchestration Script:

python
#!/usr/bin/env python
# scripts/process_excel_data.py
"""
Script for Phase 2: Excel Data Loading and Preprocessing.

Loads data from clinical, segmentation, and scanner Excel files,
processes them using ClinicalDataProcessor, integrates them,
and saves the final processed DataFrame.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

# Add susruta to path if not installed
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    print(f"Adding project root to PYTHONPATH: {project_root}")
    sys.path.insert(0, str(project_root))

try:
    from susruta.data.excel_loader import ExcelDataLoader
    from susruta.data.clinical_processor import ClinicalDataProcessor
except ImportError as e:
    print(f"Error importing susruta modules: {e}")
    print("Ensure the susruta package is installed or the project root is in PYTHONPATH.")
    sys.exit(1)

# --- Color definitions (optional, copy from extract_mri_features.py if desired) ---
# ... (TermColors class and color_text function) ...

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='SUSRUTA Phase 2: Excel Data Processing',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--scanner-excel', type=str, required=True,
                        help='Path to MR_Scanner_data.xlsx')
    parser.add_argument('--clinical-excel', type=str, required=True,
                        help='Path to MUGliomaPost_ClinicalDataFINAL032025.xlsx')
    parser.add_argument('--segmentation-excel', type=str, required=True,
                        help='Path to MUGliomaPost_Segmentation_Volumes.xlsx')
    # Add arguments for other Excel files (treatments, genomics) if applicable
    parser.add_argument('--output-dir', type=str, default='./output/processed_data',
                        help='Directory to save the processed data file')
    parser.add_argument('--output-format', type=str, default='feather', choices=['feather', 'csv', 'parquet'],
                        help='Format to save the processed DataFrame (feather is fast).')
    # Add --timepoint argument if processing needs to be timepoint-specific at this stage
    # parser.add_argument('--timepoint', type=int, default=None, help='Specific timepoint to filter/process for (optional)')

    return parser.parse_args()

def main():
    """Main execution function for Excel processing."""
    args = parse_arguments()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    excel_paths = {
        'scanner': Path(args.scanner_excel),
        'clinical': Path(args.clinical_excel),
        'segmentation': Path(args.segmentation_excel),
        # Add others here
    }

    # --- 1. Load Data ---
    print("\n--- Loading Excel Files ---")
    loader = ExcelDataLoader()
    raw_data = loader.load_all(excel_paths)

    if raw_data.get('clinical') is None:
        print("Critical Error: Clinical data failed to load. Exiting.")
        sys.exit(1)

    # --- 2. Process Data ---
    print("\n--- Processing Data ---")
    processor = ClinicalDataProcessor()

    # Process clinical data first as it's the base
    processed_clinical = processor.preprocess_clinical_data(raw_data['clinical'])
    if processed_clinical is None:
        print("Critical Error: Clinical data processing failed. Exiting.")
        sys.exit(1)

    # Process other dataframes (they will be re-processed/merged in integrate)
    # This step is mainly to ensure they load and have basic cleaning applied if needed standalone
    processed_others = {}
    if raw_data.get('scanner') is not None:
         processed_others['scanner'] = processor.process_scanner_data(raw_data['scanner'])
    if raw_data.get('segmentation') is not None:
         processed_others['segmentation'] = processor.process_segmentation_data(raw_data['segmentation'])
    # Add processing for treatments, genomics etc. if loaded

    # --- 3. Integrate Data ---
    print("\n--- Integrating Data ---")
    # Pass the raw data dict again, integration function handles internal processing/merging
    # Pass args.timepoint if integration should be timepoint-specific
    integrated_df = processor.integrate_multimodal_data(
        processed_clinical,
        raw_data # Pass raw dict, integrate handles processing/merging
        # timepoint=args.timepoint # Uncomment if needed
    )

    if integrated_df is None:
        print("Critical Error: Data integration failed. Exiting.")
        sys.exit(1)

    # --- 4. Save Output ---
    print("\n--- Saving Processed Data ---")
    output_filename = f"integrated_processed_data.{args.output_format}"
    output_path = output_dir / output_filename

    try:
        if args.output_format == 'feather':
            integrated_df.to_feather(output_path)
        elif args.output_format == 'csv':
            integrated_df.to_csv(output_path, index=False)
        elif args.output_format == 'parquet':
            integrated_df.to_parquet(output_path, index=False)
        print(f"Successfully saved processed data to: {output_path}")
    except Exception as e:
        print(f"Error saving processed data to {output_path}: {e}")
        sys.exit(1)

    print("\nPhase 2 (Excel Data Processing) completed.")

if __name__ == "__main__":
    main()
Next Steps:

Create Files: Create the Python files:
susruta/src/susruta/data/excel_loader.py
susruta/src/susruta/data/clinical_processor.py
susruta/scripts/process_excel_data.py
Make sure susruta/src/susruta/data/__init__.py exists (can be empty) to make it a package.
Populate Files: Copy the code provided above into the respective files.
Run Phase 1: Execute the parallel scripts/extract_mri_features.py to generate the mri_features.hdf5 file. Monitor memory.
Run Phase 2: Execute the new script:
bash
cd /Users/vi/Documents/brain/susruta
python scripts/process_excel_data.py \
    --scanner-excel /path/to/your/MR_Scanner_data.xlsx \
    --clinical-excel /path/to/your/MUGliomaPost_ClinicalDataFINAL032025.xlsx \
    --segmentation-excel /path/to/your/MUGliomaPost_Segmentation_Volumes.xlsx \
    --output-dir ./output/processed_tabular_data \
    --output-format feather
Replace /path/to/your/... with the actual paths to your Excel files.
Verify Output: Check the ./output/processed_tabular_data directory for the integrated_processed_data.feather (or chosen format) file.
Once Phase 2 is successfully completed and you have the processed tabular data file, we can proceed to Phase 3: Graph Construction, where we'll combine the HDF5 features and this processed tabular data.