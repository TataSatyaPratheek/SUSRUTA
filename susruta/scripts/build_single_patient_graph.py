#!/usr/bin/env python
# scripts/build_single_patient_graph.py
"""
Script for Phase 3: Multimodal Knowledge Graph Construction (Single Patient).

Loads MRI features (HDF5) and processed tabular data (Feather) for ONE patient,
builds a unified knowledge graph using NetworkX, potentially including
temporal connections, and saves the graph. Designed for parallel execution.
"""

import argparse
import sys
import re
from pathlib import Path
import pandas as pd
import networkx as nx # Ensure networkx is imported as nx
import h5py
import logging
import gc
from collections import defaultdict
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import pickle # <--- IMPORT PICKLE

# --- Setup Project Path ---
project_root = Path(__file__).resolve().parents[1]
src_path = project_root / 'src'
if str(src_path) not in sys.path:
    print(f"Adding src directory to PYTHONPATH: {src_path}")
    sys.path.insert(0, str(src_path))

try:
    from susruta.graph.knowledge_graph import GliomaKnowledgeGraph
    from susruta.graph_builder.temporal_graph import TemporalGraphBuilder
    from susruta.utils.memory import MemoryTracker
except ImportError as e:
    print(f"Error importing susruta modules: {e}")
    print("Ensure the susruta package is installed or the project structure is correct.")
    sys.exit(1)

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # Use module name

# --- Color definitions ---
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
        description=color_text('SUSRUTA Phase 3: Knowledge Graph Construction (Single Patient)', TermColors.HEADER),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--patient-id', type=str, required=True,
                        help='The specific Patient ID (string format) to process.')
    parser.add_argument('--mri-features-hdf5', type=str, required=True,
                        help='Path to the HDF5 file containing MRI features.')
    parser.add_argument('--processed-tabular-dir', type=str, required=True,
                        help='Path to the directory containing processed tabular data Feather files.')
    parser.add_argument('--output-dir', type=str, default='./output/knowledge_graph',
                        help='Directory to save the constructed graph file.')
    parser.add_argument('--output-filename-prefix', type=str, default='patient_kg',
                        help='Prefix for the output graph filename.')
    parser.add_argument('--graph-format', type=str, default='gpickle', choices=['gpickle', 'graphml'],
                        help='Format to save the graph.')
    parser.add_argument('--build-mode', type=str, default='temporal', choices=['temporal', 'latest_static'],
                        help='Build mode: "temporal" or "latest_static".')
    parser.add_argument('--memory-limit-mb', type=float, default=2500, # Lower default for parallel
                        help='Approximate memory limit in MB for THIS process.')

    return parser.parse_args()

def load_mri_features_for_patient(hdf5_path: Path, patient_id_to_load: str, logger_instance: logging.Logger) -> Dict[int, Dict[str, Dict[str, float]]]:
    """Loads MRI features from HDF5 file ONLY for the specified patient."""
    logger_instance.info(f"Loading MRI features for Patient {patient_id_to_load} from: {hdf5_path}")
    features = defaultdict(dict) # timepoint -> sequence -> {feature: value}
    patient_key_pattern = re.compile(f'PatientID_0*{patient_id_to_load}$')

    try:
        with h5py.File(hdf5_path, 'r') as f:
            patient_key_found = None
            for key in f.keys():
                if patient_key_pattern.search(key):
                    patient_key_found = key
                    break

            if not patient_key_found:
                logger_instance.warning(f"Patient key matching '{patient_id_to_load}' not found in HDF5 file. Proceeding without MRI features.")
                return {}

            try:
                patient_group = f[patient_key_found]
                for timepoint_key in patient_group.keys():
                    timepoint_match = re.search(r'(\d+)$', timepoint_key)
                    if not timepoint_match: continue
                    timepoint = int(timepoint_match.group(1))

                    timepoint_group = patient_group[timepoint_key]
                    for sequence_key in timepoint_group.keys():
                        sequence_data = {}
                        for feature_name, value in timepoint_group[sequence_key].items():
                            if hasattr(value, 'shape') and value.shape == ():
                                sequence_data[feature_name] = float(value[()])
                            elif isinstance(value, (int, float, np.number)):
                                sequence_data[feature_name] = float(value)
                            else:
                                logger_instance.warning(f"Unexpected data type for feature '{feature_name}' in {patient_key_found}/{timepoint_key}/{sequence_key}: {type(value)}")
                        features[timepoint][sequence_key] = sequence_data
            except Exception as e:
                logger_instance.warning(f"Could not process group {patient_key_found} in HDF5: {e}")

        logger_instance.info(f"Loaded MRI features for Patient {patient_id_to_load} across {len(features)} timepoints.")
        return features
    except Exception as e:
        logger_instance.error(f"Failed to load HDF5 file {hdf5_path}: {e}")
        return {}

def find_timepoints_for_patient(processed_tabular_dir: Path, patient_id_to_find: str, logger_instance: logging.Logger) -> List[int]:
    """Finds available timepoints for a specific patient from Feather file names."""
    timepoints = set()
    pattern = re.compile(r'integrated_processed_data_tp(\d+)\.feather')
    logger_instance.info(f"Scanning for processed data for Patient {patient_id_to_find} in: {processed_tabular_dir}")
    for file_path in processed_tabular_dir.glob('*.feather'):
        match = pattern.search(file_path.name)
        if match:
            timepoint = int(match.group(1))
            try:
                df_ids = pd.read_feather(file_path, columns=['patient_id'])
                if patient_id_to_find in df_ids['patient_id'].astype(str).values:
                    timepoints.add(timepoint)
            except Exception as e:
                logger_instance.warning(f"Could not read patient IDs from {file_path}: {e}")

    sorted_timepoints = sorted(list(timepoints))
    logger_instance.info(f"Found timepoints {sorted_timepoints} for Patient {patient_id_to_find}.")
    return sorted_timepoints


def main():
    """Main execution function for single patient graph construction."""
    args = parse_arguments()
    patient_logger = logging.getLogger(f"Patient_{args.patient_id}")
    if not patient_logger.hasHandlers():
         handler = logging.StreamHandler(sys.stdout)
         formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
         handler.setFormatter(formatter)
         patient_logger.addHandler(handler)
         patient_logger.setLevel(logging.INFO) # Set back to INFO, DEBUG was for checking hasattr

    memory_tracker = MemoryTracker(threshold_mb=args.memory_limit_mb)
    patient_id = args.patient_id

    patient_logger.info(color_text(f"--- Starting processing for Patient {patient_id} ---", TermColors.HEADER))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    patient_logger.info(color_text(f"Output directory set to: {output_dir}", TermColors.OKBLUE))

    processed_tabular_dir = Path(args.processed_tabular_dir)
    mri_features_path = Path(args.mri_features_hdf5)

    if not mri_features_path.exists():
        patient_logger.error(color_text(f"MRI features HDF5 file not found: {mri_features_path}", TermColors.FAIL))
        sys.exit(1)
    if not processed_tabular_dir.is_dir():
        patient_logger.error(color_text(f"Processed tabular data directory not found: {processed_tabular_dir}", TermColors.FAIL))
        sys.exit(1)

    patient_logger.info(color_text(f"Loading MRI Features for Patient {patient_id}", TermColors.OKCYAN))
    patient_mri_features_all_tps = load_mri_features_for_patient(mri_features_path, patient_id, patient_logger)
    if not patient_mri_features_all_tps:
         patient_logger.warning(color_text(f"No MRI features loaded for Patient {patient_id}. Graph will contain only tabular data.", TermColors.WARNING))
    memory_tracker.log_memory(f"Loaded MRI features for P{patient_id}")

    patient_logger.info(color_text(f"Discovering Timepoints for Patient {patient_id}", TermColors.OKCYAN))
    timepoints = find_timepoints_for_patient(processed_tabular_dir, patient_id, patient_logger)
    memory_tracker.log_memory(f"Discovered timepoints for P{patient_id}")

    if not timepoints:
        patient_logger.warning(color_text(f"No timepoints with processed data found for Patient {patient_id}. Exiting.", TermColors.WARNING))
        sys.exit(0)

    patient_logger.info(color_text(f"Building Knowledge Graph for Patient {patient_id} (Mode: {args.build_mode})", TermColors.OKCYAN))
    temporal_builder = TemporalGraphBuilder(memory_limit_mb=args.memory_limit_mb * 0.5)

    timepoint_graphs = {}
    latest_valid_timepoint = -1

    patient_logger.info(color_text(f"Processing Timepoints: {timepoints}", TermColors.OKBLUE))
    memory_tracker.log_memory(f"Start Patient {patient_id} graph building")

    for tp in timepoints:
        patient_logger.debug(f"  Processing Timepoint {tp}") # Use debug for per-timepoint
        kg_builder = GliomaKnowledgeGraph(memory_limit_mb=args.memory_limit_mb * 0.6)

        tabular_file = processed_tabular_dir / f'integrated_processed_data_tp{tp}.feather'
        if not tabular_file.exists():
            patient_logger.warning(f"    Tabular data file not found for T{tp}. Skipping timepoint.")
            continue
        try:
            tabular_df_full = pd.read_feather(tabular_file)
            tabular_df = tabular_df_full[tabular_df_full['patient_id'].astype(str) == patient_id].copy()
            if tabular_df.empty:
                 patient_logger.warning(f"    No data found for Patient {patient_id} in {tabular_file.name}. Skipping timepoint.")
                 continue
            patient_logger.debug(f"    Loaded tabular data for T{tp}: {tabular_df.shape}")
        except Exception as e:
            patient_logger.error(f"    Failed to load/filter {tabular_file.name}: {e}")
            continue

        tabular_df['patient_id'] = tabular_df['patient_id'].astype(str)
        kg_builder.add_clinical_data(tabular_df, batch_size=1)

        patient_mri_features_tp = patient_mri_features_all_tps.get(tp, {})

        if patient_mri_features_tp:
             try:
                 patient_id_int = int(patient_id)
                 kg_builder.add_imaging_features({patient_id_int: patient_mri_features_tp})
                 patient_logger.debug(f"    Added MRI features for T{tp}")
             except ValueError:
                  patient_logger.warning(f"    Could not convert patient ID {patient_id} to int for MRI feature addition.")
             except Exception as e:
                  patient_logger.error(f"    Error adding imaging features for T{tp}: {e}")

        timepoint_graphs[tp] = kg_builder.G
        latest_valid_timepoint = tp
        patient_logger.debug(f"    Built graph for T{tp}. Nodes: {kg_builder.G.number_of_nodes()}, Edges: {kg_builder.G.number_of_edges()}")
        del kg_builder, tabular_df, tabular_df_full
        gc.collect()

    final_graph = None
    if not timepoint_graphs:
        patient_logger.warning(f"No valid timepoint graphs built for Patient {patient_id}. Skipping save.")
        sys.exit(0)

    if args.build_mode == 'temporal' and len(timepoint_graphs) > 1:
        patient_logger.info(f"  Building temporal graph")
        try:
            patient_id_int_for_temporal = int(patient_id)
            final_graph = temporal_builder.build_temporal_graph(timepoint_graphs, patient_id_int_for_temporal)
        except ValueError:
             patient_logger.error(f"  Cannot build temporal graph: Patient ID {patient_id} is not an integer. Using latest static graph.")
             final_graph = timepoint_graphs[latest_valid_timepoint]
        except Exception as e:
             patient_logger.error(f"  Error building temporal graph: {e}. Using latest static graph.")
             final_graph = timepoint_graphs[latest_valid_timepoint]

    elif args.build_mode == 'latest_static' or len(timepoint_graphs) == 1:
        patient_logger.info(f"  Using latest static graph (Timepoint {latest_valid_timepoint})")
        final_graph = timepoint_graphs[latest_valid_timepoint]
    else:
         patient_logger.info(f"  Defaulting to latest static graph (Timepoint {latest_valid_timepoint})")
         final_graph = timepoint_graphs[latest_valid_timepoint]

    if final_graph is not None and final_graph.number_of_nodes() > 0:
        output_filename = f"{args.output_filename_prefix}_{patient_id}.{args.graph_format}"
        output_path = output_dir / output_filename
        patient_logger.info(f"  Saving graph to {output_path} "
                    f"(Nodes: {final_graph.number_of_nodes()}, Edges: {final_graph.number_of_edges()})")

        # --- REMOVED DEBUG PRINT ---

        # --- START CORRECTED SAVING LOGIC ---
        try:
            if args.graph_format == 'gpickle':
                # Use pickle.dump as recommended for NetworkX 3.0+
                with open(output_path, "wb") as f:
                    pickle.dump(final_graph, f)
                patient_logger.info(color_text(f"  Successfully saved graph using pickle for Patient {patient_id}", TermColors.OKGREEN))

            elif args.graph_format == 'graphml':
                # Keep GraphML logic, ensuring complex types are handled
                graph_to_save = final_graph.copy()
                for _, data in graph_to_save.nodes(data=True):
                    for key, value in data.items():
                        if isinstance(value, (np.ndarray, list, dict, set)): data[key] = str(value)
                for _, _, data in graph_to_save.edges(data=True):
                        for key, value in data.items():
                            if isinstance(value, (np.ndarray, list, dict, set)): data[key] = str(value)
                nx.write_graphml(graph_to_save, output_path)
                patient_logger.info(color_text(f"  Successfully saved graph using graphml for Patient {patient_id}", TermColors.OKGREEN))

        except Exception as e:
            patient_logger.error(color_text(f"  Error saving graph to {output_path}: {e}", TermColors.FAIL), exc_info=True)
            sys.exit(1) # Exit with error code if saving fails
        # --- END CORRECTED SAVING LOGIC ---

    else:
            patient_logger.warning(f"  Skipping save for Patient {patient_id} as the final graph is empty or None.")

    memory_tracker.log_memory(f"End Patient {patient_id}")
    del final_graph, timepoint_graphs, patient_mri_features_all_tps
    gc.collect()

    patient_logger.info(color_text(f"--- Phase 3 completed for Patient {patient_id} ---", TermColors.BOLD))

if __name__ == "__main__":
    main()
