# run_examples.py
"""
Runs all example scripts in the 'examples/' directory sequentially,
with exception handling and garbage collection.
"""

import os
import sys
import gc
import importlib.util
import importlib.machinery
import traceback
import time

# --- Configuration ---
# Assuming this script is in the project root directory
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
EXAMPLES_DIR = os.path.join(PROJECT_ROOT, "examples")
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

# Add src directory to Python path to allow example scripts to import susruta modules
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
    print(f"Added '{SRC_DIR}' to sys.path")

# List of example scripts to run in order
# Add or remove scripts as needed
EXAMPLE_SCRIPTS = [
    "data_processing.py",
    "excel_data_integration.py", # Requires example_data dir to be created
    "graph_construction.py",
    "model_training.py",         # Might take time, generates files
    "treatment_recommendation.py" # Depends on model_training output file
]
# --- End Configuration ---

def run_example(script_filename: str):
    """Loads and runs the main() function of a given example script."""
    script_path = os.path.join(EXAMPLES_DIR, script_filename)
    module_name = os.path.splitext(script_filename)[0] # e.g., "data_processing"

    if not os.path.exists(script_path):
        print(f"--- Skipping {script_filename}: File not found at {script_path} ---")
        return False

    print(f"\n{'='*10} Running Example: {script_filename} {'='*10}")
    start_time = time.time()
    success = False

    try:
        # Dynamically load the module from the file path
        spec = importlib.util.spec_from_file_location(module_name, script_path)
        if spec is None or spec.loader is None:
             raise ImportError(f"Could not create module spec for {script_path}")

        module = importlib.util.module_from_spec(spec)

        # Add the module to sys.modules *before* execution
        # This allows the module's internal imports to work correctly
        sys.modules[module_name] = module

        # Execute the module's code
        spec.loader.exec_module(module)

        # Check if the module has a main() function and run it
        if hasattr(module, "main") and callable(module.main):
            print(f"--- Executing {module_name}.main() ---")
            module.main()
            success = True
            print(f"--- Finished {module_name}.main() ---")
        else:
            print(f"--- Warning: No main() function found in {script_filename} ---")
            # Consider it successful if the module loaded without error,
            # even without a main()
            success = True

    except ImportError as e:
        print(f"--- FAILED: {script_filename} (Import Error) ---")
        print(f"Error details: {e}")
        print("Please ensure all dependencies are installed (pip install susruta[dev,viz])")
        traceback.print_exc()
        success = False
    except FileNotFoundError as e:
        print(f"--- FAILED: {script_filename} (File Not Found Error) ---")
        print(f"Error details: {e}")
        print("Check if the script expects specific files/directories (e.g., 'example_data').")
        traceback.print_exc()
        success = False
    except Exception as e:
        print(f"--- FAILED: {script_filename} ---")
        print(f"An unexpected error occurred: {type(e).__name__}")
        traceback.print_exc()
        success = False
    finally:
        # Clean up the imported module from sys.modules to avoid conflicts
        # if scripts have the same name (though unlikely here)
        if module_name in sys.modules:
            del sys.modules[module_name]

        # Perform garbage collection
        print("--- Running garbage collection ---")
        gc.collect()
        end_time = time.time()
        print(f"--- {script_filename} finished in {end_time - start_time:.2f} seconds. Status: {'Success' if success else 'Failed'} ---")

    return success

if __name__ == "__main__":
    print(f"Starting execution of SUSRUTA examples from: {EXAMPLES_DIR}")
    print("-" * 70)

    overall_success = True
    failed_scripts = []

    for script in EXAMPLE_SCRIPTS:
        if not run_example(script):
            overall_success = False
            failed_scripts.append(script)
            # Optional: Stop on first failure
            # print("\nStopping execution due to failure.")
            # break

    print("\n" + "=" * 70)
    if overall_success:
        print("All example scripts executed successfully.")
    else:
        print("Some example scripts failed:")
        for failed in failed_scripts:
            print(f"  - {failed}")
    print("=" * 70)
