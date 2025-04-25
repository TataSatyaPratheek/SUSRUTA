# susruta/src/susruta/data/mri.py
"""
Memory-efficient MRI processing for brain tumor image analysis.

Includes optimized feature extraction and region-of-interest processing.
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import gc
import numpy as np
import SimpleITK as sitk
import glob # Add import

# Assuming MemoryTracker is correctly defined in this relative path
from ..utils.memory import MemoryTracker


class EfficientMRIProcessor:
    """Memory-efficient MRI processing for large volumes."""

    def __init__(self, memory_limit_mb: float = 2000):
        """
        Initialize the MRI processor.

        Args:
            memory_limit_mb: Memory limit in MB for processing
        """
        self.memory_limit_mb = memory_limit_mb
        self.memory_tracker = MemoryTracker(threshold_mb=memory_limit_mb)

    def load_nifti_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Load NIfTI file metadata without loading full volume into memory.

        Args:
            file_path: Path to NIfTI file

        Returns:
            Dictionary of image metadata
        """
        if not Path(file_path).exists():
             raise FileNotFoundError(f"NIfTI file not found: {file_path}")
        reader = sitk.ImageFileReader()
        reader.SetFileName(file_path)
        try:
            reader.ReadImageInformation()
        except Exception as e:
            raise RuntimeError(f"Failed to read metadata from {file_path}: {e}")
        return {
            'size': reader.GetSize(),
            'spacing': reader.GetSpacing(),
            'origin': reader.GetOrigin(),
            'direction': reader.GetDirection()
        }

    def compute_bounding_box(self, mask_path: str) -> Tuple[List[int], List[int]]:
        """
        Compute tumor bounding box from mask without loading full volume.

        Args:
            mask_path: Path to tumor mask NIfTI file

        Returns:
            Tuple of (min_coords, max_coords) for bounding box
        """
        if not Path(mask_path).exists():
             raise FileNotFoundError(f"Mask file not found for bounding box: {mask_path}")
        try:
            mask = sitk.ReadImage(mask_path)
        except Exception as e:
            raise RuntimeError(f"Failed to read mask image {mask_path}: {e}")


        # Convert to binary mask if needed (ensure label 1 exists)
        # Use StatisticsImageFilter to check for non-zero pixels first
        stats_filter = sitk.StatisticsImageFilter()
        stats_filter.Execute(mask)
        if stats_filter.GetSum() == 0:
             print(f"Warning: Mask {mask_path} is empty (all zeros). Returning full image bounds.")
             size = mask.GetSize()
             return ([0, 0, 0], [size[0]-1, size[1]-1, size[2]-1])

        # Ensure mask is integer type for LabelStatistics
        if mask.GetPixelIDValue() not in [sitk.sitkUInt8, sitk.sitkInt8, sitk.sitkUInt16, sitk.sitkInt16, sitk.sitkUInt32, sitk.sitkInt32, sitk.sitkUInt64, sitk.sitkInt64]:
             print(f"Warning: Mask pixel type ({mask.GetPixelIDValue()}) is not integer. Casting to UInt8 for bounding box calculation.")
             mask = sitk.Cast(mask, sitk.sitkUInt8)

        # Use label statistics to get bounding box efficiently
        # We need a binary image (0 or 1) for LabelStatisticsImageFilter BoundingBox
        binary_mask = sitk.BinaryThreshold(mask, lowerThreshold=1, upperThreshold=mask.GetPixelIDValue(), insideValue=1, outsideValue=0) # Threshold to get only label 1 (or higher if present)

        label_stats = sitk.LabelStatisticsImageFilter()
        try:
            label_stats.Execute(binary_mask, binary_mask) # Use binary mask for both inputs
        except Exception as e:
             raise RuntimeError(f"Failed to execute LabelStatisticsImageFilter on {mask_path}: {e}")


        # Check if the label 1 exists after binarization
        if not label_stats.HasLabel(1):
            # This case might happen if the original mask had values > 1 but not == 1
            # Or if the mask was truly empty despite the initial sum check (unlikely but possible)
            print(f"Warning: Label 1 not found in binary mask derived from {mask_path} after thresholding. Returning full image bounds.")
            size = mask.GetSize()
            return ([0, 0, 0], [size[0]-1, size[1]-1, size[2]-1])

        # sitk bbox format: [xStart, yStart, zStart, xSize, ySize, zSize]
        bbox = label_stats.GetBoundingBox(1)

        # Convert to min/max coordinates format
        bbox_min = [bbox[0], bbox[1], bbox[2]] # xStart, yStart, zStart
        bbox_max = [bbox[0] + bbox[3] - 1, bbox[1] + bbox[4] - 1, bbox[2] + bbox[5] - 1] # start + size - 1

        # Cleanup
        del mask, binary_mask, label_stats, stats_filter
        gc.collect()

        return bbox_min, bbox_max


    def extract_roi_features(self, img_path: str, mask_path: str) -> Dict[str, float]:
        """
        Extract features only from tumor region to save memory.

        Args:
            img_path: Path to MRI NIfTI file
            mask_path: Path to tumor mask NIfTI file

        Returns:
            Dictionary of computed features
        """
        # Track memory
        self.memory_tracker.log_memory("Starting ROI extraction")

        # Validate inputs
        if not Path(img_path).exists():
            raise FileNotFoundError(f"Image file not found for ROI extraction: {img_path}")
        if not Path(mask_path).exists():
            raise FileNotFoundError(f"Mask file not found for ROI extraction: {mask_path}")

        try:
            # Get bounding box of tumor with margin
            bbox_min, bbox_max = self.compute_bounding_box(mask_path)
            margin = 5  # Add 5 voxel margin around tumor

            # Get image metadata
            img_info = self.load_nifti_metadata(img_path)
            size = img_info['size']

            # Add margin and clamp to image bounds
            clamped_bbox_min = [max(0, x - margin) for x in bbox_min]
            clamped_bbox_max = [min(s-1, x + margin) for x, s in zip(bbox_max, size)]

            # Calculate ROI size based on clamped bounding box
            roi_size = [clamped_bbox_max[i] - clamped_bbox_min[i] + 1 for i in range(3)]

            # Check for valid ROI size
            if any(s <= 0 for s in roi_size):
                 print(f"Warning: Invalid ROI size calculated for {img_path} (Size: {roi_size}, BBox Min: {clamped_bbox_min}). Skipping ROI feature extraction.")
                 return {} # Return empty dict if ROI is invalid

            # Estimate memory requirements
            voxel_bytes = 4  # Assume 32-bit float for estimation
            roi_memory_mb = np.prod(roi_size) * voxel_bytes / (1024 * 1024)

            self.memory_tracker.log_memory(f"ROI calculation (size: {roi_size}, memory: {roi_memory_mb:.2f}MB)")

            # Load full images (SimpleITK handles memory relatively well here)
            img = sitk.ReadImage(img_path)
            mask = sitk.ReadImage(mask_path)

            features = {}

            # --- Process ROI ---
            # Use clamped bounding box for RegionOfInterest
            roi_img = sitk.RegionOfInterest(img, roi_size, clamped_bbox_min)
            roi_mask = sitk.RegionOfInterest(mask, roi_size, clamped_bbox_min)

            # Extract features using the helper method
            features = self._compute_roi_features(roi_img, roi_mask)
            self.memory_tracker.log_memory("Feature extraction complete")

            # Clean up intermediate images explicitly
            del roi_img, roi_mask
            gc.collect()
            # --- End Process ROI ---

            # NOTE: Chunking logic removed for simplification, as SimpleITK's
            # RegionOfInterest is generally efficient. Re-introduce if memory
            # errors persist with very large ROIs relative to the limit.

        except FileNotFoundError as e:
             print(f"    Error during ROI extraction: {e}")
             return {}
        except RuntimeError as e:
             print(f"    Error during ROI extraction: {e}")
             return {}
        except Exception as e:
            print(f"    Unexpected error during ROI extraction for {img_path}: {e}")
            # Optionally re-raise or return empty features
            return {}
        finally:
            # Ensure full images are released if they were loaded
            if 'img' in locals(): del img
            if 'mask' in locals(): del mask
            gc.collect()
            self.memory_tracker.log_memory("Finished ROI extraction and cleanup")


        return features

    def _compute_roi_features(self, img: sitk.Image, mask: sitk.Image) -> Dict[str, float]:
        """
        Compute first-order statistics features from an ROI.

        Args:
            img: SimpleITK image (ROI)
            mask: SimpleITK mask (ROI)

        Returns:
            Dictionary of computed features
        """
        try:
            # Ensure mask is integer type for logical operations
            if mask.GetPixelIDValue() not in [sitk.sitkUInt8, sitk.sitkInt8, sitk.sitkUInt16, sitk.sitkInt16, sitk.sitkUInt32, sitk.sitkInt32, sitk.sitkUInt64, sitk.sitkInt64]:
                print(f"    Warning: ROI Mask pixel type ({mask.GetPixelIDValue()}) is not integer. Casting to UInt8.")
                mask = sitk.Cast(mask, sitk.sitkUInt8)

            # Convert to numpy array for processing - only for masked region
            img_array = sitk.GetArrayFromImage(img)
            mask_array = sitk.GetArrayFromImage(mask)

            # Calculate first-order statistics on masked region (where mask > 0)
            masked_img = img_array[mask_array > 0]

            if masked_img.size == 0:
                # No tumor voxels in this ROI
                print("    Warning: No non-zero voxels found in the ROI mask.")
                return {
                    'mean': 0.0,
                    'std': 0.0,
                    'min': 0.0,
                    'max': 0.0,
                    'p25': 0.0,
                    'p50': 0.0,
                    'p75': 0.0,
                    'volume_voxels': 0.0
                }

            features = {
                'mean': float(np.mean(masked_img)),
                'std': float(np.std(masked_img)),
                'min': float(np.min(masked_img)),
                'max': float(np.max(masked_img)),
                'p25': float(np.percentile(masked_img, 25)),
                'p50': float(np.percentile(masked_img, 50)), # Median
                'p75': float(np.percentile(masked_img, 75)),
                'volume_voxels': float(masked_img.size) # Number of non-zero voxels
            }

            # Clean up numpy arrays
            del img_array, mask_array, masked_img
            gc.collect()

            return features

        except Exception as e:
            print(f"    Error computing ROI features: {e}")
            return { # Return default zero values on error
                    'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0,
                    'p25': 0.0, 'p50': 0.0, 'p75': 0.0, 'volume_voxels': 0.0
            }


    def _combine_chunk_features(self, chunk_features: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Combine features from multiple chunks (weighted by volume).
        NOTE: This function is kept for potential future use if chunking is re-enabled,
              but it's not currently called with the simplified ROI extraction.

        Args:
            chunk_features: List of feature dictionaries from each chunk

        Returns:
            Combined features dictionary
        """
        # Filter out empty chunks (volume_voxels == 0) before combining
        non_empty_chunks = [f for f in chunk_features if f.get('volume_voxels', 0) > 0]

        if not non_empty_chunks:
            # If all chunks were empty, return the structure of the first chunk (all zeros)
            return chunk_features[0] if chunk_features else {
                'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0,
                'p25': 0.0, 'p50': 0.0, 'p75': 0.0, 'volume_voxels': 0.0
            }

        # Calculate total volume from non-empty chunks
        total_volume = sum(f['volume_voxels'] for f in non_empty_chunks)

        # Combined dictionary
        combined: Dict[str, float] = {}

        # Min and Max are straightforward
        combined['min'] = min(f['min'] for f in non_empty_chunks)
        combined['max'] = max(f['max'] for f in non_empty_chunks)
        combined['volume_voxels'] = float(total_volume)

        # Weighted statistics using only non-empty chunks
        if total_volume > 0:
            for stat in ['mean', 'p25', 'p50', 'p75']:
                combined[stat] = sum(f[stat] * f['volume_voxels'] for f in non_empty_chunks) / total_volume

            # Combine variances for standard deviation
            combined_mean = combined['mean']
            # E[X^2] = Var(X) + (E[X])^2
            sum_of_squares_term = sum(
                f['volume_voxels'] * (f['std']**2 + f['mean']**2)
                for f in non_empty_chunks
            )
            combined_e_x2 = sum_of_squares_term / total_volume
            combined_variance = combined_e_x2 - combined_mean**2

            # Ensure variance is non-negative due to potential floating point errors
            combined_variance = max(0, combined_variance)
            combined['std'] = np.sqrt(combined_variance)
        else:
            # Handle case where total_volume is zero (shouldn't happen with non_empty_chunks check, but safety first)
            combined['mean'] = 0.0
            combined['std'] = 0.0
            combined['p25'] = 0.0
            combined['p50'] = 0.0
            combined['p75'] = 0.0

        return combined

    def extract_features_for_patient(self,
                                 patient_id: int,
                                 data_dir: str, # EXPECTED to be the specific timepoint dir
                                 timepoint: int, # Still useful for logging/errors
                                 sequences: Optional[List[str]] = None
                                 ) -> Dict[str, Dict[str, float]]:
        """
        Extract features from all MRI sequences for a patient timepoint.

        Args:
            patient_id: Patient identifier
            data_dir: Specific directory for the patient and timepoint
                      (e.g., /path/to/base/PatientID_0003/Timepoint_1)
            timepoint: Timepoint number (used for logging/errors)
            sequences: Optional list of sequences to process (e.g., ['t1c', 't2w'])
                       Defaults to ['t1c', 't1n', 't2f', 't2w'].
                       'tumorMask' will be ignored if included here, it's handled separately.

        Returns:
            Dictionary of features {sequence_name_or_tumor: {feature: value}}
        """
        # --- START FIX: Assume data_dir is the correct timepoint path ---
        timepoint_path = Path(data_dir) # Convert the input string path to a Path object

        # Check if the provided directory actually exists
        if not timepoint_path.is_dir():
            # Handle error: directory not found
            # Use the original data_dir string in the error message for clarity
            print(f"    Error: Directory not found: {data_dir}")
            return {} # Return empty if dir not found
        # --- END FIX ---

        # Define default sequences if none provided
        default_img_sequences = ['t1c', 't1n', 't2f', 't2w']
        if sequences is None:
            sequences_to_process = default_img_sequences
        else:
            # Filter out 'tumorMask' (case-insensitive) as it's handled separately
            sequences_to_process = [s for s in sequences if s.lower() != 'tumormask']
            if not sequences_to_process:
                 # If only tumorMask was passed or list was empty, use defaults
                 print(f"    Warning: No image sequences provided or only 'tumorMask'. Using default image sequences: {default_img_sequences}")
                 sequences_to_process = default_img_sequences


        # Get file paths for specified image sequences using glob
        sequence_files = {}
        print(f"    Looking for sequences: {', '.join(sequences_to_process)}")
        for sequence in sequences_to_process:
            # Use glob to find files matching the pattern within the timepoint_path
            pattern = str(timepoint_path / f'*{sequence}*.nii*') # Search within the given dir

            matches = glob.glob(pattern)
            if matches:
                # Handle multiple matches if necessary (e.g., log warning, take first)
                if len(matches) > 1:
                    print(f"    Warning: Multiple files found for sequence '{sequence}' in {timepoint_path}. Using first match: {Path(matches[0]).name}")
                sequence_files[sequence] = matches[0]
                print(f"    Found {sequence}: {Path(matches[0]).name}") # Log found file name
            # else: # Don't warn here, warn later if *no* sequences found at all
                # print(f"    Info: Sequence '{sequence}' not found for patient {patient_id}, timepoint {timepoint} in {timepoint_path}")
                pass # Silently ignore missing optional sequences

        # Get tumor mask path using glob
        mask_pattern = str(timepoint_path / '*tumorMask*.nii*') # Search within the given dir
        mask_files = glob.glob(mask_pattern)
        mask_path = None
        if mask_files:
            if len(mask_files) > 1:
                 print(f"    Warning: Multiple files found for 'tumorMask' in {timepoint_path}. Using first match: {Path(mask_files[0]).name}")
            mask_path = mask_files[0]
            print(f"    Found tumorMask: {Path(mask_path).name}") # Log found mask file name
        # else: # Don't warn here, handle missing mask below

        # --- Validation after searching ---
        if not sequence_files and not mask_path:
             print(f"    Warning: No requested sequence files AND no tumor mask found for P{patient_id:04d} T{timepoint} in {timepoint_path}. Skipping.")
             return {} # Return empty if nothing useful found

        if not mask_path:
             # If no mask found, we cannot calculate ROI or tumor shape features.
             # Decide if this is an error or just means no features can be extracted.
             print(f"    Warning: No tumor mask found for P{patient_id:04d} T{timepoint} in {timepoint_path}. Cannot extract ROI or shape features.")
             return {} # Return empty dict as essential input is missing

        if not sequence_files:
             # If mask exists but no image sequences found, we can still extract tumor shape features.
             print(f"    Warning: No image sequence files found for P{patient_id:04d} T{timepoint} in {timepoint_path}. Only extracting tumor shape features.")
             # Proceed to extract only tumor features


        # --- Feature Extraction ---
        features: Dict[str, Dict[str, float]] = {} # Initialize features dictionary

        # Extract ROI features for each found image sequence (requires mask)
        if sequence_files and mask_path: # Only if we have images AND a mask
            print(f"    Extracting ROI features...")
            for sequence, file_path in sequence_files.items():
                print(f"      Processing {sequence}...")
                try:
                    # Pass the full paths to the feature extraction
                    sequence_features = self.extract_roi_features(file_path, mask_path)
                    if sequence_features: # Add only if features were successfully extracted
                         features[sequence] = sequence_features
                    else:
                         print(f"      No ROI features extracted for {sequence}.")
                except Exception as e:
                    # Log error but continue with other sequences/features
                    print(f"      Error processing ROI for {sequence} (P{patient_id:04d} T{timepoint}): {e}")
                    # Optionally add an empty dict or specific error marker
                    # features[sequence] = {'error': str(e)}

        # Extract tumor volume and shape features from mask (always try if mask exists)
        if mask_path:
            print(f"    Extracting tumor shape features...")
            try:
                # Pass the full path to the mask feature extraction
                tumor_shape_features = self._extract_tumor_features(mask_path)
                if tumor_shape_features: # Add only if features were successfully extracted
                     features['tumor'] = tumor_shape_features
                else:
                     print(f"      No tumor shape features extracted.")
            except Exception as e:
                # Log error but potentially continue if other features were extracted
                print(f"    Error processing tumor mask shape features for P{patient_id:04d} T{timepoint} ({mask_path}): {e}")
                # Optionally add an empty dict or specific error marker
                features['tumor'] = {} # Add empty dict on error

        if not features:
             print(f"    Warning: No features were successfully extracted for P{patient_id:04d} T{timepoint}.")

        return features


    def _extract_tumor_features(self, mask_path: str) -> Dict[str, float]:
        """
        Extract shape and volume features from tumor mask using LabelShapeStatisticsImageFilter.

        Args:
            mask_path: Path to tumor mask NIfTI file

        Returns:
            Dictionary of tumor features, or empty dict on error or if mask is empty/invalid.
        """
        if not Path(mask_path).exists():
            print(f"    Error: Mask file not found for shape features: {mask_path}")
            return {}
        try:
            mask = sitk.ReadImage(mask_path)
        except Exception as e:
            print(f"    Error reading mask file {mask_path} for shape features: {e}")
            return {}

        # Check if mask is empty first using StatisticsImageFilter (more robust)
        stats_filter = sitk.StatisticsImageFilter()
        stats_filter.Execute(mask)
        if stats_filter.GetSum() == 0:
            print(f"    Warning: Mask {mask_path} is empty (all zeros). Returning zero shape features.")
            return {
                'volume_mm3': 0.0,
                'surface_area': 0.0,
                'elongation': 0.0,
                'roundness': 0.0
            }

        # Ensure mask is integer type for LabelShapeStatistics
        if mask.GetPixelIDValue() not in [sitk.sitkUInt8, sitk.sitkInt8, sitk.sitkUInt16, sitk.sitkInt16, sitk.sitkUInt32, sitk.sitkInt32, sitk.sitkUInt64, sitk.sitkInt64]:
             print(f"    Warning: Mask pixel type ({mask.GetPixelIDValue()}) is not integer. Casting to UInt8 for shape statistics.")
             mask = sitk.Cast(mask, sitk.sitkUInt8)

        # Use label shape statistics for efficient feature extraction
        shape_stats = sitk.LabelShapeStatisticsImageFilter()
        try:
            shape_stats.Execute(mask)
        except Exception as e:
             print(f"    Error executing LabelShapeStatisticsImageFilter on {mask_path}: {e}")
             return {} # Return empty dict on failure


        # Check if the label (usually 1 for tumor) exists
        label_to_analyze = 1 # Assuming tumor label is 1
        if not shape_stats.HasLabel(label_to_analyze):
            # Check if *any* label > 0 exists if label 1 is missing
            existing_labels = shape_stats.GetLabels()
            valid_labels = [l for l in existing_labels if l > 0]
            if valid_labels:
                 label_to_analyze = valid_labels[0] # Use the first valid label found
                 print(f"    Warning: Label 1 not found in mask {mask_path}. Using first found label {label_to_analyze} for shape statistics.")
            else:
                 print(f"    Warning: No valid labels (> 0) found in mask {mask_path}. Returning zero shape features.")
                 return {
                    'volume_mm3': 0.0,
                    'surface_area': 0.0,
                    'elongation': 0.0,
                    'roundness': 0.0
                 }

        # Get physical space measurements for the chosen label
        try:
            volume = shape_stats.GetPhysicalSize(label_to_analyze)
            # Perimeter in 3D is surface area in SimpleITK/ITK
            surface = shape_stats.GetPerimeter(label_to_analyze)
            elongation = shape_stats.GetElongation(label_to_analyze)
            roundness = shape_stats.GetRoundness(label_to_analyze)
            # Add Feret Diameter (max distance between any two points on surface)
            feret_diameter = shape_stats.GetFeretDiameter(label_to_analyze)

        except Exception as e:
            print(f"    Error getting shape statistics for label {label_to_analyze} from {mask_path}: {e}")
            return {} # Return empty dict if getting specific stats fails


        features = {
            'volume_mm3': float(volume),
            'surface_area_mm2': float(surface), # More descriptive name
            'elongation': float(elongation),
            'roundness': float(roundness),
            'feret_diameter_mm': float(feret_diameter) # Added feature
        }

        # Cleanup
        del mask, shape_stats, stats_filter
        gc.collect()

        return features

# Example Usage (Optional - for testing within this file)
if __name__ == '__main__':
    # This block will only run if the script is executed directly
    # It's useful for testing the class methods
    print("Testing EfficientMRIProcessor...")

    # Create a dummy directory structure and files for testing
    test_base_dir = Path("./temp_mri_test_data")
    patient_id = 999
    timepoint = 1
    test_patient_dir = test_base_dir / f"PatientID_{patient_id:04d}"
    test_timepoint_dir = test_patient_dir / f"Timepoint_{timepoint}"
    test_timepoint_dir.mkdir(parents=True, exist_ok=True)

    # Create dummy NIfTI images (replace with actual image creation if needed)
    # For simplicity, we'll just create empty files with the right names
    dummy_t1c_path = test_timepoint_dir / f"PatientID_{patient_id:04d}_Timepoint_{timepoint}_brain_t1c.nii.gz"
    dummy_t2w_path = test_timepoint_dir / f"PatientID_{patient_id:04d}_Timepoint_{timepoint}_brain_t2w.nii.gz"
    dummy_mask_path = test_timepoint_dir / f"PatientID_{patient_id:04d}_Timepoint_{timepoint}_tumorMask.nii.gz"

    # --- Create SimpleITK dummy images ---
    # Create a small dummy image (e.g., 10x10x5 voxels)
    image_size = [10, 10, 5]
    dummy_img_data = np.random.rand(*image_size).astype(np.float32) * 1000
    dummy_mask_data = np.zeros(image_size, dtype=np.uint8)
    # Create a small mask region
    dummy_mask_data[3:7, 3:7, 1:4] = 1

    sitk_img_t1c = sitk.GetImageFromArray(dummy_img_data)
    sitk_img_t1c.SetSpacing([1.0, 1.0, 3.0]) # Example spacing

    sitk_img_t2w = sitk.GetImageFromArray(dummy_img_data + 50) # Slightly different data
    sitk_img_t2w.SetSpacing([1.0, 1.0, 3.0])

    sitk_mask = sitk.GetImageFromArray(dummy_mask_data)
    sitk_mask.SetSpacing([1.0, 1.0, 3.0])

    try:
        print(f"Writing dummy files to {test_timepoint_dir}...")
        sitk.WriteImage(sitk_img_t1c, str(dummy_t1c_path))
        sitk.WriteImage(sitk_img_t2w, str(dummy_t2w_path))
        sitk.WriteImage(sitk_mask, str(dummy_mask_path))
        print("Dummy files written.")

        # Initialize the processor
        processor = EfficientMRIProcessor(memory_limit_mb=100) # Low limit for testing

        # Test feature extraction
        print("\n--- Testing extract_features_for_patient ---")
        # Pass the specific timepoint directory as data_dir
        extracted_features = processor.extract_features_for_patient(
            patient_id=patient_id,
            data_dir=str(test_timepoint_dir), # Pass the specific directory
            timepoint=timepoint,
            sequences=['t1c', 't2w'] # Specify sequences
        )

        print("\nExtracted Features:")
        import json
        print(json.dumps(extracted_features, indent=2))

        # Test with missing sequence
        print("\n--- Testing with missing sequence ---")
        extracted_features_missing = processor.extract_features_for_patient(
            patient_id=patient_id,
            data_dir=str(test_timepoint_dir),
            timepoint=timepoint,
            sequences=['t1c', 'flair'] # flair doesn't exist
        )
        print("\nExtracted Features (missing flair):")
        print(json.dumps(extracted_features_missing, indent=2))

        # Test with missing mask (should return empty or raise error depending on logic)
        print("\n--- Testing with missing mask ---")
        Path(dummy_mask_path).unlink() # Delete the mask file
        try:
             extracted_features_no_mask = processor.extract_features_for_patient(
                 patient_id=patient_id,
                 data_dir=str(test_timepoint_dir),
                 timepoint=timepoint,
                 sequences=['t1c', 't2w']
             )
             print("\nExtracted Features (no mask):")
             print(json.dumps(extracted_features_no_mask, indent=2))
        except ValueError as e:
             print(f"\nCaught expected error when mask missing: {e}")


    except Exception as e:
        print(f"An error occurred during testing: {e}")
    finally:
        # Clean up dummy files and directories
        print("\nCleaning up test data...")
        import shutil
        if test_base_dir.exists():
            shutil.rmtree(test_base_dir)
            print(f"Removed {test_base_dir}")

