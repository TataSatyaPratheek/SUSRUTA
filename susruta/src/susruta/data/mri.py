# susruta/src/susruta/data/mri.py
"""
Memory-efficient MRI processing for brain tumor image analysis.

Includes optimized feature extraction and region-of-interest processing.
"""

import os
from typing import Dict, List, Tuple, Any, Optional
import gc
import numpy as np
import SimpleITK as sitk

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
        reader = sitk.ImageFileReader()
        reader.SetFileName(file_path)
        reader.ReadImageInformation()
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
        mask = sitk.ReadImage(mask_path)

        # Convert to binary mask if needed
        if mask.GetPixelID() != sitk.sitkUInt8:
            # Use BinaryThreshold instead of direct comparison
            binary_mask = sitk.BinaryThreshold(mask, lowerThreshold=0, upperThreshold=0,
                                            insideValue=0, outsideValue=1)
        else:
            binary_mask = mask

        # Use label statistics to get bounding box efficiently
        label_stats = sitk.LabelStatisticsImageFilter()
        label_stats.Execute(binary_mask, binary_mask)

        # Check if any non-zero voxels exist
        if not label_stats.HasLabel(1):
            # Return full image bounds if mask is empty
            size = mask.GetSize()
            return ([0, 0, 0], [size[0]-1, size[1]-1, size[2]-1])

        # sitk bbox format: [xStart, yStart, zStart, xSize, ySize, zSize]
        bbox = label_stats.GetBoundingBox(1)

        # Convert to min/max coordinates format
        # --- Start Fix: Correct index extraction for bbox_min ---
        bbox_min = [bbox[0], bbox[1], bbox[2]] # xStart, yStart, zStart
        # --- End Fix ---
        bbox_max = [bbox[0] + bbox[3] - 1, bbox[1] + bbox[4] - 1, bbox[2] + bbox[5] - 1] # start + size - 1

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

        # Get bounding box of tumor with margin
        bbox_min, bbox_max = self.compute_bounding_box(mask_path)
        margin = 5  # Add 5 voxel margin around tumor

        # Get image metadata
        img_info = self.load_nifti_metadata(img_path)
        size = img_info['size']

        # Add margin and clamp to image bounds
        bbox_min = [max(0, x - margin) for x in bbox_min]
        bbox_max = [min(s-1, x + margin) for x, s in zip(bbox_max, size)]

        # Calculate ROI size
        roi_size = [bbox_max[i] - bbox_min[i] + 1 for i in range(3)]

        # Estimate memory requirements
        voxel_bytes = 4  # 32-bit float
        roi_memory_mb = np.prod(roi_size) * voxel_bytes / (1024 * 1024)

        self.memory_tracker.log_memory(f"ROI calculation (size: {roi_size}, memory: {roi_memory_mb:.2f}MB)")

        # Load ROI using SimpleITK's efficient region extraction
        img = sitk.ReadImage(img_path)
        mask = sitk.ReadImage(mask_path)

        features = {}

        if roi_memory_mb < self.memory_limit_mb * 0.5:
            # Can process ROI at once
            roi_img = sitk.RegionOfInterest(img, roi_size, bbox_min)
            roi_mask = sitk.RegionOfInterest(mask, roi_size, bbox_min)

            # Extract features
            features = self._compute_roi_features(roi_img, roi_mask)
            self.memory_tracker.log_memory("Feature extraction complete")

            # Clean up
            del roi_img, roi_mask
            gc.collect()
        else:
            # Process in chunks along Z-axis
            chunk_features = []
            chunk_count = max(1, int(np.ceil(roi_memory_mb / (self.memory_limit_mb * 0.3))))
            chunk_size_z = int(np.ceil(roi_size[2] / chunk_count))

            for chunk_idx in range(chunk_count):
                z_start = bbox_min[2] + chunk_idx * chunk_size_z
                z_end = min(bbox_min[2] + (chunk_idx + 1) * chunk_size_z - 1, bbox_max[2])

                chunk_bbox_min = [bbox_min[0], bbox_min[1], z_start]
                chunk_roi_size = [roi_size[0], roi_size[1], z_end - z_start + 1]

                # Extract chunk
                chunk_img = sitk.RegionOfInterest(img, chunk_roi_size, chunk_bbox_min)
                chunk_mask = sitk.RegionOfInterest(mask, chunk_roi_size, chunk_bbox_min)

                # Compute features for chunk
                chunk_features.append(self._compute_roi_features(chunk_img, chunk_mask))
                self.memory_tracker.log_memory(f"Chunk {chunk_idx+1}/{chunk_count} processed")

                # Explicitly clean up
                del chunk_img, chunk_mask
                gc.collect()

            # Combine chunk features
            features = self._combine_chunk_features(chunk_features)
            self.memory_tracker.log_memory("All chunks combined")

        # Final cleanup
        del img, mask
        gc.collect()

        return features

    def _compute_roi_features(self, img: sitk.Image, mask: sitk.Image) -> Dict[str, float]:
        """
        Compute radiomics features from an ROI.

        Args:
            img: SimpleITK image
            mask: SimpleITK mask

        Returns:
            Dictionary of computed features
        """
        # Convert to numpy array for processing - only for masked region
        img_array = sitk.GetArrayFromImage(img)
        mask_array = sitk.GetArrayFromImage(mask)

        # Calculate first-order statistics on masked region
        masked_img = img_array[mask_array > 0]

        if len(masked_img) == 0:
            # No tumor voxels in this region
            return {
                'mean': 0.0, # Use float
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'p25': 0.0,
                'p50': 0.0,
                'p75': 0.0,
                'volume_voxels': 0.0 # Use float for consistency
            }

        features = {
            'mean': float(np.mean(masked_img)),
            'std': float(np.std(masked_img)),
            'min': float(np.min(masked_img)),
            'max': float(np.max(masked_img)),
            'p25': float(np.percentile(masked_img, 25)),
            'p50': float(np.percentile(masked_img, 50)),
            'p75': float(np.percentile(masked_img, 75)),
            'volume_voxels': float(len(masked_img)) # Use float
        }

        return features

    def _combine_chunk_features(self, chunk_features: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Combine features from multiple chunks.

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

        # --- Start Fix: Calculate min/max only from non-empty chunks ---
        combined['min'] = min(f['min'] for f in non_empty_chunks)
        combined['max'] = max(f['max'] for f in non_empty_chunks)
        # --- End Fix ---
        combined['volume_voxels'] = float(total_volume)

        # Weighted statistics using only non-empty chunks
        for stat in ['mean', 'p25', 'p50', 'p75']:
            combined[stat] = sum(f[stat] * f['volume_voxels'] for f in non_empty_chunks) / total_volume

        # For standard deviation, combine variances using only non-empty chunks
        combined_mean = combined['mean']
        sum_of_squares_term = sum(
            f['volume_voxels'] * (f['std']**2 + f['mean']**2)
            for f in non_empty_chunks # Iterate over non_empty_chunks
        )
        combined_e_x2 = sum_of_squares_term / total_volume
        combined_variance = combined_e_x2 - combined_mean**2

        # Ensure variance is non-negative due to potential floating point errors
        combined_variance = max(0, combined_variance)
        combined['std'] = np.sqrt(combined_variance)

        return combined

    def extract_features_for_patient(self,
                                 patient_id: int,
                                 data_dir: str,
                                 timepoint: int = 1,
                                 sequences: Optional[List[str]] = None # Add optional argument
                                 ) -> Dict[str, Dict[str, float]]:
        """
        Extract features from all MRI sequences for a patient timepoint.

        Args:
            patient_id: Patient identifier
            data_dir: Base directory for data
            timepoint: Timepoint number
            sequences: Optional list of sequences to process (e.g., ['t1c', 't2w'])
                    Defaults to ['t1c', 't1n', 't2f', 't2w'].

        Returns:
            Dictionary of features for each sequence
        """
        timepoint_dir = os.path.join(data_dir, f'PatientID_{patient_id:04d}', f'Timepoint_{timepoint}') # Adjusted path format

        if not os.path.exists(timepoint_dir):
            # Use f-string for error message consistency
            raise ValueError(f"Data directory not found for patient {patient_id}, timepoint {timepoint}: {timepoint_dir}")

        # Define default sequences if none provided
        if sequences is None:
            sequences_to_process = ['t1c', 't1n', 't2f', 't2w']
        else:
            sequences_to_process = sequences

        # Get file paths for specified sequences using glob for flexibility
        import glob # Add import
        sequence_files = {}
        for sequence in sequences_to_process:
            # Use glob to find files matching the pattern
            # Adjust pattern based on actual filename conventions
            pattern = os.path.join(timepoint_dir, f'*{sequence}*.nii*') # More robust pattern
            matches = glob.glob(pattern)
            if matches:
                # Handle multiple matches if necessary, here we take the first
                sequence_files[sequence] = matches[0]
            # else:
                # Optionally log or warn if a requested sequence is not found
                # print(f"Warning: Sequence '{sequence}' not found for patient {patient_id}, timepoint {timepoint}")


        # Get tumor mask path using glob
        mask_pattern = os.path.join(timepoint_dir, '*tumorMask*.nii*')
        mask_files = glob.glob(mask_pattern)
        if not mask_files:
            raise ValueError(f"No tumor mask found for patient {patient_id}, timepoint {timepoint} in {timepoint_dir}")

        mask_path = mask_files[0] # Take the first match

        # Extract features for each found sequence
        features = {}
        for sequence, file_path in sequence_files.items():
            print(f"Processing {sequence} for patient {patient_id}, timepoint {timepoint}")
            try:
                sequence_features = self.extract_roi_features(file_path, mask_path)
                features[sequence] = sequence_features
            except Exception as e:
                print(f"Error processing {sequence} for patient {patient_id}: {e}")
                # Decide how to handle errors: skip sequence, return partial results, or raise

        # Get tumor volume and shape features from mask
        try:
            features['tumor'] = self._extract_tumor_features(mask_path)
        except Exception as e:
            print(f"Error processing tumor mask for patient {patient_id}: {e}")
            # Decide how to handle errors

        return features

    def _extract_tumor_features(self, mask_path: str) -> Dict[str, float]:
        """
        Extract shape and volume features from tumor mask.

        Args:
            mask_path: Path to tumor mask NIfTI file

        Returns:
            Dictionary of tumor features
        """
        mask = sitk.ReadImage(mask_path)

        # Use label shape statistics for efficient feature extraction
        shape_stats = sitk.LabelShapeStatisticsImageFilter()
        shape_stats.Execute(mask)

        if not shape_stats.HasLabel(1):
            return {
                'volume_mm3': 0.0, # Use float
                'surface_area': 0.0,
                'elongation': 0.0,
                'roundness': 0.0
            }

        # Get physical space measurements
        volume = shape_stats.GetPhysicalSize(1)
        surface = shape_stats.GetPerimeter(1)  # Note: Perimeter might not be the best 3D surface area measure
        elongation = shape_stats.GetElongation(1)
        roundness = shape_stats.GetRoundness(1)

        features = {
            'volume_mm3': float(volume),
            'surface_area': float(surface),
            'elongation': float(elongation),
            'roundness': float(roundness)
        }

        return features
