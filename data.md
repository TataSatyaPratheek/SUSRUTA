# MU-Glioma-Post Dataset Documentation

## Overview

The MU-Glioma-Post dataset is a comprehensive collection of medical imaging and clinical data designed to facilitate AI model development for distinguishing radiation necrosis from tumor progression in post-treatment glioma patients. This documentation provides a detailed overview of the dataset structure, contents, and usage guidelines.

## Dataset Purpose and Significance

- **Primary Objective**: Enable AI/ML applications to differentiate between radiation necrosis and tumor progression in post-treatment glioma patients
- **Clinical Relevance**: Addresses a major diagnostic challenge in neuro-oncology where radiation effects and tumor recurrence share overlapping imaging features
- **Research Applications**: Supports development of segmentation algorithms and predictive models for personalized therapy
- **Longitudinal Value**: Enables analysis of tumor evolution and treatment response over time

## Dataset Composition

### Core Statistics

| Metric | Value | Notes |
|--------|-------|-------|
| Patients | 203 | Glioma patients with post-treatment follow-up |
| Timepoints | 617 | Longitudinal MR imaging sessions |
| Source | University of Missouri Hospital | 10-year retrospective data collection |
| Ethics | IRB-approved | Compliant with research ethics standards |

### Data Components

The dataset consists of four key components:

1. **MR Imaging Data** (NIfTI format)
2. **Clinical Data** (Excel spreadsheet)
3. **MR Scanner Metadata** (Excel spreadsheet)
4. **Segmentation Volumes** (Excel spreadsheet)

## Imaging Data Details

### MRI Sequences

Each timepoint includes multiple MRI sequences, each serving a specific diagnostic purpose:

| Suffix | Description | Clinical Use |
|--------|-------------|--------------|
| `t1c` | T1-weighted with contrast | Enhancing tumor/blood vessels visualization |
| `t1n` | T1-weighted without contrast | Baseline anatomy assessment |
| `t2f` | T2-weighted FLAIR | Edema and lesion detection |
| `t2w` | T2-weighted | Fluid/edema visualization |
| `tumorMask` | Binary segmentation mask | Tumor region delineation |

### File Structure

```
MU-Glioma-Post/
├── PatientID_XXXX/
│   ├── Timepoint_1/
│   │   ├── PatientID_XXXX_Timepoint_1_t1c.nii.gz
│   │   ├── PatientID_XXXX_Timepoint_1_t1n.nii.gz
│   │   ├── PatientID_XXXX_Timepoint_1_t2f.nii.gz
│   │   ├── PatientID_XXXX_Timepoint_1_t2w.nii.gz
│   │   └── PatientID_XXXX_Timepoint_1_tumorMask.nii.gz
│   ├── Timepoint_2/
│   │   └── ...
│   └── ...
└── ...
```

### Segmentation Methodology

- **Initial Segmentation**: Automated via nnUNet deep learning pipeline
- **Expert Validation**: Manual refinement by neuroradiologists
- **Quality Control**: Rigorous validation to ensure accurate tumor delineation

## Clinical Data (MUGliomaPost_ClinicalDataFINAL032025.xlsx)

Contains comprehensive patient-level clinical information including:

- **Demographics**: Age, sex, race/ethnicity
- **Clinical History**: Initial diagnosis, tumor grade, histopathology
- **Treatment Details**: Surgery type, radiation dose/planning, chemotherapy regimens
- **Genomics**: Molecular profiling (IDH status, MGMT methylation, 1p/19q co-deletion)
- **Outcome Measures**: Progression status, survival data

## MR Scanner Data (MR_Scanner_data.xlsx)

Contains technical metadata about the imaging equipment and protocols:

- **Scanner Models**: Manufacturer, model, software version
- **Acquisition Parameters**: Field strength, sequence parameters, resolution
- **Quality Metrics**: Signal-to-noise ratio, artifact assessment
- **Protocol Standardization**: Information on harmonization across different scanners

## Segmentation Volumes Data (MUGliomaPost_Segmentation_Volumes.xlsx)

Provides quantitative data extracted from tumor segmentations:

- **Volume Measurements**: Total tumor volume, enhancing portion, edema, necrosis
- **Longitudinal Tracking**: Volume changes across timepoints
- **Shape Features**: 3D morphological characteristics
- **Location Data**: Anatomical coordinates and relations to critical structures

## Data Processing and Working with the Dataset

### Preprocessing Pipeline

All MR images underwent a standardized preprocessing pipeline:

1. Bias field correction
2. Registration to standard space
3. Intensity normalization
4. Quality control

### Visualization Tools

The dataset is optimized for visualization with neuroimaging libraries, particularly **nilearn**:

```python
import os
from nilearn import plotting, image

# Example for loading and visualizing a patient timepoint
base_path = 'path/to/MU-Glioma-Post/PatientID_XXXX/Timepoint_Y'
t1c_path = os.path.join(base_path, [f for f in os.listdir(base_path) if 't1c' in f][0])
mask_path = os.path.join(base_path, [f for f in os.listdir(base_path) if 'tumorMask' in f][0])

# Load images
t1c_img = image.load_img(t1c_path)
mask_img = image.load_img(mask_path)

# Visualize
plotting.plot_roi(mask_img, bg_img=t1c_img, title="Tumor Mask on T1c",
                 display_mode='ortho', cmap='autumn')
```

### AI/ML Applications

The dataset is specifically structured to support:

1. **Segmentation Models**: Training and validation of automated tumor segmentation algorithms
2. **Classification Models**: Differentiating radiation necrosis from tumor progression
3. **Longitudinal Analysis**: Tracking tumor evolution over time with multiple timepoints
4. **Multimodal Integration**: Combining imaging and clinical data for enhanced predictions

## Comparison to Related Datasets

| Feature | MU-Glioma-Post | BraTS Challenge (2024) | TCGA-GBM |
|---------|----------------|------------------------|----------|
| Focus | Post-treatment progression | Segmentation challenge | Pre-treatment genomics |
| Timepoints | 617 (longitudinal) | Single timepoint (2,200) | Pre-operative only |
| Sequences | T1c, T1n, T2w, T2f | T1, T1c, T2, FLAIR | Variable |
| Segmentation | Auto + expert refinement | Multi-institutional manual | Limited annotations |
| Clinical Data | Comprehensive | Basic demographics | TCGA genomic linkage |

## Access and Usage

- **Availability**: Accessible via The Cancer Imaging Archive (TCIA)
- **Registration**: Requires application and approval
- **Format**: Imaging data in NIfTI format; clinical data in Excel spreadsheets
- **Citation**: Please cite as per TCIA guidelines when using this dataset

## Conclusion

The MU-Glioma-Post dataset represents a significant contribution to the field of neuro-oncology imaging and AI-assisted diagnosis. Its comprehensive longitudinal design, expert-refined segmentations, and rich clinical metadata make it an invaluable resource for developing and validating advanced AI models for post-treatment glioma assessment.

## References

1. University of Missouri Hospital Neuro-Oncology Department
2. The Cancer Imaging Archive (TCIA)
3. nnUNet segmentation pipeline documentation