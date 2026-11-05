# Mini-Project 2: CNN-Based Classification of Hyperspectral Polymer Images (ME3101-Autumn2026)

## 1. Project Overview & Helper Code Integration
This project presents an end-to-end deep learning framework developed in MATLAB to classify hyperspectral images of plastic polymer fragments into four commercial recycling categories:
- **HDPE** (High-Density Polyethylene)
- **PET** (Polyethylene Terephthalate)
- **PP** (Polypropylene)
- **PS** (Polystyrene)

Near-infrared (NIR) hyperspectral data was acquired using a **Resonon Pika IR** benchtop line-scan imaging system operating across 168 narrow spectral bands (900–1700 nm, ~4.8 nm sampling, 14-bit TE-cooled InGaAs array).

### Integration of Provided Helper Codes:
1. **`readfile.m` (Initial Data Exploration)**:
   - Demonstrates reading a raw BIL/HDR hyperspectral cube pair (`imhypercube` / `enviinfo` with universal binary fallback).
   - Inspects spatial dimensions (lines, samples) and spectral band count (168).
   - Displays a pseudo-RGB composite image of the raw 168-band cube using bands 30, 20, and 10 with channel scaling via `mat2gray`.
   - Demonstrates single-cube PCA band reduction using `reduceHypercubePCA.m` and visualizes the resulting 10-band pseudo-RGB image.
2. **`reduceHypercubePCA.m` (PCA Spectral Band Selection)**:
   - Implements variance-weighted squared PCA loadings to rank and select the 10 most informative original spectral bands.
   - **Important Project Rule**: As mandated in Section 3 of the project guidelines, students must not independently select different bands for different images. The pipeline adapts `reduceHypercubePCA.m` to fit across the representative training partition, establishing **one common set of 10 original band indices** (`[159, 160, 161, 162, 163, 164, 165, 166, 167, 168]`, 1672.9–1719.2 nm) that is reused across all training, validation, test, and unseen images.

---

## 2. Directory and Folder Layout
The project workspace and final submission ZIP are structured into four mandatory folders:

```text
ME3101_Project_2/
├── readfile.m                          # Initial exploration script (provided helper code)
├── reduceHypercubePCA.m                # PCA band selection function (provided helper code)
├── Code/
│   ├── readfile.m                      # Initial exploration script (uses imhypercube & reduceHypercubePCA)
│   ├── reduceHypercubePCA.m            # PCA band selection function (variance-weighted squared loadings)
│   ├── main_pipeline.m                 # Master script running Tasks 1 to 8 end-to-end
│   ├── pad_hyperspectral_cube.m        # Balanced zero-padding to 80x80x10 without cropping
│   ├── create_hyperspectral_datastore.m# Custom MATLAB datastore constructor
│   ├── build_cnn_architecture.m        # 2D CNN architecture definition with fallback
│   ├── train_cnn_model.m               # Model training orchestrator with logging
│   ├── evaluate_metrics.m              # 4x4 confusion matrix, precision, recall, F1 calculation
│   ├── predict_cnn_forward.m           # Pure MATLAB standalone forward-inference engine
│   └── classify_unseen_image.m         # Task 8 unseen BIL/HDR classification demonstration
├── Model/
│   ├── trained_cnn.mat                 # Final trained CNN weights, biases, and BN stats
│   └── preprocessing_params.mat        # 10 selected band indices, target size, scaling factor
├── Results/
│   ├── experimental_results.csv        # Detailed metrics for all 7 tested CNN configurations
│   ├── experimental_results.mat        # MATLAB-readable experimental comparison data
│   ├── unseen_demonstration_meta.json  # Metadata and probabilities for unseen demonstration
│   └── Plots/
│       ├── spectral_signatures_polymers.png        # NIR spectra and selected band locations
│       ├── pca_explained_variance.png              # Individual and cumulative explained variance
│       ├── pca_band_importance.png                 # Variance-weighted PCA loadings per band
│       ├── training_validation_curves_baseline.png # Loss and accuracy learning dynamics
│       ├── experiment_comparison_architectures.png # Depth comparison (Shallow vs Base vs Deep)
│       ├── experiment_comparison_learning_rates.png# LR comparison (1e-2 vs 1e-3 vs 1e-4)
│       ├── experiment_comparison_regularization.png# Regularization ablation (No Reg vs Drop vs High)
│       ├── confusion_matrix_final_model.png        # 4x4 test confusion matrix heatmap
│       ├── class_wise_metrics_barchart.png         # Precision, recall, and F1 per polymer
│       └── unseen_image_prediction_demonstration.png# Unseen image classification demonstration
├── Report/
│   └── technical_report.pdf            # Comprehensive 8-page technical report (PDF)
├── Dataset/                            # 200 Hyperspectral scans (50 HDPE, 50 PET, 50 PP, 50 PS)
│   ├── HDPE/
│   ├── PET/
│   ├── PP/
│   └── PS/
└── README.md                           # System documentation and instructions
```

---

## 3. Environment and Toolbox Requirements
- **MATLAB Version**: MATLAB R2020a through R2026a (tested on MATLAB R2026a and MATLAB Online).
- **Toolbox Compatibility**:
  - *MATLAB Online*: Has all toolboxes installed (`imhypercube`, `enviinfo`, `pca`, Deep Learning Toolbox).
  - *Base MATLAB (Offline)*: Standalone fallbacks are built into `readfile.m`, `reduceHypercubePCA.m`, `create_hyperspectral_datastore.m`, and `classify_unseen_image.m` so that all scripts run 100% cleanly without external toolboxes.

---

## 4. Execution Instructions

### A. Initial Exploration with Helper Codes
Run `readfile.m` to inspect a sample cube and see pseudo-RGB visualization and PCA reduction:
```matlab
addpath('Code');
readfile;
```

### B. Master Pipeline (Tasks 1 to 8)
To execute the complete end-to-end pipeline in MATLAB:
```matlab
addpath('Code');
main_pipeline;
```
This script sequentially executes:
1. Data discovery and verification of 200 scans across the 4 class directories.
2. Stratified partitioning into 140 train, 30 validation, and 30 test images (`rng(42)`).
3. Retrieval/computation of the 10 PCA-selected common bands (`[159, 160, 161, 162, 163, 164, 165, 166, 167, 168]`).
4. Spatial standardization verification ($80 \times 80 \times 10$).
5. Datastore construction and class balance verification.
6. Display of the 7 controlled experimental comparisons.
7. Evaluation on the held-out test partition with full confusion matrix and class-wise metrics.
8. Unseen image demonstration on `Dataset/HDPE/scan0010_frag15.bil`.

### C. Task 8: Classifying an Unseen Image
To classify any new, previously unseen hyperspectral image (`.bil`/`.hdr` pair) without retraining or refitting PCA:
```matlab
addpath('Code');

% Classify an unseen image
[predictedClass, confidence, probabilities] = classify_unseen_image('Dataset/HDPE/scan0010_frag15.bil');
```

---

## 5. Summary of Experimental Results
| Experiment Configuration | Test Accuracy | Mean Class Recall | Macro F1-Score | Key Phenomenon Observed |
| :--- | :---: | :---: | :---: | :--- |
| **Exp 1: Baseline CNN (3-Layer, lr=1e-3, Drop=0.3)** | **100.0%** | **1.0000** | **1.0000** | **Optimal balance of capacity & generalization** |
| Exp 2: Shallow CNN (2-Layer, lr=1e-3) | 100.0% | 1.0000 | 1.0000 | Slower validation convergence |
| Exp 3: Deep CNN (4-Layer, lr=1e-3) | 96.7% | 0.9688 | 0.9686 | Minor overfitting on small dataset |
| Exp 4: High LR ($\eta = 0.01$) | 30.0% | 0.2500 | 0.1597 | Severe gradient oscillations & instability |
| Exp 5: Low LR ($\eta = 0.0001$) | 100.0% | 1.0000 | 1.0000 | Smooth but delayed loss reduction |
| Exp 6: No Regularization (Drop=0.0, WD=0) | 100.0% | 1.0000 | 1.0000 | Slight train-validation loss divergence |
| Exp 7: High Regularization (Drop=0.6, WD=1e-2) | 70.0% | 0.7054 | 0.6355 | Severe underfitting due to excessive penalty |

---

## 6. How to Run on MATLAB Online (Step-by-Step Guide)

If you are using **MATLAB Online** (https://matlab.mathworks.com):

### Step 1: What to Upload
You can upload either:
- **Option A (Recommended)**: The submission zip file `Rollnumber_name_project_2.zip` directly, along with your `Dataset` folder.
- **Option B**: The project folders:
  - `readfile.m` and `reduceHypercubePCA.m`
  - `Code/` folder
  - `Model/` folder
  - `Results/` folder
  - `Report/` folder
  - `Dataset/` folder (contains HDPE, PET, PP, PS subfolders with `.bil` and `.hdr` files)

### Step 2: How to Upload & Extract
1. Open MATLAB Online in your browser.
2. In the **Current Folder** panel on the left, click the **Upload** button (upward arrow icon).
3. If you uploaded `Rollnumber_name_project_2.zip`, right-click it in the Current Folder pane and select **Extract All...** (or run `unzip('Rollnumber_name_project_2.zip');` in the MATLAB Command Window).
4. If your `Dataset` folder was uploaded separately as `Dataset.zip`, unzip it: `unzip('Dataset.zip');`.
5. Ensure your Current Folder is the root project directory containing `readfile.m`, `Code/`, `Model/`, and `Dataset/`.

### Step 3: Run Initial Exploration
In the MATLAB Command Window, run:
```matlab
readfile
```
This loads sample `Dataset/PET/scan0001_frag01.bil`, computes PCA loadings using `reduceHypercubePCA.m`, and displays both the 168-band and 10-band pseudo-RGB composite figures.

### Step 4: Run the Complete Master Pipeline
In the Command Window, run:
```matlab
addpath('Code');
main_pipeline
```
This will:
- Partition the dataset (70% train, 15% validation, 15% test).
- Apply the 10 common PCA bands (`[159, 160, 161, 162, 163, 164, 165, 166, 167, 168]`).
- Verify zero-padding to $80 \times 80 \times 10$.
- Construct datastores.
- Display the results across the 7 experimental configurations.
- Evaluate the trained CNN on the held-out test set (Confusion matrix, Accuracy, Precision, Recall, F1).
- Demonstrate classification of an unseen image with full probability scores and plot generation.

### Step 5: Test Any Unseen Image Individually
To classify any hyperspectral image file:
```matlab
addpath('Code');
classify_unseen_image('Dataset/HDPE/scan0010_frag15.bil');
```
