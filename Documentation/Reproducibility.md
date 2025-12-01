# Reproducibility Measures

This document outlines all steps taken to ensure the project is fully reproducible.

---

## 1. Environment Management

### 1.1 Virtual Environment Isolation

All dependencies are installed in an isolated Python virtual environment to avoid conflicts with system packages. The
`run_pipeline.py` script automatically creates and manages the virtual environment - no manual setup required.

### 1.2 Pinned Dependencies

All Python packages are pinned to exact versions in `requirements.txt`:

```
torch==2.9.0
torchvision==0.24.0
numpy==2.2.6
scikit-image==0.25.2
opencv-python==4.12.0.88
networkx==3.5
pandas==2.3.3
matplotlib==3.10.7
xlsxwriter==3.2.9
seaborn==0.13.2
openpyxl==3.1.2
```

This ensures anyone can recreate the exact same environment.

### 1.3 Environment Setup Module

`setup_environment.py` provides reusable environment management:

- Checks Python version requirements (3.10+)
- Creates virtual environment if not exists
- Installs dependencies from `requirements.txt`
- Can re-launch scripts in the virtual environment

```python
from Runners.setup_environment import setup_environment, relaunch_in_venv

python_path, in_venv = setup_environment()
if not in_venv:
    relaunch_in_venv(python_path)
```

---

## 2. Version Control with Git

### 2.1 Commit Practices

- Meaningful commit messages describing changes
- Logical commits representing single units of work
- Feature branches for development, merged to main after testing

### 2.2 Example Commit History

```
2c14402 Merge pull request #1 from vessel-segmentation-script
651a572 refactor: raw data to solution folder
9596f51 feat: predict script
8c2cd93 add: challenges to readme as notes
671c691 fix: dataset load processing bug fix
```

### 2.3 .gitignore

Large files and generated outputs are excluded from version control:

- `venv/` - Virtual environment
- `__pycache__/` - Python cache
- `.ipynb_checkpoints/` - Jupyter checkpoints

---

## 3. Pipeline Automation

### 3.1 Automation Scripts

All automation scripts are located in the `Runners/` directory:

| Script                         | Purpose                                                                          |
|--------------------------------|----------------------------------------------------------------------------------|
| `Runners/run_pipeline.py`      | Main pipeline runner - handles environment setup and all pipeline stages         |
| `Runners/oneclick.py`          | One-click reproduction - downloads outputs, runs pipeline, executes EDA notebook |
| `Runners/setup_environment.py` | Environment setup module - creates venv and installs dependencies                |

### 3.2 Single Entry Point

`Runners/run_pipeline.py` provides a single command to execute the entire pipeline:

```bash
python Runners/run_pipeline.py
```

### 3.3 One-Click Reproduction

`Runners/oneclick.py` enables complete reproduction with a single command:

```bash
python Runners/oneclick.py
```

This script will:

1. Download pre-computed outputs from the provided URL
2. Extract and place files in correct locations
3. Run the pipeline (creates venv, installs dependencies, skips existing outputs)
4. Install Jupyter kernel for the virtual environment
5. Execute the EDA notebook
6. Launch Jupyter for interactive exploration

### 3.4 Idempotent Execution with Checkpoints

The pipeline checks for existing outputs before re-running each stage, preventing unnecessary recomputation:

```python
def run_pipeline():
    # Check if preprocessing already done
    if not os.path.exists('Data/preprocessed/train/image'):
        print("Running data preprocessing...")
        process_data(project_root)
    else:
        print("Preprocessed data found, skipping...")

    # Check if model already trained
    if not os.path.exists('Model/unet_checkpoint.pth.tar'):
        print("Training model...")
        train_model(project_root)
    else:
        print("Trained model found, skipping...")

    # Check if predictions exist
    if not os.path.exists('Data/outputs/predictions'):
        print("Running predictions...")
        run_prediction(project_root)
    else:
        print("Predictions found, skipping...")

    # Feature extraction
    run_feature_extraction(project_root)
```

### 3.5 Standalone Scripts

Each pipeline stage can also be run independently:

```bash
python Scripts/data_preprocessing.py  # Step 1: Preprocess data
python Scripts/train.py               # Step 2: Train model
python Scripts/predict.py             # Step 3: Run inference
python Scripts/feature_extraction/feature_extraction.py  # Step 4: Extract features
```

---

## 4. Code Organization

### 4.1 Modular Architecture

Code is organized into logical modules:

```
Scripts/
├── constants.py             # Centralized hyperparameters
├── data_preprocessing.py    # Data preparation
├── train.py                 # Model training
├── predict.py               # Inference
├── unet_model.py            # Network architecture
├── dataset/                 # PyTorch Dataset classes
│   ├── vessel_dataset.py
│   └── test_vessel_dataset.py
└── feature_extraction/      # Graph-based analysis
    ├── feature_extraction.py
    ├── connectivity_matrix_test.py
    └── utility.py
```

### 4.2 Centralized Configuration

All hyperparameters are centralized in `Scripts/core/constants.py`:

```python
# Random seed for reproducibility
SEED = 42

# Training
LEARNING_RATE = 1e-5
BATCH_SIZE = 8  # Reduce to 2-4 if GPU memory < 8GB
NUM_EPOCHS = 7
NUM_WORKERS = 2  # DataLoader workers
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
```

### 4.4 Deterministic Training

Random seeds are set automatically on import for full reproducibility:

```python
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
```

This ensures identical results across runs when using the same seed.

### 4.3 Parameterized Functions

Functions accept path parameters for flexibility:

```python
def train_model(project_root: str):
    train_img_dir = os.path.join(project_root, 'Data/preprocessed/train/image')
    ...


def run_prediction(base_path: str):
    model_path = os.path.join(base_path, 'Model/unet_checkpoint.pth.tar')
    ...
```

---

## 5. Documentation

### 5.1 README.md

Comprehensive project documentation including:

- Requirements and dependencies
- Project flow explanation
- Model architecture details
- Dataset class documentation
- Automation scripts documentation
- Troubleshooting guide
- Expected results

### 5.2 Documentation/EDA.md

Detailed documentation of EDA functions:

- Function descriptions and usage
- Feature explanations
- Biological interpretation guide
- Quick start examples

### 5.3 Code Comments and Docstrings

All functions include docstrings:

```python
def create_patches(image, mask, patch_size, stride, output_path_image, output_path_mask, name):
    """
    Extract overlapping patches from image-mask pairs.

    Args:
        image: Input image array
        mask: Ground truth mask array
        patch_size: Size of patches (default 512)
        stride: Overlap stride (default 256)
        ...
    """
```

---

## 6. Pre-computed Outputs (Downloadable)

To enable reproduction without GPU or long training times, we provide pre-computed outputs as a single downloadable zip
file:

**Download URL**: `https://placeholder-link.com/precomputed_outputs.zip`

### 6.1 Trained Model

**File**: `Model/unet_checkpoint.pth.tar`

Contains:

- Model state dict (weights)
- Optimizer state dict
- Achieved Dice score: 0.936

Usage:

```python
checkpoint = torch.load('Model/unet_checkpoint.pth.tar')
model.load_state_dict(checkpoint['state_dict'])
```

### 6.2 Predictions

**Location**: `Data/outputs/predictions/`

Pre-computed binary segmentation masks for all test images (P2-P7).

Format: `.tif` files matching input filenames

### 6.3 Extracted Features

**Location**: `Data/outputs/features/`

For each image patch:

- `{filename}_alldata.xlsx` - Vessel segment features (length, width, tortuosity)
- `{filename}_degreedata.xlsx` - Node features (degree, distance from center)
- `{filename}_network.png` - Network visualization

---

## 7. Reproducible EDA

### 7.1 Jupyter Notebook

**File**: `eda_angio_net.ipynb`

All EDA is conducted in a reproducible notebook with:

- Function-based analysis (reusable)
- Markdown documentation
- Labeled visualizations
- Recorded decisions

### 7.2 EDA Functions

```python
# Single file analysis
run_full_eda(base_path)

# Developmental comparison (P2-P7)
run_developmental_analysis(file_paths)

# Individual analysis functions
load_feature_data(base_path)
plot_morphology_distributions(segments_df)
plot_node_degree_analysis(nodes_df)
analyze_tortuosity(segments_df)
detect_and_plot_outliers(segments_df)
```

### 7.3 Jupyter Kernel Installation

The `oneclick.py` script automatically installs a Jupyter kernel for the virtual environment:

```python
# Kernel name: 'vessel-segmentation'
# Display name: 'Vessel Segmentation (venv)'
```

---

## 8. Data Organization

### 8.1 Immutable Raw Data

Original images are stored separately and never modified:

```
Data/raw_data/
├── train/
│   ├── retina.tif
│   └── retina groundtruth.tif
├── validation/
│   ├── retina.tif
│   └── retina groundtruth.tif
└── test/
    └── *.tif  # P2-P7 images
```

### 8.2 Generated Outputs

All generated files are stored in separate directories:

```
Data/
├── preprocessed/     # Patches (generated)
└── outputs/
    ├── predictions/  # Model outputs (generated)
    └── features/     # Extracted features (generated)
```

---

## 9. Quick Reproduction Guide

### Option A: One-Click Reproduction (Recommended)

```bash
# 1. Clone repository
git clone <repository-url>
cd project

# 2. Run one-click script (downloads outputs, runs pipeline, executes EDA)
python Runners/oneclick.py
```

### Option B: Full Pipeline

```bash
# 1. Clone repository
git clone <repository-url>
cd project

# 2. Run complete pipeline (automatically creates venv and installs dependencies)
python Runners/run_pipeline.py
```

### Option C: Use Pre-computed Outputs (no GPU needed)

```bash
# 1. Clone repository
git clone <repository-url>
cd project

# 2. Download and extract pre-computed outputs
# (or use Runners/oneclick.py which does this automatically)

# 3. Run pipeline - it will skip training since model already exists
python Runners/run_pipeline.py

# Pre-computed outputs already available:
#    - Model/unet_checkpoint.pth.tar (trained model)
#    - Data/outputs/predictions/ (segmentation masks)
#    - Data/outputs/features/ (extracted features)

# 4. Run EDA notebook
jupyter notebook eda_angio_net.ipynb
```

### Option D: Run Individual Steps

```bash
# Skip to any step using pre-computed outputs from previous steps
python Scripts/predict.py  # Uses pre-trained model
python Scripts/feature_extraction/feature_extraction.py  # Uses predictions
```

---

## 10. Directory Structure

```
.
├── Data/
│   ├── raw_data/                # Original images (immutable)
│   │   ├── train/
│   │   ├── validation/
│   │   └── test/
│   ├── preprocessed/            # Generated patches
│   └── outputs/
│       ├── predictions/         # Model outputs
│       └── features/            # Extracted features
├── Model/
│   └── unet_checkpoint.pth.tar  # Trained model weights
├── Scripts/
│   ├── constants.py             # Centralized hyperparameters
│   ├── data_preprocessing.py
│   ├── train.py
│   ├── predict.py
│   ├── unet_model.py
│   ├── dataset/
│   │   ├── vessel_dataset.py
│   │   └── test_vessel_dataset.py
│   └── feature_extraction/
│       ├── feature_extraction.py
│       ├── connectivity_matrix_test.py
│       └── utility.py
├── Runners/
│   ├── run_pipeline.py          # Main pipeline runner
│   ├── oneclick.py              # One-click reproduction script
│   └── setup_environment.py     # Environment setup module
├── Documentation/
│   ├── Reproducibility.md
│   ├── EDA.md
│   └── Results.md
├── eda_angio_net.ipynb          # EDA notebook
└── requirements.txt             # Pinned dependencies
```

---

## 11. Summary of Reproducibility Features

| Feature                   | Implementation                                          |
|---------------------------|---------------------------------------------------------|
| Environment Isolation     | Python venv + requirements.txt                          |
| Dependency Pinning        | Exact versions in requirements.txt                      |
| Environment Setup Module  | setup_environment.py for reusable venv management       |
| Version Control           | Git with meaningful commits                             |
| Single Entry Point        | run_pipeline.py                                         |
| One-Click Reproduction    | oneclick.py (download, extract, run, execute EDA)       |
| Idempotent Execution      | Checkpoint detection before each stage                  |
| Modular Code              | Standalone scripts with parameterized functions         |
| Centralized Configuration | Scripts/constants.py (no hardcoding)                    |
| Documentation             | README.md, EDA.md, Reproducibility.md, docstrings       |
| Pre-computed Outputs      | Model, predictions, features in single downloadable zip |
| Reproducible EDA          | Jupyter notebook with reusable functions                |
| Jupyter Kernel Setup      | Automatic kernel installation for venv                  |
| Data Organization         | Immutable raw data, separate generated outputs          |
| Platform Independence     | Works on Windows, macOS, and Linux                      |

## 12. Reproducibility Checkpoints
- [x] Data published and downloadable ([Release on GitHub]())
- [x] Models published and downloadable ([Release on GitHub]())
- [x] Source code published and downloadable ([GitHub](https://github.com/yashas-hm-unc/AngioNet))
- [x] Dependencies set up in a single command ([setup_environment.py](../Runners/setup_environment.py))
- [x] Key analysis details recorded ([Results.md](./Results.md)) 
- [x] Analysis components set to deterministic ([set_seed](../Scripts/core/constants.py))
- [x] Entire analysis reproducible with a single command ([run_pipeline.py](../Runners/run_pipeline.py))
- [x] Entire workflow can be run with a single command ([oneclick.py](../Runners/oneclick.py))