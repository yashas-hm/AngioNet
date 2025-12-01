# AngioNet
A deep learning pipeline for mouse retinal vessel segmentation using U-Net architecture.

## Requirements

- **Python**: 3.10+ (developed with Python 3.14)
- **CUDA**: 11.8+ (optional but recommended for GPU acceleration)
- **GPU Memory**: 8GB+ recommended for batch size 8; reduce batch size for smaller GPUs
- **Disk Space**: ~2GB for dependencies, ~500MB for data and outputs
- **OS**: Windows, macOS, or Linux

### Key Dependencies

| Package       | Purpose                                 |
|---------------|-----------------------------------------|
| torch         | Deep learning framework                 |
| torchvision   | Image transforms and utilities          |
| numpy         | Numerical computing                     |
| scikit-image  | Image processing (skeletonization, I/O) |
| opencv-python | Computer vision operations              |
| networkx      | Graph-based vessel network analysis     |
| pandas        | Data manipulation and export            |
| matplotlib    | Visualization                           |
| xlsxwriter    | Excel export for features               |

All dependencies with exact versions are pinned in `requirements.txt`.

## Quick Start

### Run Complete Pipeline

```bash
python Runners/run_pipeline.py
```

This automatically creates a virtual environment, installs dependencies, and runs all pipeline stages.

### One-Click Reproduction

```bash
python Runners/oneclick.py
```

Downloads pre-computed outputs, runs pipeline, and executes EDA notebook.

### Run Individual Steps

```bash
python Scripts/data_preprocessing.py  # Preprocess raw data
python Scripts/train.py               # Train the model
python Scripts/predict.py             # Run inference
python Scripts/feature_extraction/feature_extraction.py  # Extract features
```

## Documentation

| Document                                                     | Description                                    |
|--------------------------------------------------------------|------------------------------------------------|
| **[Reproducibility.md](./Documentation/Reproducibility.md)** | Detailed reproducibility measures and guide    |
| **[EDA.md](./Documentation/EDA.md)**                         | EDA functions and vessel feature documentation |
| **[Results.md](./Documentation/Results.md)**                 | Results and Key Findings                       |

## Pre-computed Outputs (Download)

To reproduce results without training, download the pre-computed outputs:

**[Download Pre-computed Outputs](https://github.com/yashas-hm-unc/AngioNet/releases/download/v1.0.0/precomputed_outputs.zip)**

The zip contains:

- `Model/unet_checkpoint.pth.tar` - Trained U-Net checkpoint (Dice: 0.936)
- `Data/outputs/predictions/` - Segmentation masks for P2-P7 test images
- `Data/outputs/features/` - Vessel network features (xlsx + png)

Extract the zip to the project root directory to place files in their correct locations.


## Project Flow

```
Data Preprocessing → Training → Prediction → Feature Extraction
```

### 1. Data Preprocessing (`Scripts/data_preprocessing.py`)

Prepares raw retina images for training by:

- Converting RGB images to grayscale
- Normalizing pixel values to float [0, 1]
- Creating overlapping patches (512x512) with configurable stride
- Filtering out patches with insufficient vessel content (<5% vessel pixels)
- Handling inverted masks between train/validation datasets

### 2. Training (`Scripts/train.py`)

Trains the U-Net model on preprocessed patches:

- Uses BCE with Logits loss for binary segmentation
- Mixed precision training (FP16) on CUDA for memory efficiency
- Validates after each epoch and saves best model based on Dice score

### 3. Prediction (`Scripts/predict.py`)

Runs inference on test images:

- Loads trained model checkpoint
- Applies sigmoid activation and thresholds at 0.5
- Saves binary segmentation masks

### 4. Feature Extraction (`Scripts/feature_extraction/`)

Extracts quantitative vessel network features from predicted masks:

- Skeletonizes the segmentation mask
- Builds a graph representation of the vessel network
- Computes metrics: vessel length, width, tortuosity, node degrees
- Exports features to Excel files and network visualizations

## Model Architecture

### U-Net (`Scripts/unet_model.py`)

A convolutional encoder-decoder architecture with skip connections:

```
Input (1, 512, 512)
    │
    ▼
┌─────────────────────────────────────┐
│  Encoder (Contracting Path)         │
│  DoubleConv: 1 → 64                 │
│  Down: 64 → 128 → 256 → 512 → 512   │
│  (MaxPool + DoubleConv at each step)│
└─────────────────────────────────────┘
    │
    ▼ Dropout (0.5)
    │
┌─────────────────────────────────────┐
│  Decoder (Expanding Path)           │
│  Up: 512 → 256 → 128 → 64           │
│  (Upsample + Concat + DoubleConv)   │
└─────────────────────────────────────┘
    │
    ▼
Output (1, 512, 512)
```

## Directory Structure

```
.
├── Data/
│   ├── raw_data/                # Original images
│   ├── preprocessed/            # Processed patches
│   └── outputs/
│       ├── predictions/         # Model predictions
│       └── features/            # Extracted vessel features
├── Model/
│   └── unet_checkpoint.pth.tar  # Trained model weights
├── Scripts/
│   ├── constants.py             # Centralized hyperparameters
│   ├── data_preprocessing.py
│   ├── train.py
│   ├── predict.py
│   ├── unet_model.py
│   ├── dataset/
│   └── feature_extraction/
├── Runners/
│   ├── run_pipeline.py          # Main pipeline runner
│   ├── oneclick.py              # One-click reproduction script
│   └── setup_environment.py     # Environment setup module
├── Documentation/
│   ├── Reproducibility.md
│   ├── EDA.md
│   └── Results.md
├── eda_angio_net.ipynb          # EDA notebook
└── requirements.txt
```

## Expected Results

- **Dice Score**: ~0.936 on validation set
- **Pixel Accuracy**: ~91.85%
- **Training Time**: ~2-5 minutes on GPU (L100)

## Contributing

We welcome contributions from the community! To contribute:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Submit a pull request with a detailed description of changes.

Please adhere to our [Code of Conduct](CODE_OF_CONDUCT.md) when interacting with the project.

---

## Security

If you discover any security vulnerabilities, please report them
via [yashashm.dev@gmail.com](mailto:yashashm.dev@gmail.com). We take security issues seriously and appreciate your
efforts to responsibly disclose them. Read more at [SECURITY](SECURITY.md)

---

## Code of Conduct

This project is governed by a [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you agree to uphold a welcoming
and inclusive environment.

---

## License

AngioNet is licensed under the [BSD 2-Clause License](LICENSE).