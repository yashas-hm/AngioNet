import os

import numpy as np
from skimage import io
from torch.utils.data import DataLoader

try:
    # When imported from run_pipeline.py (from project root)
    from Scripts.dataset.test_vessel_dataset import TestVesselDataset
    from Scripts.unet_model import UNet
    from Scripts.core.utils import print_progress
    from Scripts.core.constants import *
except ImportError:
    # When running directly from Scripts directory
    from dataset.test_vessel_dataset import TestVesselDataset
    from unet_model import UNet
    from core.utils import print_progress
    from core.constants import *


def load_checkpoint(checkpoint_path, model, optimizer=None):
    """
    Load model checkpoint from disk.

    Restores model weights and optionally optimizer state for
    inference or continued training.

    Args:
        checkpoint_path: Path to saved checkpoint file.
        model: U-Net model to load weights into.
        optimizer: Optional optimizer to restore state (for training).

    Returns:
        None. Model and optimizer are updated in-place.
    """
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer"])


def predict_fn(loader, model, output_dir):
    """
    Run inference on test images and save segmentation masks.

    Applies sigmoid activation and thresholds at 0.5 to produce
    binary segmentation masks. Inverts output for visualization.

    Args:
        loader: PyTorch DataLoader containing test images.
        model: Trained U-Net model.
        output_dir: Directory to save predicted masks.

    Returns:
        None. Predictions are saved as TIFF files.
    """
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    total = len(loader)
    with torch.no_grad():
        for batch_idx, sample in enumerate(loader):
            images = sample['image'].to(device=DEVICE)
            filenames = sample['filename']  # Assuming dataset returns filenames

            predictions = torch.sigmoid(model(images))
            predictions = (predictions > 0.5).float()  # Threshold to get binary mask

            for i, pred_mask in enumerate(predictions):
                filename = filenames[i]
                # Convert tensor to numpy array, remove channel dimension, and scale to 0-255
                pred_mask_np = ((1 - pred_mask.squeeze().cpu().numpy()) * 255)

                # Save the predicted mask
                output_path = os.path.join(output_dir, filename)
                io.imsave(str(output_path), pred_mask_np.astype(np.uint8), check_contrast=False)

            print_progress(batch_idx + 1, total, prefix='Predicting')

    model.train()  # Set model back to training mode


def run_prediction(base_path):
    """
    Main entry point for running inference on test images.

    Loads trained model and generates segmentation masks for all
    preprocessed test patches.

    Args:
        base_path: Path to project root directory.

    Returns:
        None. Predictions are saved to Data/outputs/predictions/.
    """
    model_path = os.path.join(base_path, 'Model/unet_checkpoint.pth.tar')
    test_img_dir = os.path.join(base_path, 'Data/preprocessed/test/image')
    output_pred_dir = os.path.join(base_path, 'Data/outputs/predictions')

    if not os.path.isdir(test_img_dir):
        print("\nError: Preprocessed test data directories not found!")
        print("Please ensure you have run the data_preprocessing.py script")
        print(f"and that the data is located in '{os.path.abspath(base_path)}'")
        return

    print(f"Using device: {DEVICE}")

    # Initialize the model
    model = UNet(n_channels=1, n_classes=1).to(DEVICE)

    # Load the trained model checkpoint
    load_checkpoint(model_path, model)

    test_ds = TestVesselDataset(image_dir=test_img_dir)
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=False,
    )

    # Make predictions
    predict_fn(test_loader, model, output_pred_dir)
    print(f"Predictions saved to: {os.path.abspath(output_pred_dir)}")


if __name__ == "__main__":
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    run_prediction(root)
