import os

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

try:
    # When imported from run_pipeline.py (from project root)
    from Scripts.dataset.vessel_dataset import VesselDataset
    from Scripts.unet_model import UNet
    from Scripts.core.utils import print_progress
    from Scripts.core.constants import *
except ImportError:
    # When running directly from Scripts directory
    from dataset.vessel_dataset import VesselDataset
    from unet_model import UNet
    from core.utils import print_progress
    from core.constants import *


def train_fn(loader, model, optimizer, loss_fn, scaler):
    """
    Train the model for one epoch.

    Performs forward and backward passes for all batches in the loader,
    using mixed precision training on CUDA for memory efficiency.

    Args:
        loader: PyTorch DataLoader containing training data.
        model: U-Net model to train.
        optimizer: Adam optimizer.
        loss_fn: BCEWithLogitsLoss for binary segmentation.
        scaler: GradScaler for mixed precision training (None for CPU).

    Returns:
        None. Model weights are updated in-place.
    """
    total = len(loader)
    for batch_idx, sample in enumerate(loader):
        data = sample['image'].to(device=DEVICE)
        targets = sample['mask'].to(device=DEVICE).float().unsqueeze(1)

        # Forward pass
        if DEVICE == "cuda":
            with torch.cuda.amp.autocast():
                predictions = model(data)
                loss = loss_fn(predictions, targets)
        else:
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # Backward pass
        optimizer.zero_grad()
        if DEVICE == "cuda":
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        print_progress(batch_idx + 1, total, prefix='Training', suffix=f'loss={loss.item():.4f}')


def check_accuracy(loader, model, device="cpu"):
    """
    Evaluate model performance on validation set.

    Computes pixel accuracy and Dice score (F1) for segmentation quality.
    Note: Validation targets are inverted to match training logic.

    Args:
        loader: PyTorch DataLoader containing validation data.
        model: Trained U-Net model.
        device: Device to run evaluation on ('cuda' or 'cpu').

    Returns:
        tuple: (accuracy, dice_score) - Pixel accuracy (%) and Dice coefficient.
    """
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    total = len(loader)
    with torch.no_grad():
        for batch_idx, sample in enumerate(loader):
            x = sample['image'].to(device)
            y = sample['mask'].to(device).float().unsqueeze(1)
            y = 1 - y  # Invert the validation targets to match training logic

            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()

            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)

            dice_score += (2 * (preds * y).sum()) / (
                    (preds + y).sum() + 1e-8
            )
            print_progress(batch_idx + 1, total, prefix='Validation')

    accuracy = num_correct / num_pixels * 100
    dice = dice_score / len(loader)
    print(f"Validation Accuracy: {accuracy:.2f}%")
    print(f"Dice Score: {dice:.4f}")

    model.train()
    return accuracy, dice


def save_checkpoint(model, optimizer, model_path, dice_score):
    """
    Save model checkpoint to disk.

    Saves both model weights and optimizer state for resuming training.

    Args:
        model: Trained U-Net model.
        optimizer: Optimizer with current state.
        model_path: Path to save checkpoint file.
        dice_score: Current Dice score (logged for reference).

    Returns:
        None. Checkpoint is saved to disk.
    """
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(checkpoint, model_path)
    print(f"-> Model saved with Dice score: {dice_score:.4f}")


def train_model(project_root: str):
    """
    Main entry point for model training.

    Trains U-Net model on preprocessed patches using BCE loss with
    mixed precision training. Saves best model based on Dice score.

    Args:
        project_root: Path to project root directory.

    Returns:
        None. Best model is saved to Model/unet_checkpoint.pth.tar.
    """
    train_img_dir = os.path.join(project_root, 'Data/preprocessed/train/image')
    train_mask_dir = os.path.join(project_root, 'Data/preprocessed/train/mask')
    val_img_dir = os.path.join(project_root, 'Data/preprocessed/validation/image')
    val_mask_dir = os.path.join(project_root, 'Data/preprocessed/validation/mask')
    model_path = os.path.join(project_root, 'Model/unet_checkpoint.pth.tar')

    if not os.path.isdir(train_img_dir) or not os.path.isdir(val_img_dir):
        print("\nError: Preprocessed data directories not found!")
        print("Please ensure you have run the data_preprocessing.py script")
        return

    print(f"Using device: {DEVICE}")

    # Initialize the model
    model = UNet(n_channels=1, n_classes=1).to(DEVICE)

    # Loss function and optimizer
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- Create DataLoaders ---
    train_ds = VesselDataset(image_dir=train_img_dir, mask_dir=train_mask_dir)
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        persistent_workers=False
    )

    val_ds = VesselDataset(image_dir=val_img_dir, mask_dir=val_mask_dir, validation=True)
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=False,
        persistent_workers=False
    )

    # For mixed precision training
    scaler = None
    if DEVICE == "cuda":
        scaler = torch.cuda.amp.GradScaler()

    best_dice_score = -1.0

    print("\nPress Ctrl+C to stop training (best model will be saved)\n")
    print("Note: This only works when running in terminal (IDE Limitations)")

    # --- Training Loop ---
    try:
        for epoch in range(NUM_EPOCHS):
            print(f"\n--- Epoch {epoch + 1}/{NUM_EPOCHS} ---")
            train_fn(train_loader, model, optimizer, loss_fn, scaler)

            # Check accuracy
            _, dice_score = check_accuracy(val_loader, model, device=DEVICE)

            # Save model if it's the best one so far
            if dice_score > best_dice_score:
                best_dice_score = dice_score
                save_checkpoint(model, optimizer, model_path, dice_score)

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        if best_dice_score > 0:
            print(f"Best model was already saved with Dice score: {best_dice_score:.4f}")

    print(f"\nTraining complete. Best Dice score: {best_dice_score:.4f}")


if __name__ == "__main__":
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    train_model(root)
