import os

import numpy as np
from skimage import io, color, util


try:
    # When imported from run_pipeline.py (from project root)
    from Scripts.core.utils import print_progress
except ImportError:
    # When running directly from Scripts directory
    from core.utils import print_progress


def create_patches(image, mask, patch_size, stride, output_path_image, output_path_mask, name):
    """
    Extract overlapping patches from image-mask pairs for training/validation.

    Filters out patches with insufficient vessel content (<5% vessel pixels)
    to ensure quality training data.

    Args:
        image: Input retina image (grayscale or RGB, will be converted to grayscale).
        mask: Binary ground truth mask (vessels = white).
        patch_size: Size of square patches to extract (default 512).
        stride: Step size between patches, controls overlap (default 256).
        output_path_image: Directory to save image patches.
        output_path_mask: Directory to save corresponding mask patches.
        name: Base filename for saved patches.

    Returns:
        None. Patches are saved to disk as TIFF files.
    """
    os.makedirs(output_path_image, exist_ok=True)
    os.makedirs(output_path_mask, exist_ok=True)

    if image.ndim == 3:
        image = color.rgb2gray(image)

    image = util.img_as_float(image)
    mask = mask.astype(float)

    patch_count = 0
    total_patches = 0
    for y in range(0, image.shape[0] - patch_size + 1, stride):
        for x in range(0, image.shape[1] - patch_size + 1, stride):
            total_patches += 1
            patch_image = image[y:y + patch_size, x:x + patch_size]
            patch_mask = mask[y:y + patch_size, x:x + patch_size]

            black_pixel_ratio = np.sum(patch_mask) / (patch_size * patch_size)
            is_mostly_black = black_pixel_ratio < 0.05

            if not is_mostly_black:
                patch_filename = f"{os.path.splitext(name)[0]}_patch_{patch_count}.tif"
                io.imsave(os.path.join(output_path_image, patch_filename), patch_image, check_contrast=False)
                io.imsave(os.path.join(output_path_mask, patch_filename), (patch_mask * 255).astype(np.uint8),
                          check_contrast=False)
                patch_count += 1

    print(f"Created {patch_count} valid patches out of {total_patches}")


def create_patches_test(image, patch_size, stride, output_path_image, name):
    """
    Extract overlapping patches from test images (no masks).

    Unlike create_patches(), this function saves all patches without
    filtering, as test images don't have ground truth masks.

    Args:
        image: Input test retina image (grayscale or RGB).
        patch_size: Size of square patches to extract (default 512).
        stride: Step size between patches, controls overlap (default 256).
        output_path_image: Directory to save image patches.
        name: Base filename for saved patches.

    Returns:
        None. Patches are saved to disk as TIFF files.
    """
    os.makedirs(output_path_image, exist_ok=True)

    if image.ndim == 3:
        image = color.rgb2gray(image)

    image = util.img_as_float(image)

    patch_count = 0
    total_patches = 0
    for y in range(0, image.shape[0] - patch_size + 1, stride):
        for x in range(0, image.shape[1] - patch_size + 1, stride):
            total_patches += 1
            patch_image = image[y:y + patch_size, x:x + patch_size]

            patch_filename = f"{os.path.splitext(name)[0]}_patch_{patch_count}.tif"
            io.imsave(os.path.join(output_path_image, patch_filename), patch_image, check_contrast=False)
            patch_count += 1

    print(f"Created {patch_count} patches for test image {name} out of {total_patches}")


def process_train_data(project_root: str):
    """
    Process training data by extracting patches from raw retina images.

    Handles mask inversion (training masks have inverted foreground/background
    compared to validation masks in this dataset).

    Args:
        project_root: Path to project root directory.

    Returns:
        None. Patches are saved to Data/preprocessed/train/.
    """
    base_data_path = os.path.join(project_root, 'Data/raw_data')
    base_output_path = os.path.join(project_root, 'Data/preprocessed')

    image_path = os.path.join(base_data_path, 'train/retina.tif')
    mask_path = os.path.join(base_data_path, 'train/retina groundtruth.tif')
    output_image_path = os.path.join(base_output_path, 'train/image')
    output_mask_path = os.path.join(base_output_path, 'train/mask')

    print("Processing training data...")
    image = io.imread(image_path)
    mask = io.imread(mask_path)
    mask = mask > 0
    # Training mask is inverted compared to validation
    mask = np.logical_not(mask)

    create_patches(image, mask, PATCH_SIZE, STRIDE, output_image_path, output_mask_path, 'train')


def process_validation_data(project_root: str):
    """
    Process validation data by extracting patches from raw retina images.

    Args:
        project_root: Path to project root directory.

    Returns:
        None. Patches are saved to Data/preprocessed/validation/.
    """
    base_data_path = os.path.join(project_root, 'Data/raw_data')
    base_output_path = os.path.join(project_root, 'Data/preprocessed')

    image_path = os.path.join(base_data_path, 'validation/retina.tif')
    mask_path = os.path.join(base_data_path, 'validation/retina groundtruth.tif')
    output_image_path = os.path.join(base_output_path, 'validation/image')
    output_mask_path = os.path.join(base_output_path, 'validation/mask')

    print("Processing validation data...")
    image = io.imread(image_path)
    mask = io.imread(mask_path)
    mask = mask > 0

    create_patches(image, mask, PATCH_SIZE, STRIDE, output_image_path, output_mask_path, 'validation')


def process_test_data(project_root: str):
    """
    Process test data by extracting patches from raw retina images.

    Test images have no ground truth masks, so all patches are saved
    without filtering.

    Args:
        project_root: Path to project root directory.

    Returns:
        None. Patches are saved to Data/preprocessed/test/.
    """
    base_data_path = os.path.join(project_root, 'Data/raw_data')
    base_output_path = os.path.join(project_root, 'Data/preprocessed')
    test_image_dir = os.path.join(base_data_path, 'test')
    output_test_image_path = os.path.join(base_output_path, 'test/image')

    print("Processing test data...")
    test_image_filenames = sorted([f for f in os.listdir(test_image_dir) if f.endswith('.tif')])
    total = len(test_image_filenames)
    for idx, filename in enumerate(test_image_filenames):
        image_path = os.path.join(test_image_dir, filename)
        image = io.imread(image_path)
        create_patches_test(image, PATCH_SIZE, STRIDE, output_test_image_path, filename)
        print_progress(idx + 1, total, prefix='Test images')


def process_data(project_root: str):
    """
    Main entry point for data preprocessing pipeline.

    Processes all data splits (train, validation, test) by extracting
    overlapping patches from raw retina images.

    Args:
        project_root: Path to project root directory.

    Returns:
        None. All patches are saved to Data/preprocessed/.
    """
    process_train_data(project_root)

    process_validation_data(project_root)

    process_test_data(project_root)


if __name__ == "__main__":
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    process_data(root)
