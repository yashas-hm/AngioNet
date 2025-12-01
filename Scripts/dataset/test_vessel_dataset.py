import os

import torch
from skimage import io, util, color

try:
    from Scripts.dataset.vessel_dataset import VesselDataset
except ImportError:
    from .vessel_dataset import VesselDataset


class TestVesselDataset(VesselDataset):
    """
    PyTorch Dataset for loading test images without masks.

    Extends VesselDataset for inference mode where ground truth masks
    are not available. Returns image tensors with filenames for saving.

    Attributes:
        image_dir: Directory containing test image patches.
        transform: Optional transforms to apply.
    """

    def __init__(self, image_dir, transform=None):
        # mask_dir is dummy, as test data doesn't have masks
        super().__init__(image_dir=image_dir, mask_dir=image_dir, transform=transform)
        self.image_filenames = sorted([f for f in os.listdir(image_dir) if f.endswith('.tif')])

    def __getitem__(self, idx):
        image_filename = self.image_filenames[idx]
        image_path = os.path.join(self.image_dir, image_filename)
        image = io.imread(image_path)

        if image.ndim == 3:
            image = color.rgb2gray(image)
        image = util.img_as_float(image)
        image = torch.from_numpy(image).float()
        image = image.unsqueeze(0)

        sample = {'image': image, 'filename': image_filename}
        if self.transform:
            sample = self.transform(sample)
        return sample
