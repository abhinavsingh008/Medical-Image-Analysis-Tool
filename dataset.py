# dataset.py
import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class LungSegmentationDataset(Dataset):
    def __init__(self, transform=None):
        self.images_dir = r"C:\Users\Abhinav Singh\Desktop\lung_segmentation_project\data\images"
        self.masks_dir = r"C:\Users\Abhinav Singh\Desktop\lung_segmentation_project\data\masks"

        self.image_paths = [os.path.join(self.images_dir, fname) for fname in os.listdir(self.images_dir) if fname.endswith(('.png', '.jpg', '.jpeg')) and not fname.startswith('.')]
        self.mask_paths = [os.path.join(self.masks_dir, fname) for fname in os.listdir(self.masks_dir) if fname.endswith(('.png', '.jpg', '.jpeg')) and not fname.startswith('.')]

        self.image_paths.sort()
        self.mask_paths.sort()

        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        image = Image.open(img_path).convert('L')
        mask = Image.open(mask_path).convert('L')

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask
