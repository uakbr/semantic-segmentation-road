import os
import numpy as np
from torch.utils.data import Dataset

class CityscapesDataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None):
        """
        Custom dataset class for loading Cityscapes dataset.

        Args:
            data_dir (str): Path to the directory containing preprocessed data.
            split (str): Data split to load ('train', 'val', or 'test'). Default: 'train'.
            transform (callable): Optional transform to be applied on a sample. Default: None.
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.image_dir = os.path.join(data_dir, split, 'images')
        self.label_dir = os.path.join(data_dir, split, 'labels')
        self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.npy')])
        self.label_files = sorted([f for f in os.listdir(self.label_dir) if f.endswith('.npy')])

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.image_files)

    def __getitem__(self, index):
        """
        Retrieves a sample from the dataset at the given index.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the image and corresponding label.
        """
        image_file = self.image_files[index]
        label_file = self.label_files[index]

        image = np.load(os.path.join(self.image_dir, image_file))
        label = np.load(os.path.join(self.label_dir, label_file))

        sample = (image, label)

        if self.transform:
            sample = self.transform(sample)

        return sample