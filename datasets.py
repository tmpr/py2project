from pathlib import Path
import matplotlib.pyplot as plt

import torch
import numpy as np
import pickle as pkl
from torch.utils.data import Dataset
from PIL import Image

from functional import *

X_MAP = np.array([np.linspace(start=-1, stop=1, num=100)
                  for _ in range(100)])


class DisectedSet(Dataset):
    """Dataset consisting of images with a random rectangle removed.

    Args:
        root_dir (str or Path): Directory containing .jpg images.
    """

    def __init__(self, root_dir: str):
        root_dir = Path(root_dir)
        self.path_names = sorted(
            list(p for p in root_dir.glob('*.jpg') if int(p.stem) < 50_000))

    def __getitem__(self, idx: int):
        """Retrieve sample of dataset with random part cropped out.

        Args:
            idx (int): ID of image

        Returns:
            dict: Dictionary with keys:
            {'input', 'original', 'target', 'mean, 'std, '
            size': size, 'center': center}
        """
        image = Image.open(self.path_names[idx])
        array = np.asarray(image, dtype=np.float32)
        width, height = array.shape
        center, size = crop_dimensions(height, width)
        array = custom_pad(array, 100)
        original = array.copy()
        cropped, mask, _ = disect(array, size, center)

        o_mean = original.mean()
        o_std = original.std()

        mean = cropped.mean()
        std = cropped.std()

        cropped -= mean
        cropped /= std

        original -= o_mean
        original /= o_std

        # Positional Layers
        x_map = X_MAP.copy()
        y_map = X_MAP.copy().T

        input_tensor = torch.tensor(np.stack([cropped, mask, x_map, y_map]),
                                    dtype=torch.float32)

        original = torch.tensor(original, dtype=torch.float32).unsqueeze(0)

        return input_tensor, original

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __len__(self):
        return len(self.path_names)


class ScoreSet(Dataset):
    """Dataset retrievend from pickle-file. Used for scoring and testing.

    Args:
        Dataset (str or Path): Path to pickle-file.
    """

    def __init__(self, source):
        with open(source, 'rb') as f:
            self.unpickled = pkl.load(f)

    def __getitem__(self, idx):
        array, size, center = (subdict[idx]
                               for subdict in self.unpickled.values())
        array = np.array(array, dtype=np.float32)
        array = custom_pad(array, 100)
        cropped, mask, _ = disect(array, size, center, border_distance=False)

        mean = cropped.mean()
        std = cropped.std()

        cropped -= mean
        cropped /= std

        # Positional Layers        x_map = np.array([np.linspace(start=-1, stop=1, num=100)
        x_map = X_MAP.copy()
        y_map = X_MAP.copy().T

        input_tensor = torch.tensor(np.stack([cropped, mask, x_map, y_map]),
                                    dtype=torch.float32)

        out = {
            'input': input_tensor,
            'mean': mean,
            'std': std,
            'center': center,
            'size': size
        }

        return out

    def __len__(self):
        return len(self.unpickled['images'])
