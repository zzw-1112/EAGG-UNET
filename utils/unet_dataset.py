# -*- coding: utf-8 -*-
"""LCMA dataset loader.

A data list file contains one sample per line::

    <image_path>\t<label_path>

Paths may be absolute or relative to the list file itself.
"""

import os

import numpy as np
import torch

# IMPORTANT: import GDAL before torchvision. In some conda environments
# loading torchvision first pulls a different libjpeg, which breaks GDAL's
# libtiff with "undefined symbol: jpeg12_write_raw_data". Keep this order.
from osgeo import gdal

from torch.utils.data.dataset import Dataset
from torchvision import transforms

# Per-band normalisation statistics of the LCMA 5-band dataset.
LCMA_MEAN = (0.209, 0.394, 0.380, 0.344, 0.481)
LCMA_STD = (0.141, 0.027, 0.032, 0.046, 0.069)

# Per-band scaling factors used to stretch raw DN values into [0, 255].
_BAND_SCALE = (122.0, 1375.0, 1583.0, 1267.0, 2612.0)


def read_txt(path):
    """Read a data list file into ``(image_paths, label_paths)``."""
    ims, labels = [], []
    base_dir = os.path.dirname(os.path.abspath(path))

    with open(path, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                im, label = parts[0], parts[1]

                if not os.path.isabs(im):
                    im = os.path.join(base_dir, im)
                if not os.path.isabs(label):
                    label = os.path.join(base_dir, label)

                ims.append(im)
                labels.append(label)
    return ims, labels


def read_label(filename):
    """Read a label GeoTIFF as a numpy array."""
    dataset = gdal.Open(filename)
    if dataset is None:
        raise ValueError(f'Cannot open file: {filename}')

    im_width = dataset.RasterXSize
    im_height = dataset.RasterYSize
    im_data = dataset.ReadAsArray(0, 0, im_width, im_height)

    del dataset
    return im_data


def read_tiff(filename, train=True):
    """Read a multi-band GeoTIFF and stretch each band into [0, 255].

    ``train`` is kept for API compatibility; both branches apply the same
    per-band scaling.
    """
    dataset = gdal.Open(filename)
    if dataset is None:
        raise ValueError(f'Cannot open file: {filename}')

    im_width = dataset.RasterXSize
    im_height = dataset.RasterYSize
    im_data = dataset.ReadAsArray(0, 0, im_width, im_height)

    for band, scale in enumerate(_BAND_SCALE):
        im_data[band, ...] = im_data[band, ...] * 255 / scale

    del dataset
    return im_data


def class_7(filename):
    """Merge 20 raw labels into 7 classes (legacy helper)."""
    label = np.array(read_tiff(filename))
    label_7 = label
    for i in range(len(label)):
        for j in range(len(label[i])):
            if label[i][j] in range(0, 3):
                label_7[i][j] = 0
            elif label[i][j] in range(3, 7):
                label_7[i][j] = 1
            elif label[i][j] in range(7, 11):
                label_7[i][j] = 2
            elif label[i][j] in range(11, 13):
                label_7[i][j] = 3
            elif label[i][j] in range(13, 16):
                label_7[i][j] = 4
            elif label[i][j] in range(16, 19):
                label_7[i][j] = 5
            elif label[i][j] == 19:
                label_7[i][j] = 6
    return label_7


def build_transform():
    """Normalisation shared by training and evaluation."""
    return transforms.Compose([
        transforms.Normalize(mean=LCMA_MEAN, std=LCMA_STD)
    ])


def to_tensor(image):
    """Convert an ``[H, W, C]`` numpy array into a float ``[C, H, W]`` tensor."""
    return transforms.ToTensor()(image)


class UnetDataset(Dataset):
    """Sample = ``(image, label, label_path)``.

    ``label_path`` is returned as well so that per-image results can be saved.
    """

    def __init__(self, txtpath, transform, train=True):
        super().__init__()
        self.ims, self.labels = read_txt(txtpath)
        self.transform = transform
        self.train = train

        print(f'Loaded dataset: {len(self.ims)} samples')
        if len(self.ims) > 0:
            print(f'  example image: {self.ims[0]}')
            print(f'  example label: {self.labels[0]}')

    def __getitem__(self, index):
        im_path = self.ims[index]
        label_path = self.labels[index]

        # image: [C, H, W] -> [H, W, C] -> tensor
        image = read_tiff(im_path, self.train)
        image = np.array(image)
        image = np.transpose(image, (1, 2, 0))

        image = to_tensor(image)
        image = image.to(torch.float32)
        image = self.transform(image)

        label = torch.from_numpy(
            np.asarray(read_label(label_path), dtype=np.int32)
        ).long()

        return image, label, label_path

    def __len__(self):
        return len(self.ims)
