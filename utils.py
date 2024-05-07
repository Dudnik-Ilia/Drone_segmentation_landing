import os
from typing import Literal
import re

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch
from torch.utils.data.dataloader import default_collate


class SegDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.images_dir = os.path.join(root_dir, 'data', split)

        def numerical_sort(file):
            numbers = re.compile(r'(\d+)')
            parts = numbers.split(file)
            parts[1::2] = map(int, parts[1::2])  # Convert all numerical strings to integers
            return parts

        # Apply the numerical sorting method
        self.images = sorted(
            (file for file in os.listdir(self.images_dir) if file.endswith(".png")),
            key=numerical_sort
        )

        if self.split == 'train':
            self.masks = np.unpackbits(np.load(os.path.join(root_dir, f"{split}.npy"))).reshape((-1, 256, 256))
        assert len(self.images) == self.masks.shape[0], "The number of images and masks must match"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.images[idx])
        image = Image.open(img_path).convert("RGB")
        mask = self.masks[idx]

        if self.transform:
            image = self.transform(image)
        if self.split == 'train':
            mask = torch.tensor(mask, dtype=torch.long)
        else:
            mask = []
        return {"pixel_values": image, "labels": mask, "path": self.images[idx]}


def collate_fn(batch):
    pixel_values = [item['pixel_values'] for item in batch]
    paths = [item['path'] for item in batch]
    labels = [item['labels'] for item in batch if 'labels' in item]

    batched_data = {}
    batched_data['pixel_values'] = default_collate(pixel_values)
    batched_data['paths'] = paths
    if labels:
        batched_data['labels'] = default_collate(labels)

    return batched_data


def plot_sample(index: int, split: Literal['train', 'test'] = 'train', axis=None):
    img = Image.open(os.path.join('data', split, '{}.png'.format(index)))
    axis = axis if axis else plt.gca()
    axis.imshow(np.asarray(img))
    if split == 'train':
        seg = np.unpackbits(np.load('{}.npy'.format(split))).reshape((-1, 256, 256))[index]
        axis.contourf(seg, levels=1, colors='none', hatches=['', 'oo'])
    axis.set_title('Sample #{}'.format(index), fontname='Geologica')
    axis.axis('off')
    if axis is None:
        plt.show()  # Otherwise we'll assume you're composing your own plot


def load() -> np.ndarray:
    annotations = np.unpackbits(np.load('test.npy'))
    return annotations.reshape((6500, 256, 256))


def save(solution: np.ndarray):
    assert solution.shape == (1000, 256, 256)
    assert solution.dtype == bool
    np.save('test.npy', np.packbits(solution))


if __name__ == '__main__':
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(6.3, 2.1))
    plot_sample(2583, axis=axes[0])
    plot_sample(2709, axis=axes[1])
    plot_sample(1099, axis=axes[2])
    plt.tight_layout()
