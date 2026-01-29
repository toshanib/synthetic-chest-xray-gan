import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import medmnist
from medmnist import ChestMNIST


def get_dataloader(batch_size=64, shuffle=True):
    """
    Returns DataLoader for ChestMNIST dataset
    """

    transform = transforms.Compose([
        transforms.ToTensor(),              # [0, 1]
        transforms.Normalize((0.5,), (0.5,))  # [-1, 1]
    ])

    dataset = ChestMNIST(
        split='train',
        transform=transform,
        download=True
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # ✅ IMPORTANT (Windows-safe)
        pin_memory=False
    )

    return loader