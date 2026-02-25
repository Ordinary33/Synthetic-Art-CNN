from torch.utils.data import DataLoader, random_split, Subset
import torch

from src.dataset import ImageDataset
from src.transforms import get_transform


def get_loaders(batch_size, val_split=0.2):
    transform = get_transform()

    train_full_dataset = ImageDataset(transform=transform["train"])
    val_full_dataset = ImageDataset(transform=transform["val"])

    dataset_size = len(train_full_dataset)
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - val_size

    generator = torch.Generator().manual_seed(42)
    train_subset_temp, val_subset_temp = random_split(
        train_full_dataset, [train_size, val_size], generator=generator
    )

    train_subset = train_subset_temp
    val_subset = Subset(val_full_dataset, val_subset_temp.indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    return train_loader, val_loader
