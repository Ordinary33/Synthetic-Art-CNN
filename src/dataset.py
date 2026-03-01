from torch.utils.data import Dataset
from pathlib import Path
from src import logger_config
from torchvision.datasets import ImageFolder

PROJECT_ROOT = Path(__file__).resolve().parent.parent
root_dir = PROJECT_ROOT / "data" / "train"

logger = logger_config.setup_logger(name="dataset_logger")


class ImageDataset(Dataset):
    def __init__(self, root_dir=root_dir, transform=None):
        self.dataset = ImageFolder(root=root_dir, transform=transform)
        logger.info(f"Dataset initialized with {len(self.dataset)} samples.")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        return image, label

    @property
    def classes(self):
        return self.dataset.classes
