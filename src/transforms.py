from torchvision import transforms


def get_transform():
    mean = [0.4843, 0.4340, 0.3911]
    std = [0.2415, 0.2331, 0.2263]

    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.RandomCrop(256),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.CenterCrop(256),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        ),
    }

    return data_transforms
