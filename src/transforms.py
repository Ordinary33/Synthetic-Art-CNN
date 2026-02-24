from torchvision import transforms


def get_transform():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

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
