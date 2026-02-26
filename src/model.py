import torch.nn as nn
import torch.nn.functional as F


class AIDCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=2):
        super(AIDCNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=3, padding=1, stride=1
        )
        self.bn1 = nn.BatchNorm2d(32)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=1
        )
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1
        )
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1
        )
        self.bn4 = nn.BatchNorm2d(256)

        self.conv5 = nn.Conv2d(
            in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=1
        )
        self.bn5 = nn.BatchNorm2d(512)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.pool(F.relu(self.bn5(self.conv5(x))))
        x = self.avg_pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)

        return x
