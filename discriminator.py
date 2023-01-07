import torch.nn as nn
import torch


class discriminator(nn.Module):
    def __init__(self, num_classes=100,init_weights=False):
        super(discriminator, self).__init__()
        self.features = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            #nn.Dropout(p=0.5),
            nn.Linear(3 * 28 * 28,16),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(16,8),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True),
            nn.Linear(8, num_classes),
      )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

