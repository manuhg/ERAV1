import torch.nn as nn
import torch.nn.functional as F


#  Observations:
#    6, 9, 8 have smaller circles
#    1 and 7 have similar shape
#    0 2 3 4 5 don't have re-usable similarities
#    -> so 7 or 8 channels should be enough to capture the basic patterns
# Target:
#   Reduce the model parameters to less than 8k
# Results:
#   Parameters: 7,924
#   Best train accuracy: 99.25
#   Best test accuracy: 99.99
# Analysis:
#   The model still managed to converge fast after param reduction.
#   The model starts to overfit from 10th epoch.

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(1, 8, 3),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, 3),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 24, 3),
            nn.ReLU(),
            nn.BatchNorm2d(24),
        )

        self.tran1 = nn.Sequential(
            nn.MaxPool2d(2, 2),
        )

        self.block_2 = nn.Sequential(
            nn.Conv2d(24, 10, 3),
            nn.ReLU(),
            nn.BatchNorm2d(10)
        )

        self.block_3 = nn.Sequential(
            nn.AdaptiveAvgPool2d(3),
            nn.Conv2d(10, 10, 3),
        )

    def forward(self, x):
        x = self.block_1(x)
        x = self.tran1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = x.view(-1, 10)
        return F.log_softmax(x)
