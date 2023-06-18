import torch.nn as nn
import torch.nn.functional as F


# Target:
#   Make the model converge faster.
# Results:
#   Parameters: 7,882
#   Best train accuracy: 98.4
#   Best test accuracy: 98.46
# Analysis:
#   reduced LR step size from 15 to 10, this gave some goodness but the overall accuracy is still not enough
#   Model now has less than 8k params due to replacing of 1 large conv layer with adaptive pooling and reducing num channels
#   Observations:
#       6, 9, 8 have smaller circles
#       1 and 7 have similar shape
#       0 2 3 4 5 don't have re-usable similarities
#       -> so 7 or 8 channels should be enough to capture the basic patterns

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
