import torch.nn as nn
import torch.nn.functional as F

# Target:
#   Make the model converge faster.
# Results:
#   Parameters: 511,354
#   Best train accuracy: 99.83
#   Best test accuracy: 99.50
# Analysis:
#   Model now converges much faster. The model is able to reach 99% (train and test) accuracy in 4 epochs.
#   Train accuracy plateau's at ~99.8% after 17th epoch.
#   Test accuracy stabilises above 99.4% after 17th epoch.
#   Model needs to converge faster i.e the accuracy target is being hit but needs to be done in fewer epochs

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 3),
            nn.ReLU(),
            nn.BatchNorm2d(128),
        )

        self.tran1 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 16, 1),
        )

        self.block_2 = nn.Sequential(
            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 3),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, 3),
            nn.ReLU(),
            nn.BatchNorm2d(256)

        )

        self.block_3 = nn.Sequential(
            nn.Conv2d(256, 10, 3),
        )

    def forward(self, x):
        x = self.block_1(x)
        x = self.tran1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = x.view(-1, 10)
        return F.log_softmax(x)
