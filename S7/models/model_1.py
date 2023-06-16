import torch.nn as nn
import torch.nn.functional as F

# Target:
#   Set up a basic working model that reaches 99% train and test accuracy in 20 epochs
# Results:
#   Parameters: 509,946
#   Best train accuracy: 99.11
#   Best test accuracy: 99:03
# Analysis:
#   Heavy model and converges very slowly.
#   The model seems like a good starting point. Converges slowly, but converges to 99%.
#   The model does not seem to have a lot of over fitting.

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3),
            nn.ReLU())

        self.tran1 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 16, 1)
        )

        self.block_2 = nn.Sequential(
            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3),
            nn.ReLU()

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
