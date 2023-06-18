import torch
from tqdm import tqdm


class ModelTrainer:
    def __int__(self, criterion):
        # Data to plot accuracy and loss graphs
        self.train_losses = []
        self.test_losses = []
        self.train_acc = []
        self.test_acc = []
        self.criterion = criterion

        self.test_pred = {'images': [], 'ground_truths': [], 'predicted_vals': []}

    def get_correct_pred_count(self, pPrediction, pLabels):
        return pPrediction.argmax(dim=1).eq(pLabels).sum().item()

    def train(self, model, device, train_loader, optimizer):
        model.train()
        pbar = tqdm(train_loader)

        train_loss = 0
        correct = 0
        processed = 0

        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()

            # Predict
            pred = model(data)

            # Calculate loss
            loss = self.criterion(pred, target)
            train_loss += loss.item()

            # Backpropagation
            loss.backward()

            optimizer.step()

            correct += self.get_correct_pred_count(pred, target)
            processed += len(data)

            pbar.set_description(
                desc=f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100 * correct / processed:0.2f}')

        self.train_acc.append(100 * correct / processed)
        self.train_losses.append(train_loss / len(train_loader))

    def test(self, model, device, test_loader):
        model.eval()

        test_loss = 0
        correct = 0

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = data.to(device), target.to(device)

                output = model(data)
                test_loss += self.criterion(output, target, reduction='sum').item()  # sum up batch loss

                correct += self.get_correct_pred_count(output, target)

        test_loss /= len(test_loader.dataset)
        self.test_acc.append(100. * correct / len(test_loader.dataset))
        self.test_losses.append(test_loss)

        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
