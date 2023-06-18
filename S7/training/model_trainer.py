import matplotlib.pyplot as plt
import torch
from tqdm import tqdm


class ModelTrainer:
    train_losses = []
    test_losses = []
    train_acc = []
    test_acc = []

    def __int__(self):
        # Data to plot accuracy and loss graphs
        self.train_losses = []
        self.test_losses = []
        self.train_acc = []
        self.test_acc = []

        self.test_pred = {'images': [], 'ground_truths': [], 'predicted_vals': []}
        self.train_loader = None
        self.test_loader = None

    def init_data_loaders(self, train_dataset, test_dataset, batch_size=512, shuffle=True, num_workers=2,
                          pin_memory=True):
        kwargs = {'batch_size': batch_size, 'shuffle': shuffle, 'num_workers': num_workers, 'pin_memory': pin_memory}

        self.test_loader = torch.utils.data.DataLoader(test_dataset, **kwargs)
        self.train_loader = torch.utils.data.DataLoader(train_dataset, **kwargs)

    def get_correct_pred_count(self, pPrediction, pLabels):
        return pPrediction.argmax(dim=1).eq(pLabels).sum().item()

    def train(self, model, device, optimizer, criterion):
        model.train()
        pbar = tqdm(self.train_loader)

        train_loss = 0
        correct = 0
        processed = 0

        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()

            # Predict
            pred = model(data)

            # Calculate loss
            loss = criterion(pred, target)
            train_loss += loss.item()

            # Backpropagation
            loss.backward()

            optimizer.step()

            correct += self.get_correct_pred_count(pred, target)
            processed += len(data)

            pbar.set_description(
                desc=f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100 * correct / processed:0.2f}')

        self.train_acc.append(100 * correct / processed)
        self.train_losses.append(train_loss / len(self.train_loader))

    def test(self, model, device, criterion):
        model.eval()

        test_loss = 0
        correct = 0

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.test_loader):
                data, target = data.to(device), target.to(device)

                output = model(data)
                test_loss += criterion(output, target, reduction='sum').item()  # sum up batch loss

                correct += self.get_correct_pred_count(output, target)

        test_loss /= len(self.test_loader.dataset)
        self.test_acc.append(100. * correct / len(self.test_loader.dataset))
        self.test_losses.append(test_loss)

        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(self.test_loader.dataset),
            100. * correct / len(self.test_loader.dataset)))

    def run_training(self, model, device, optimizer, criterion, scheduler, num_epochs):
        for epoch in range(1, num_epochs + 1):
            print(f'Epoch {epoch}')
            self.train(model, device, optimizer, criterion)
            self.test(model, device, criterion)
            scheduler.step()

    def show_sample_images_from_dataset(self, num_rows, num_cols):
        batch_data, batch_label = next(iter(self.train_loader))
        fig = plt.figure()

        for i in range(num_rows * num_cols):
            plt.subplot(num_rows, num_cols, i + 1)
            plt.tight_layout()
            plt.imshow(batch_data[i].squeeze(0), cmap='gray')
            plt.title(batch_label[i].item())
            plt.xticks([])
            plt.yticks([])

    def plot_accuracy_and_loss(self, fig_size=(15, 10)):
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        axs[0, 0].plot(self.train_losses)
        axs[0, 0].set_title("Training Loss")
        axs[1, 0].plot(self.train_acc)
        axs[1, 0].set_title("Training Accuracy")
        axs[0, 1].plot(self.test_losses)
        axs[0, 1].set_title("Test Loss")
        axs[1, 1].plot(self.test_acc)
        axs[1, 1].set_title("Test Accuracy")
