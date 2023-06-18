from torchvision import datasets, transforms


def load_data(train_transforms=[]):
    mean = 0.1307
    std_dev = 0.3081
    # Train data transformations
    train_transforms = transforms.Compose(train_transforms + [
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std_dev,)),
    ])

    # Test data transformations
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std_dev,))
    ])

    train_data = datasets.MNIST('../data', train=True, download=True, transform=train_transforms)
    test_data = datasets.MNIST('../data', train=False, download=True, transform=test_transforms)
    return train_data, test_data
