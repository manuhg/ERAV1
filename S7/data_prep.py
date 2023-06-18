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
    test_data = datasets.MNIST('../data', train=False, download=True, transform=train_transforms)
    return train_data, test_data



def get_data_loaders(train_dataset, test_dataset, batch_size=512, shuffle=True, num_workers=2, pin_memory=True):
    kwargs = {'batch_size': batch_size, 'shuffle': shuffle, 'num_workers': num_workers, 'pin_memory': pin_memory}

    test_loader = torch.utils.data.DataLoader(test_dataset, **kwargs)
    train_loader = torch.utils.data.DataLoader(train_dataset, **kwargs)
    return train_loader, test_loader
