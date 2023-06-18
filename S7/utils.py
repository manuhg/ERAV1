import torch
import matplotlib.pyplot as plt

def get_device():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    return device


def print_cuda_info():
    cuda = torch.cuda.is_available()
    print("CUDA Available?", cuda)


def show_sample_images_from_dataset(train_loader_obj):
    batch_data, batch_label = next(iter(train_loader_obj))
    fig = plt.figure()

    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.tight_layout()
        plt.imshow(batch_data[i].squeeze(0), cmap='gray')
        plt.title(batch_label[i].item())
        plt.xticks([])
        plt.yticks([])
