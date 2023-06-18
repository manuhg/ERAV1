import torch


def get_device():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    return device


def print_cuda_info():
    cuda = torch.cuda.is_available()
    print("CUDA Available?", cuda)
