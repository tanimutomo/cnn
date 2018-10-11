import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

from src.model import CNN
from src.trainer import Trainer


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_data = MNIST('./data/mnist_data/',
            train=True, download=True, transform=transforms.ToTensor())
    train_loader = DataLoader(train_data, batch_size=4, shuffle=True)

    test_data = MNIST('./data/mnist_data/',
            train=False, download=True, transform=transforms.ToTensor())
    test_loader = DataLoader(test_data, batch_size=4, shuffle=False)

    trainer = Trainer(train_loader, test_loader, CNN(), device)
    trainer.train(epochs=50, lr=1e-3)

if __name__ == '__main__':
    main()
