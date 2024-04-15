import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import matplotlib.pyplot as plt
from AutoEncoder import StackedAutoencoder, train, test


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #设置超参数

    input_dim = 28*28
    hidden_dims = [256,128,64]
    num_epochs = 10
    batch_size = 128
    learning_rate = 0.001

    train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Create the Stacked Autoencoder model and move it to the device
    model = StackedAutoencoder(input_dim, hidden_dims).to(device)

    # Train the model
    train(model, train_loader, num_epochs, learning_rate)

    # Test the model
    test(model, test_loader)

    # Generate a random image and its reconstruction
    with torch.no_grad():
        z = torch.randn(1, hidden_dims[-1]).to(device)
        xhat = model.decoder(z)
        xhat = xhat.view(28, 28).cpu().numpy()
        plt.imshow(xhat, cmap='gray')
        plt.show()


if __name__ == '__main__':
    main()