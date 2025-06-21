# mnist_cvae.py

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # Force sync CUDA errors

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader

# Force all default tensors to CUDA Float if GPU is available
if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

# Conditional Variational Autoencoder (CVAE) architecture
class CVAE(nn.Module):
    def __init__(self, latent_dim=20):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(784 + 10, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + 10, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Sigmoid()
        )

    def encode(self, x, c):
        c = c.view(-1, 10)
        x = torch.cat([x.view(-1, 784), c], dim=1)
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):
        c = c.view(-1, 10)
        z = torch.cat([z, c], dim=1)
        return self.decoder(z).view(-1, 1, 28, 28)

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar


# Loss function with flattening fix
def cvae_loss(recon_x, x, mu, logvar):
    recon_x = recon_x.view(-1, 784)
    x = x.view(-1, 784)
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


# Train function
def train_model(epochs=10, batch_size=128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load MNIST dataset
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])
    train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)



    # Initialize model with try/except
    try:
        model = CVAE(latent_dim=20)
        model = model.to(device)
        print("✅ Model moved to device successfully.")
    except RuntimeError as e:
        print("❌ RuntimeError during model initialization:")
        import traceback
        traceback.print_exc()
        exit()



    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_idx, (data, labels) in enumerate(train_loader):
            data = data.to(device)

            # 🔐 Assert correct labels
            assert labels.max() < 10 and labels.min() >= 0, f"Invalid label found: {labels}"
            labels_onehot = torch.nn.functional.one_hot(labels, num_classes=10).float().to(device)

            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data, labels_onehot)
            loss = cvae_loss(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        print(f'Epoch {epoch+1}/{epochs}, Loss: {train_loss / len(train_loader.dataset):.4f}')

    # Save model
    torch.save(model.state_dict(), 'mnist_cvae.pt')
    return model


if __name__ == "__main__":
    train_model()
