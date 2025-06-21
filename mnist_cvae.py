import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# --- Model Definition ---

class CVAE(nn.Module):
    """
    Conditional Variational Autoencoder (CVAE) for MNIST.
    It learns a latent space representation of the digits, conditioned on their labels.
    This allows us to generate a specific digit by providing its label to the decoder.
    """
    def __init__(self, latent_dim=20):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        image_size = 784 # 28x28 pixels
        num_classes = 10 # Digits 0-9

        # Encoder: Takes an image and its label, and outputs the parameters
        # of the latent distribution (mu and logvar).
        self.encoder = nn.Sequential(
            nn.Linear(image_size + num_classes, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

        # Decoder: Takes a point from the latent space and a label,
        # and reconstructs an image.
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, image_size),
            nn.Sigmoid() # Use Sigmoid to output pixel values between 0 and 1
        )

    def encode(self, x, c):
        # Concatenate the flattened image (x) and the one-hot encoded label (c)
        x_flat = x.view(-1, 28 * 28)
        combined_input = torch.cat([x_flat, c], dim=1)
        h = self.encoder(combined_input)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        # The reparameterization trick allows gradients to flow through the sampling process.
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std) # Sample from a standard normal distribution
        return mu + eps * std

    def decode(self, z, c):
        # Concatenate the latent vector (z) and the one-hot encoded label (c)
        combined_input = torch.cat([z, c], dim=1)
        return self.decoder(combined_input).view(-1, 1, 28, 28)

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, c)
        return recon_x, mu, logvar

# --- Loss Function ---

def cvae_loss_function(recon_x, x, mu, logvar):
    """
    Calculates the CVAE loss, which is a sum of two terms:
    1. Reconstruction Loss (BCE): How well the decoder reconstructs the input image.
       Measures the difference between the original and the reconstructed image.
    2. KL Divergence (KLD): A regularizer that forces the learned latent distribution
       to be close to a standard normal distribution. This helps with generating
       new, plausible samples.
    """
    # Flatten images for BCE calculation
    recon_x_flat = recon_x.view(-1, 784)
    x_flat = x.view(-1, 784)
    BCE = nn.functional.binary_cross_entropy(recon_x_flat, x_flat, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


# --- Training Function ---

def train_model(epochs=20, batch_size=128, learning_rate=1e-3, latent_dim=20):
    """
    Main function to handle the training process.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load MNIST dataset
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])
    train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    # *** FIX FOR RuntimeError: Create a generator on the correct device ***
    # This generator is passed to the DataLoader to ensure that data shuffling
    # operations occur on the same device as the model and data (the GPU).
    generator = torch.Generator(device=device)

    # Note: num_workers is set to 0 to avoid potential conflicts with CUDA in some environments.
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator # Pass the device-specific generator here
    )


    # Initialize model, optimizer and move to the selected device
    model = CVAE(latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print("Starting model training...")
    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_idx, (data, labels) in enumerate(train_loader):
            data = data.to(device)
            labels_onehot = torch.nn.functional.one_hot(labels, num_classes=10).float().to(device)

            # Forward pass
            recon_batch, mu, logvar = model(data, labels_onehot)
            
            # Calculate loss
            loss = cvae_loss_function(recon_batch, data, mu, logvar)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        avg_loss = train_loss / len(train_loader.dataset)
        print(f'====> Epoch: {epoch+1}/{epochs} Average loss: {avg_loss:.4f}')

    print("Training finished.")

    # Save the trained model
    model_path = 'mnist_cvae.pt'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    return model

# --- Generation Function ---

def generate_digits(model, num_images_per_digit=5):
    """
    Uses the trained model to generate and display sample digits.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval() # Set model to evaluation mode
    
    with torch.no_grad(): # No need to track gradients for generation
        for digit in range(10):
            # Create the conditional input (the digit label)
            c = torch.zeros(num_images_per_digit, 10).to(device)
            c[:, digit] = 1

            # Sample from the latent space (standard normal distribution)
            z = torch.randn(num_images_per_digit, model.latent_dim).to(device)
            
            # Generate images by passing latent samples and labels to the decoder
            generated_images = model.decode(z, c).cpu()
            
            # Plot the generated images
            fig, axes = plt.subplots(1, num_images_per_digit, figsize=(10, 2))
            fig.suptitle(f"Generated Digit: {digit}", fontsize=16)
            for i in range(num_images_per_digit):
                ax = axes[i]
                ax.imshow(generated_images[i].squeeze(), cmap='gray')
                ax.axis('off')
            plt.show()


# --- Main execution block ---
if __name__ == "__main__":
    # Train the model
    trained_model = train_model(epochs=20)
    
    # Generate and display sample digits using the trained model
    print("\nGenerating sample images from the trained model...")
    generate_digits(trained_model, num_images_per_digit=5)
