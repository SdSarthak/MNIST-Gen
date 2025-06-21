import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# --- CVAE Model Definition (same as before) ---
class CVAE(nn.Module):
    def __init__(self, latent_dim=20):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        image_size = 784
        num_classes = 10

        self.encoder = nn.Sequential(
            nn.Linear(image_size + num_classes, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, image_size),
            nn.Sigmoid()
        )

    def encode(self, x, c):
        x_flat = x.view(-1, 28 * 28)
        combined_input = torch.cat([x_flat, c], dim=1)
        h = self.encoder(combined_input)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):
        combined_input = torch.cat([z, c], dim=1)
        return self.decoder(combined_input).view(-1, 1, 28, 28)

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, c)
        return recon_x, mu, logvar

# --- Load trained model ---
@st.cache_resource
def load_model(path='mnist_cvae.pt', latent_dim=20):
    device = torch.device("cpu")
    model = CVAE(latent_dim=latent_dim)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

# --- Generate 5 images for a digit ---
def generate_images(model, digit, num_images=5):
    device = torch.device("cpu")
    c = torch.zeros(num_images, 10).to(device)
    c[:, digit] = 1
    z = torch.randn(num_images, model.latent_dim).to(device)
    with torch.no_grad():
        images = model.decode(z, c).cpu()
    return images

# --- Streamlit Web App ---
st.title("🧠 Conditional VAE Digit Generator (MNIST)")
st.markdown("Generate handwritten digits using a pretrained Conditional Variational Autoencoder (CVAE).")

# User selects a digit
digit = st.selectbox("Select a digit (0-9):", list(range(10)), index=0)

# Load pretrained model
model = load_model()

# Generate and display images
if st.button("Generate Images"):
    st.write(f"Generating 5 images of the digit **{digit}**...")
    generated_imgs = generate_images(model, digit)

    cols = st.columns(5)
    for i in range(5):
        img = generated_imgs[i].squeeze().numpy()
        img = (img * 255).astype(np.uint8)
        pil_img = Image.fromarray(img)
        cols[i].image(pil_img, caption=f"Digit {digit}", use_column_width=True)

st.markdown("---")
st.markdown("✅ This app uses a CVAE model trained on the MNIST dataset.")
