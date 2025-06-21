# app.py
import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from mnist_cvae import CVAE

# Load trained model
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CVAE(latent_dim=20)
    model.load_state_dict(torch.load('mnist_cvae.pt', map_location=device))
    model.eval()
    return model

# Generate digits
def generate_digits(model, digit, num_samples=5):
    device = next(model.parameters()).device
    labels = torch.full((num_samples,), digit, dtype=torch.long)
    labels_onehot = torch.nn.functional.one_hot(labels, num_classes=10).float().to(device)
    
    with torch.no_grad():
        z = torch.randn(num_samples, model.latent_dim).to(device)
        generated = model.decode(z, labels_onehot)
    
    return generated.cpu().numpy()

# Streamlit app
def main():
    st.title("Handwritten Digit Generator")
    st.write("Generate MNIST-style handwritten digits using a Conditional Variational Autoencoder")
    
    # Digit selection
    digit = st.selectbox("Select digit to generate:", options=list(range(10)), index=0)
    
    # Generate button
    if st.button("Generate 5 Images"):
        model = load_model()
        generated = generate_digits(model, digit)
        
        # Display images
        st.subheader(f"Generated Digit: {digit}")
        fig, axes = plt.subplots(1, 5, figsize=(15, 3))
        for i in range(5):
            axes[i].imshow(generated[i, 0], cmap='gray')
            axes[i].axis('off')
        st.pyplot(fig)

if __name__ == "__main__":
    main()
