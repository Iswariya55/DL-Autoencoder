# DL- Convolutional Autoencoder for Image Denoising

## AIM
To develop a convolutional autoencoder for image denoising application.

## DESIGN STEPS
STEP 1:
Collect Dataset – Use image dataset and add noise to the images

STEP 2:
Preprocess Data – Resize images, normalize pixel values, and split into training and testing sets.

STEP 3:
Build Model – Create a Convolutional Autoencoder with encoder (Conv + MaxPooling) and decoder (Conv + UpSampling).

STEP 4:
Train Model – Train the network using noisy images as input and clean images as target.

STEP 5:
Evaluate Model – Test the model on noisy images to check denoising performance.

STEP 6:
Display Output – Compare original image, noisy image, and denoised image.




## PROGRAM

### Name:ISHWARYA R

### Register Number:212224220039

```
# Autoencoder for Image Denoising using PyTorch
#self supervised learning,both encoderand decoder

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from torchsummary import summary

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transform: Normalize and convert to tensor
transform = transforms.Compose([
    transforms.ToTensor()
])

# Load MNIST dataset
dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Add noise to images
def add_noise(inputs, noise_factor=0.5):
    noisy = inputs + noise_factor * torch.randn_like(inputs) #lime=same
    return torch.clamp(noisy, 0., 1.) #keeps in same scale=clamp

class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        # Include your code here
        super(DenoisingAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2,padding=1),
            nn.ReLU()
        )
        self.decoder=nn.Sequential(
            nn.ConvTranspose2d(32,16,kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16,1,kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Initialize model, loss function and optimizer
model = DenoisingAutoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Print model summary
summary(model, input_size=(1, 28, 28))

# Train the autoencoder
def train(model, loader, criterion, optimizer, epochs=5):
  model.train()
  print("Name:ISHWARYA R")
  print("Register Number:212224220039")
  for epoch in range(epochs):
    running_loss = 0.0 
    for images, _ in loader:
      images = images.to(device)
      noisy_images = add_noise(images).to(device)

      outputs=model(noisy_images)
      loss=criterion(outputs,images)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      running_loss += loss.item()
  
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(loader)}")

# Evaluate and visualize
def visualize_denoising(model, loader, num_images=10):
    model.eval()
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            noisy_images = add_noise(images).to(device)
            outputs = model(noisy_images)
            break

    images = images.cpu().numpy()
    noisy_images = noisy_images.cpu().numpy()
    outputs = outputs.cpu().numpy()

    print("Name:DHAN VARSHINI J P")
    print("Register Number:212224230055")
    plt.figure(figsize=(18, 6))
    for i in range(num_images):
        # Original
        ax = plt.subplot(3, num_images, i + 1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        ax.set_title("Original")
        plt.axis("off")

        # Noisy
        ax = plt.subplot(3, num_images, i + 1 + num_images)
        plt.imshow(noisy_images[i].squeeze(), cmap='gray')
        ax.set_title("Noisy")
        plt.axis("off")

        # Denoised
        ax = plt.subplot(3, num_images, i + 1 + 2 * num_images)
        plt.imshow(outputs[i].squeeze(), cmap='gray')
        ax.set_title("Denoised")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

# Run training and visualization
train(model, train_loader, criterion, optimizer, epochs=5)
visualize_denoising(model, test_loader)
```

### OUTPUT

### Model Summary

<img width="717" height="307" alt="image" src="https://github.com/user-attachments/assets/f5af6cb2-4569-4679-9973-0d0b10077193" />


### Training loss

<img width="562" height="209" alt="image" src="https://github.com/user-attachments/assets/36f4bfce-1bd7-4563-903d-6ab79711b9ba" />

## Original vs Noisy Vs Reconstructed Image

<img width="1706" height="655" alt="image" src="https://github.com/user-attachments/assets/8b897c45-fd79-461f-ae10-53cadc4d7797" />

## RESULT
The Convolutional Autoencoder successfully removed noise from images, producing clearer images that are close to the original clean images. The model effectively learned image features and improved image quality.
