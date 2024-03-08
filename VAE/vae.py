import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, latent_dim)
        self.fc3 = nn.Linear(hidden_dim, latent_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        h = self.relu(self.fc1(x))
        mu = self.fc2(h)
        log_var = self.fc3(h)
        return mu, log_var

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, z):
        h = self.relu(self.fc1(z))
        x_recon = self.sigmoid(self.fc2(h))
        return x_recon

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)
    
    # trick to make the sampling process differentiable
    def reparameterize(self, mu, log_var):
        # z = mu + eps * e^(sqrt(log_var))
        std = torch.exp(0.5 * log_var) # std is the standard deviation derived from log_var
        eps = torch.randn_like(std) # Sample a random noise variable (epsilon) from a standard Gaussian distribution (mean=0, variance=1)
        z = mu + eps * std # Scale and shift the noise variable using the mean (mu) and standard deviation (derived from log_var) of the desired Gaussian distribution
        return z
    
    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decoder(z)
        return x_recon, mu, log_var

# MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)

input_dim = 784  # MNIST images are 28x28 pixels, flattened to 784-dimensional vectors
hidden_dim = 256
latent_dim = 20

model = VAE(input_dim, hidden_dim, latent_dim)

def loss_function(x, x_recon, mu, log_var):
    recon_loss = nn.BCELoss(reduction='sum')(x_recon, x) # Binary cross-entropy loss
    kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) # KL divergence
    return recon_loss + kl_div

optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Training loop
num_epochs = 10
best_val_loss = float('inf')
history = {'train_loss': [], 'val_loss': []}

for epoch in range(num_epochs):
    train_loss = 0.0
    model.train()
    for batch_idx, (data, _) in enumerate(train_dataloader):
        data = data.view(data.size(0), -1)  # Flatten the data
        
        optimizer.zero_grad()
        x_recon, mu, log_var = model(data)
        loss = loss_function(data, x_recon, mu, log_var)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
    train_loss /= len(train_dataloader)
    history['train_loss'].append(train_loss)
    
    val_loss = 0.0
    model.eval()
    with torch.no_grad():
        for data, _ in test_dataloader:
            data = data.view(data.size(0), -1)  # Flatten the data
            x_recon, mu, log_var = model(data)
            loss = loss_function(data, x_recon, mu, log_var)
            val_loss += loss.item()
    
    val_loss /= len(test_dataloader)
    history['val_loss'].append(val_loss)
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_vae_model.pth')

# Plot the training and validation loss
plt.figure()
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('history.png')
# plt.show()

# Load the best model and test it
model.load_state_dict(torch.load('best_vae_model.pth'))

# Test the model
model.eval()
with torch.no_grad():
    for batch_idx, (data, _) in enumerate(test_dataloader):
        data = data.view(data.size(0), -1)  # Flatten the data
        x_recon, _, _ = model(data)
        
        # Plot the original and reconstructed images
        fig, axs = plt.subplots(2, 10, figsize=(20, 4))
        for i in range(10):
            axs[0, i].imshow(data[i].view(28, 28).numpy(), cmap='gray')
            axs[0, i].axis('off')
            axs[1, i].imshow(x_recon[i].view(28, 28).detach().numpy(), cmap='gray')
            axs[1, i].axis('off')
        
        axs[0, 0].set_title("Original")
        axs[1, 0].set_title("Reconstructed")
        plt.tight_layout()
        plt.savefig('result.png')
        # plt.show()
        break