import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm
from pathlib import Path
import hydra
from networks import (
    Generator2D,
    VanillaDiscriminator2D,
    LeastSquaresDiscriminator2D,
    WassersteinDiscriminator2D
)

config_path = Path(__file__).parent / 'cfg'
data_root = Path(__file__).parent.parent / 'data'
data_root.mkdir(parents=True, exist_ok=True)

@hydra.main(config_path=str(config_path), config_name="config", version_base=None)
def main(cfg):
    gan_type = cfg.gan_type
    epochs = cfg.epochs
    batch_size = cfg.batch_size
    latent_dim = cfg.latent_dim
    lr = cfg.lr
    device = cfg.device
    dataset = cfg.dataset

    if cfg.dataset == 'mnist':
        # Prepare MNIST dataset
        # Transform MNIST (28x28x1) to match network architecture (64x64x3)
        transform = transforms.Compose([
            transforms.Resize(64),  # Resize to 64x64
            transforms.ToTensor(),  # Convert to tensor [0, 1]
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # Convert grayscale to RGB by repeating channels
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
        ])

        # Download and load MNIST dataset
        print("Downloading MNIST dataset...")
        dataset = datasets.MNIST(
            root=data_root,
            train=True,
            download=True,
            transform=transform
        )
        
        print(f"MNIST dataset loaded: {len(dataset)} training samples")
    
    elif cfg.dataset == 'celeba':
        # Prepare CelebA dataset
        # Transform CelebA (218x178x3) to match network architecture (64x64x3)
        transform = transforms.Compose([
            transforms.Resize(64),  # Resize to 64x64
            transforms.ToTensor(),  # Convert to tensor [0, 1]
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
        ])
    
        # Download and load CelebA dataset
        print("Downloading CelebA dataset...")
        dataset = datasets.CelebA(
            root=data_root,
            split='train',
            download=True,
            transform=transform
        )
        
        print(f"CelebA dataset loaded: {len(dataset)} training samples")
    else:
        raise ValueError(f"Invalid dataset: {cfg.dataset}")


    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )

    discriminator_map = {
        'vanilla': VanillaDiscriminator2D,
        'lsgan': LeastSquaresDiscriminator2D,
        'wgan': WassersteinDiscriminator2D
    }
    Discriminator = discriminator_map[gan_type]

    generator = Generator2D(latent_dim=latent_dim).to(device)
    discriminator: nn.Module = Discriminator(latent_dim=latent_dim).to(device)
    optimizer_g = optim.Adam(generator.parameters(), lr=lr)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr)

    print(f"Starting training on {device}...")
    print(f"Generator parameters: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}")

    for epoch in range(epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, (real_images, _) in enumerate(pbar):
            real_images = real_images.to(device)
            batch_size = real_images.size(0)
            
            # Generate noise for fake images
            noise = torch.randn(batch_size, latent_dim, device=device)
            fake_images = generator(noise)
            
            # Train Discriminator
            d_loss_dict = discriminator.compute_loss(real_images, fake_images.detach())
            d_loss = d_loss_dict['loss']
            
            optimizer_d.zero_grad()
            d_loss.backward()
            optimizer_d.step()
            
            fake_loss_dict = discriminator.compute_generator_loss(fake_images)
            g_loss = fake_loss_dict['loss']
            
            optimizer_g.zero_grad()
            g_loss.backward()
            optimizer_g.step()
            
            # Logging and saving samples
            if batch_idx % 100 == 0:
                print(f"\nEpoch {epoch+1}, Batch {batch_idx}")
                print(f"  D_loss: {d_loss.item():.4f}, G_loss: {g_loss.item():.4f}")
                print(f"  Real score: {d_loss_dict['real_score'].item():.4f}, Fake score: {d_loss_dict['fake_score'].item():.4f}")
                
                # Save sample generated images
                with torch.no_grad():
                    sample_noise = torch.randn(16, latent_dim, device=device)
                    sample_images = generator(sample_noise)
                    save_image(
                        sample_images,
                        f'epoch_{epoch+1}_batch_{batch_idx}.png',
                        nrow=4,
                        normalize=True
                    )


if __name__ == "__main__":
    main()


