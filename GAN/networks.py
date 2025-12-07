import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder2D(nn.Module):
    """
    A simple encoder network for 2D data of CelebA-scale images.
    Encodes 64x64x3 images to a latent vector.
    """
    def __init__(self, latent_dim=100, act=nn.Mish):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Encoder: 64x64x3 -> latent_dim
        # Using GroupNorm(num_groups=1) which is equivalent to LayerNorm but more flexible
        self.conv_layers = nn.Sequential(
            # 64x64x3 -> 32x32x64
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.GroupNorm(1, 64),
            act(),
            # 32x32x64 -> 16x16x128
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.GroupNorm(1, 128),
            act(),
            # 16x16x128 -> 8x8x256
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.GroupNorm(1, 256),
            act(),
            # 8x8x256 -> 4x4x512
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.GroupNorm(1, 512),
            act(),
        )
        
        # Flatten and map to latent dimension
        self.fc = nn.Sequential(
            nn.Flatten(start_dim=-3, end_dim=-1),
            nn.Linear(512 * 4 * 4, latent_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = self.fc(x)
        return x


class Generator2D(nn.Module):
    """
    Generator network for CelebA (64x64x3 images).
    Takes noise vector and generates images.
    Uses GroupNorm instead of BatchNorm for better performance in modern GANs.
    """
    def __init__(self, latent_dim=64, act=nn.Mish):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Generator: latent_dim -> 64x64x3
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512 * 4 * 4),
            nn.Unflatten(dim=-1, unflattened_size=(512, 4, 4)),
        )
        
        self.conv_layers = nn.Sequential(
            # 4x4x512 -> 8x8x256
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.GroupNorm(1, 256),
            act(),
            # 8x8x256 -> 16x16x128
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.GroupNorm(1, 128),
            act(),
            # 16x16x128 -> 32x32x64
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.GroupNorm(1, 64),
            act(),
            # 32x32x64 -> 64x64x3
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()  # Output in [-1, 1]
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z)
        x = self.conv_layers(x)
        return x


class VanillaDiscriminator2D(nn.Module):
    """
    Standard GAN Discriminator with binary cross-entropy loss.
    Uses GroupNorm instead of BatchNorm for better performance in modern GANs.
    """
    def __init__(self, latent_dim=64, act=nn.Mish):
        super().__init__()
        
        # Discriminator: 64x64x3 -> 1 (real/fake probability)
        self.encoder = Encoder2D(latent_dim=latent_dim, act=act)
        
        self.output = nn.Linear(latent_dim, 1)
        self.loss = nn.BCEWithLogitsLoss()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.output(x)
        return x

    def compute_loss(self, pos_samples: torch.Tensor, neg_samples: torch.Tensor) -> dict:
        """
        Compute vanilla GAN loss (binary cross-entropy).
        pos_samples: real images
        neg_samples: fake images
        """
        # Real images should be classified as 1
        real_pred = self.forward(pos_samples)
        real_loss = self.loss(real_pred, torch.ones_like(real_pred))
        
        # Fake images should be classified as 0
        fake_pred = self.forward(neg_samples)
        fake_loss = self.loss(fake_pred, torch.zeros_like(fake_pred))
        
        # Total discriminator loss
        d_loss = real_loss + fake_loss
        
        return {
            'loss': d_loss,
            'real_loss': real_loss,
            'fake_loss': fake_loss,
            'real_score': real_pred.mean(),
            'fake_score': fake_pred.mean()
        }
    
    def compute_generator_loss(self, fake_samples: torch.Tensor) -> dict:
        """
        Compute generator loss (binary cross-entropy).
        fake_samples: fake images
        """
        fake_pred = self.forward(fake_samples)
        g_loss = self.loss(fake_pred, torch.ones_like(fake_pred))
        return {
            'loss': g_loss,
            'score': fake_pred.mean()
        }


class LeastSquaresDiscriminator2D(nn.Module):
    """
    LSGAN Discriminator with least squares loss.
    Uses GroupNorm instead of BatchNorm for better performance in modern GANs.
    """
    def __init__(self, latent_dim=64, act=nn.Mish):
        super().__init__()
        
        # Same architecture as VanillaDiscriminator but no sigmoid
        self.encoder = Encoder2D(latent_dim=latent_dim, act=act)
        
        # No sigmoid - output raw score
        self.output = nn.Linear(latent_dim, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.output(x)
        return x
    
    def compute_loss(self, pos_samples: torch.Tensor, neg_samples: torch.Tensor) -> dict:
        """
        Compute LSGAN loss (least squares).
        Real images should score close to 1, fake images close to 0.
        """
        real_pred = self(pos_samples)
        fake_pred = self(neg_samples)
        
        # Real images: minimize (D(x) - 1)^2
        real_loss = 0.5 * F.mse_loss(real_pred, torch.ones_like(real_pred))
        
        # Fake images: minimize (D(G(z)) - 0)^2
        fake_loss = 0.5 * F.mse_loss(fake_pred, torch.zeros_like(fake_pred))
        
        d_loss = real_loss + fake_loss
        
        return {
            'loss': d_loss,
            'real_loss': real_loss,
            'fake_loss': fake_loss,
            'real_score': real_pred.mean(),
            'fake_score': fake_pred.mean()
        }
    
    def compute_generator_loss(self, fake_samples: torch.Tensor) -> dict:
        """
        Compute generator loss (least squares).
        fake_samples: fake images
        """
        fake_pred = self.forward(fake_samples)
        g_loss = F.mse_loss(fake_pred, torch.ones_like(fake_pred))
        return {
            'loss': g_loss,
            'score': fake_pred.mean()
        }


class WassersteinDiscriminator2D(nn.Module):
    """
    WGAN Discriminator (Critic) with Wasserstein distance.
    Uses GroupNorm instead of BatchNorm for better performance in modern GANs.
    Note: For WGAN, you should use weight clipping or gradient penalty in training.
    """
    def __init__(self, latent_dim=64, act=nn.Mish, grad_penalty_weight=10.0):
        super().__init__()
        
        # Same architecture but no sigmoid
        self.encoder = Encoder2D(latent_dim=latent_dim, act=act)
        self.output = nn.Linear(latent_dim, 1)
        self.grad_penalty_weight = grad_penalty_weight
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.output(x)
        return x
    
    def compute_loss(self, pos_samples: torch.Tensor, neg_samples: torch.Tensor) -> dict:
        """
        Compute WGAN loss.
        Maximize: E[D(real)] - E[D(fake)]
        For discriminator training, we minimize the negative of this.
        """
        if not neg_samples.shape[0] == pos_samples.shape[0]:
            raise ValueError(f"pos_samples and neg_samples must have the same batch size, but got {neg_samples.shape[0]} and {pos_samples.shape[0]}")
        real_pred = self(pos_samples)
        fake_pred = self(neg_samples)
        
        # WGAN loss: minimize E[D(fake)] - E[D(real)]
        d_loss = fake_pred.mean() - real_pred.mean()

        eps = torch.rand(pos_samples.shape[0], 1, 1, 1, device=pos_samples.device)
        
        interpolated = eps * pos_samples + (1 - eps) * neg_samples
        interpolated.requires_grad_(True)
        interp_pred = self(interpolated)
        grad_outputs = torch.ones_like(interp_pred)
        gradients = torch.autograd.grad(
            outputs=interp_pred,
            inputs=interpolated,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
        )[0]

        gradient_norm = gradients.view(pos_samples.shape[0], -1).norm(2, dim=1)
        gradient_penalty = ((gradient_norm - 1.0) ** 2).mean()
        d_loss += self.grad_penalty_weight * gradient_penalty
        
        return {
            'loss': d_loss,
            'grad_penalty': gradient_penalty,
            'real_loss': -real_pred.mean(),  # For logging
            'fake_loss': fake_pred.mean(),   # For logging
            'real_score': real_pred.mean(),
            'fake_score': fake_pred.mean()
        }
    
    def compute_generator_loss(self, fake_samples: torch.Tensor) -> dict:
        """
        Compute generator loss (Wasserstein distance).
        fake_samples: fake images
        """
        fake_pred = self(fake_samples)
        g_loss = -fake_pred.mean()
        return {
            'loss': g_loss,
            'score': fake_pred.mean()
        }

