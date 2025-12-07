# GAN Implementation for CelebA

A simple but complete implementation of GANs for practicing on the CelebA dataset. Includes three variants:
- **Vanilla GAN**: Standard GAN with binary cross-entropy loss
- **LSGAN**: Least Squares GAN
- **WGAN**: Wasserstein GAN

## Architecture

The networks use a simple DCGAN-style architecture suitable for 64x64 images:
- **Generator**: Transposed convolutions from noise (100-dim) to 64x64x3 images
- **Discriminator**: Convolutional layers to classify real vs fake images

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Training (Vanilla GAN)
```bash
python train.py --gan_type vanilla --epochs 50 --batch_size 64
```

### LSGAN
```bash
python train.py --gan_type lsgan --epochs 50 --batch_size 64
```

### WGAN
```bash
python train.py --gan_type wgan --epochs 50 --batch_size 64
```

### Arguments

- `--data_dir`: Directory for CelebA dataset (default: `./data`)
- `--output_dir`: Output directory for images and checkpoints (default: `./output`)
- `--gan_type`: Type of GAN (`vanilla`, `lsgan`, `wgan`)
- `--batch_size`: Batch size (default: 64)
- `--latent_dim`: Latent dimension (default: 100)
- `--image_size`: Image size (default: 64)
- `--epochs`: Number of epochs (default: 50)
- `--lr`: Learning rate (default: 0.0002)
- `--d_steps`: Discriminator steps per generator step (default: 1, auto-set to 5 for WGAN)
- `--clip_value`: Weight clipping value for WGAN (default: 0.01)

## Output

The training script will:
- Save generated images every 5 epochs to `output_dir/epoch_*.png`
- Save model checkpoints (`generator.pth` and `discriminator.pth`)

## Notes

- The CelebA dataset will be automatically downloaded on first run
- Images are normalized to [-1, 1] range
- The networks use simple architectures suitable for learning and practice
- For WGAN, consider using gradient penalty (WGAN-GP) for better stability



