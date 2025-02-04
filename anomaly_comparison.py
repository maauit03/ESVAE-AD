import os
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import sqrtm
import global_v as glv

import svae_models.esvae as esvae
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def load_single_image(image_path, input_size):
    """Load and preprocess a single image."""
    if not os.path.exists(image_path):
        raise ValueError(f"Image path '{image_path}' does not exist.")
    
    SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        SetRange
    ])
    
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor


def reconstruct_image(network, image_tensor):
    """Reconstruct the image using the trained network."""
    n_steps = glv.network_config['n_steps']
    network = network.eval()

    with torch.no_grad():
        spike_input = image_tensor.unsqueeze(-1).repeat(1, 1, 1, 1, n_steps)  # Add time steps
        x_recon, _, _, _ = network(spike_input, scheduled=False)  # Reconstruct the image

    return x_recon


def calculate_metrics(original_tensor, reconstructed_tensor):
    """Calculate metrics for reconstruction quality."""
    # Denormalize images (from [-1, 1] back to [0, 1])
    original = (original_tensor + 1) / 2
    reconstructed = (reconstructed_tensor + 1) / 2

    # Convert tensors to numpy arrays for comparison
    original_np = original.squeeze(0).permute(1, 2, 0).cpu().numpy()  # (H, W, C)
    reconstructed_np = reconstructed.squeeze(0).permute(1, 2, 0).cpu().numpy()  # (H, W, C)

    # Convert to grayscale for SSIM
    original_gray = np.dot(original_np[..., :3], [0.2989, 0.5870, 0.1140])
    reconstructed_gray = np.dot(reconstructed_np[..., :3], [0.2989, 0.5870, 0.1140])

    # SSIM
    ssim_index = ssim(original_gray, reconstructed_gray, data_range=original_gray.max() - original_gray.min())

    # PSNR
    psnr_value = psnr(original_np, reconstructed_np, data_range=1.0)

    # MSE
    mse = np.mean((original_np - reconstructed_np) ** 2)

    print(f"SSIM: {ssim_index:.4f}")
    print(f"PSNR: {psnr_value:.2f} dB")
    print(f"MSE: {mse:.4f}")

    return ssim_index, psnr_value, mse


def calculate_fid(original_tensor, reconstructed_tensor, device):
    """Calculate FID between original and reconstructed images."""
    # Denormalize images (from [-1, 1] back to [0, 1])
    original = (original_tensor + 1) / 2
    reconstructed = (reconstructed_tensor + 1) / 2

    # Convert tensors to numpy arrays
    original_np = original.squeeze(0).permute(1, 2, 0).cpu().numpy()  # (H, W, C)
    reconstructed_np = reconstructed.squeeze(0).permute(1, 2, 0).cpu().numpy()  # (H, W, C)

    # Flatten and reshape arrays to (N, D) where D = H * W * C
    original_flat = original_np.reshape(-1, original_np.shape[-1]).astype(np.float64)
    reconstructed_flat = reconstructed_np.reshape(-1, reconstructed_np.shape[-1]).astype(np.float64)

    # Compute mean and covariance for original and reconstructed features
    mu1, sigma1 = original_flat.mean(axis=0), np.cov(original_flat, rowvar=False)
    mu2, sigma2 = reconstructed_flat.mean(axis=0), np.cov(reconstructed_flat, rowvar=False)

    # Calculate FID
    diff = mu1 - mu2
    covmean, _ = sqrtm(sigma1 @ sigma2, disp=False)

    # Handle numerical instability
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid_value = np.sum(diff**2) + np.trace(sigma1 + sigma2 - 2 * covmean)
    print(f"FID: {fid_value:.4f}")
    return fid_value

def generate_ad_map_with_visuals(original_tensor, reconstructed_tensor, ssim_index, psnr_value, mse, fid_value, save_path="demo_imgs/ad_map_overview.png"):
    """Generate an anomaly detection (AD) map and include visuals for better overview, with metrics in a single line on top."""
    # Denormalize images (from [-1, 1] back to [0, 1])
    original = (original_tensor + 1) / 2
    reconstructed = (reconstructed_tensor + 1) / 2

    # Calculate pixel-wise absolute difference
    ad_map = torch.abs(original - reconstructed).squeeze(0).mean(dim=0)  # Mean over channels (C)

    # Convert AD map to numpy array for visualization
    ad_map_np = ad_map.cpu().numpy()

    # Convert tensors to numpy arrays for visualization
    original_np = original.squeeze(0).permute(1, 2, 0).cpu().numpy()  # (H, W, C)
    reconstructed_np = reconstructed.squeeze(0).permute(1, 2, 0).cpu().numpy()  # (H, W, C)

    # Create a figure for combined visualization with additional space for the metrics
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))  # Increase figure height for better spacing
    
    # Display the original image
    axes[0].imshow(original_np)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Display the reconstructed image
    axes[1].imshow(reconstructed_np)
    axes[1].set_title("Reconstructed Image")
    axes[1].axis("off")

    # Display the anomaly detection map (AD Map)
    im = axes[2].imshow(ad_map_np, cmap='hot')
    axes[2].set_title("Anomaly Detection Map")
    axes[2].axis("off")

    # Add colorbar to the AD map
    cbar = fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    cbar.set_label("Pixel-wise Difference", rotation=270, labelpad=15)

    # Adjust the subplot to add space for the metrics line on top
    plt.subplots_adjust(top=0.85)  # Add more space at the top for metrics

    # Add metrics in a single line on the top margin of the figure
    metrics_text = (f"SSIM: {ssim_index:.4f}   "
                    f"PSNR: {psnr_value:.2f} dB   "
                    f"MSE: {mse:.4f}   "
                    f"FID: {fid_value:.4f}")
    
    # Place the metrics text at the top of the figure
    fig.text(0.5, 0.92, metrics_text, ha='center', va='top', fontsize=14, bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))

    # Save the overview figure
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Overview saved at {save_path}")


if __name__ == "__main__":
    # Setup
    init_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    network_config = {
        "batch_size": 128,
        "n_steps": 16,
        "dataset": "Cifar10",
        "in_channels": 3,
        "latent_dim": 128,
        "input_size": 32,
        "k": 20,
        "loss_func": "mmd",
        "mmd_type": 'rbf',
        "lr": 0.006,
        "distance_lambda": 0.001,
    }
    
    glv.init(network_config, devs=[0])
    distance_lambda = glv.network_config['distance_lambda']
    mmd_type = glv.network_config['mmd_type']
    input_size = glv.network_config['input_size']

    # Load the model
    net = esvae.ESVAE(device=init_device, distance_lambda=distance_lambda, mmd_type=mmd_type)
    net = net.to(init_device)
    checkpoint = torch.load('./data/ESVAE/checkpoint/CIFAR10/ESVAE-MVTEC-Hazelnut/checkpoint.pth', map_location=init_device)
    net.load_state_dict(checkpoint)

    # Load a single image
    image_path = "./data/MVTEC/hazelnut/test/crack/001.png" 
    image_tensor = load_single_image(image_path, input_size).to(init_device)

    # Reconstruct the image
    reconstructed_tensor = reconstruct_image(net, image_tensor)

    # Calculate metrics
    ssim_index, psnr_value, mse = calculate_metrics(image_tensor, reconstructed_tensor)
    fid_value = calculate_fid(image_tensor, reconstructed_tensor, device=init_device)

    # Generate the AD map and add metrics
    generate_ad_map_with_visuals(image_tensor, reconstructed_tensor, ssim_index, psnr_value, mse, fid_value, save_path="demo_imgs/ad_map_overview_with_metrics2.png")
