import torch
print(torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import seaborn as sns
import scanpy as sc
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize']=(4,4)
from scanpy import AnnData
import squidpy as sq

import torch
import torch.nn as nn
import torch.nn.functional as F

class AutoencoderWithGeneExpression(nn.Module):
    def __init__(self, num_genes):
        super(AutoencoderWithGeneExpression, self).__init__()
        self.num_genes = num_genes  # Number of genes to predict (N)
        
        # Encoder: same as original
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.flatten_size = 128 * 37 * 31
        self.fc1 = nn.Linear(self.flatten_size, 1024)
        self.fc2 = nn.Linear(1024, self.flatten_size)
        
        # Decoder: upsample to original resolution
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3+num_genes , kernel_size=3, stride=2, padding=1, output_padding=1)
            # Output channels: 3 for RGB image + num_genes for gene expression
        )

    
    def forward(self, x):
        # Encode
        x = self.encoder(x)  # [1, 128, 37, 31]
        x = x.view(-1, self.flatten_size)  # Flatten
        x = self.fc1(x)  # [1, 1024]
        embedding = x
        x = self.fc2(x)  # [1, 128*37*31]
        x = x.view(-1, 128, 37, 31)  # Reshape
        
        # Decode
        x = self.decoder(x)  # [1, 3 + num_genes, 600, 508]
        x = F.interpolate(x, size=(600, 508), mode='bilinear', align_corners=False)
        # Split outputs
        reconstructed_image = torch.sigmoid(x[:, :3, :, :])  # [1, 3, 600, 508]
        gene_expression_map = torch.sigmoid(x[:, 3:, :, :])  # [1, num_genes, 600, 508]
        
        return embedding, reconstructed_image, gene_expression_map
    
#SSIM Loss Function
def ssim(img1, img2, window_size=11, size_average=True):
    #img1 and img2 are [batch, channels, height, width]
    channel = img1.size(1)
    
    #Gaussian window
    window = torch.ones((1, 1, window_size, window_size)) / (window_size ** 2)
    window = window.to(img1.device)
    window = window.repeat(channel, 1, 1, 1)  # Match number of channels
    
    #means
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    #variances and covariance
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2
    
    #constants
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    #ssim formula
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    if size_average:
        return ssim_map.mean()
    return ssim_map

class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
    
    def forward(self, img1, img2):
        return 1 - ssim(img1, img2, self.window_size, self.size_average)
criterion = SSIMLoss(window_size=11)

def compute_loss(reconstructed_image, gene_expression_map, original_image, spot_data, spot_size, lambda_ge=1,no_gene_loss=False):
    # Image reconstruction loss
    #loss_img = F.mse_loss(reconstructed_image, original_image)
    loss_img=criterion(reconstructed_image, original_image)
    if not no_gene_loss:
        # Gene expression loss
        loss_ge = 0.0
        for x_k, y_k, g_k in spot_data:
            # Define the region R_k (e.g., S x S square centered at (x_k, y_k))
            x_start = max(0, int(x_k - spot_size // 2))
            x_end = min(507, int(x_k + spot_size // 2))
            y_start = max(0, int(y_k - spot_size // 2))
            y_end = min(599, int(y_k + spot_size // 2))
    #         print(gene_expression_map.shape)
            # Extract predicted gene expression for the region
            pred_region = gene_expression_map[:, :, y_start:y_end, x_start:x_end]  # [1, N, height, width]
            pred_avg = pred_region.mean(dim=(2, 3))  # [1, N], average over the region

            # Measured gene expression (convert to tensor)
            g_k_tensor = torch.tensor(g_k, dtype=torch.float32).unsqueeze(0).to(device)  # [1, N]
            loss=F.mse_loss(pred_avg, g_k_tensor)
            loss_ge += loss

        loss_ge = loss_ge / len(spot_data)  # Average over all spots
    else:
        loss_ge=loss_img*0
    # Combined loss
    total_loss = loss_img + lambda_ge * loss_ge
    return total_loss, loss_img, loss_ge