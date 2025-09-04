import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class VariationalConv3d(nn.Module):
    """Variational 3D Convolution layer with reparameterization trick."""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, prior_std=1.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Weight parameters (mean and log variance)
        self.weight_mu = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        self.weight_logvar = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        
        # Bias parameters
        self.bias_mu = nn.Parameter(torch.randn(out_channels))
        self.bias_logvar = nn.Parameter(torch.randn(out_channels))
        
        # Prior parameters
        self.prior_std = prior_std
        
        # Initialize parameters
        self.reset_parameters()
        
    def reset_parameters(self):
        # Initialize weight means with Xavier initialization
        nn.init.xavier_uniform_(self.weight_mu)
        # Initialize log variances to small negative values (small initial variance)
        nn.init.constant_(self.weight_logvar, -5.0)
        
        # Initialize bias
        nn.init.zeros_(self.bias_mu)
        nn.init.constant_(self.bias_logvar, -5.0)
    
    def forward(self, x):
        if self.training:
            # Sample weights using reparameterization trick
            weight_std = torch.exp(0.5 * self.weight_logvar)
            weight_eps = torch.randn_like(self.weight_mu)
            weight = self.weight_mu + weight_eps * weight_std
            
            # Sample bias
            bias_std = torch.exp(0.5 * self.bias_logvar)
            bias_eps = torch.randn_like(self.bias_mu)
            bias = self.bias_mu + bias_eps * bias_std
        else:
            # Use mean values during inference
            weight = self.weight_mu
            bias = self.bias_mu
            
        return F.conv3d(x, weight, bias, self.stride, self.padding)
    
    def kl_divergence(self):
        """Compute KL divergence between posterior and prior."""
        # KL divergence for weights
        weight_var = torch.exp(self.weight_logvar)
        weight_kl = 0.5 * torch.sum(
            (self.weight_mu**2 + weight_var) / (self.prior_std**2) - 
            self.weight_logvar + 
            2 * math.log(self.prior_std) - 1
        )
        
        # KL divergence for bias
        bias_var = torch.exp(self.bias_logvar)
        bias_kl = 0.5 * torch.sum(
            (self.bias_mu**2 + bias_var) / (self.prior_std**2) - 
            self.bias_logvar + 
            2 * math.log(self.prior_std) - 1
        )
        
        return weight_kl + bias_kl

def variational_convbn(in_ch, out_ch, dropout=0.0, prior_std=1.0):
    """Variational convolution + batch norm + activation block."""
    layers = [
        VariationalConv3d(in_ch, out_ch, prior_std=prior_std),
        nn.GroupNorm(8, out_ch),
        nn.GELU()
    ]
    if dropout > 0.0:
        layers.append(nn.Dropout3d(p=dropout))
    else:
        layers.append(nn.Identity())
    return nn.Sequential(*layers)

class VariationalUNet3D(nn.Module):
    """3D U-Net with variational (Bayesian) layers for uncertainty quantification."""
    
    def __init__(self, in_ch, out_ch, base_ch=32, dropout=0.0, prior_std=1.0, kl_weight=1e-5):
        super().__init__()
        self.dropout_p = dropout
        self.prior_std = prior_std
        self.kl_weight = kl_weight
        
        # Encoder
        self.enc1 = nn.Sequential(
            variational_convbn(in_ch, base_ch, dropout, prior_std),
            variational_convbn(base_ch, base_ch, dropout, prior_std)
        )
        self.pool1 = nn.MaxPool3d(2)
        
        self.enc2 = nn.Sequential(
            variational_convbn(base_ch, base_ch*2, dropout, prior_std),
            variational_convbn(base_ch*2, base_ch*2, dropout, prior_std)
        )
        self.pool2 = nn.MaxPool3d(2)
        
        self.enc3 = nn.Sequential(
            variational_convbn(base_ch*2, base_ch*4, dropout, prior_std),
            variational_convbn(base_ch*4, base_ch*4, dropout, prior_std)
        )
        self.pool3 = nn.MaxPool3d(2)
        
        # Bottleneck
        self.bott = nn.Sequential(
            variational_convbn(base_ch*4, base_ch*8, dropout, prior_std),
            variational_convbn(base_ch*8, base_ch*8, dropout, prior_std)
        )
        
        # Decoder
        self.up3 = nn.ConvTranspose3d(base_ch*8, base_ch*4, 2, stride=2)
        self.dec3 = nn.Sequential(
            variational_convbn(base_ch*8, base_ch*4, dropout, prior_std),
            variational_convbn(base_ch*4, base_ch*4, dropout, prior_std)
        )
        
        self.up2 = nn.ConvTranspose3d(base_ch*4, base_ch*2, 2, stride=2)
        self.dec2 = nn.Sequential(
            variational_convbn(base_ch*4, base_ch*2, dropout, prior_std),
            variational_convbn(base_ch*2, base_ch*2, dropout, prior_std)
        )
        
        self.up1 = nn.ConvTranspose3d(base_ch*2, base_ch, 2, stride=2)
        self.dec1 = nn.Sequential(
            variational_convbn(base_ch*2, base_ch, dropout, prior_std),
            variational_convbn(base_ch, base_ch, dropout, prior_std)
        )
        
        # Output layer (also variational)
        self.out = VariationalConv3d(base_ch, out_ch, kernel_size=1, padding=0, prior_std=prior_std)
        
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        
        # Bottleneck
        b = self.bott(p3)
        
        # Decoder
        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], 1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], 1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], 1)
        d1 = self.dec1(d1)
        
        return self.out(d1)
    
    def raw_kl_divergence(self):
        """Compute raw KL divergence for all variational layers (no weighting)."""
        total_kl = 0.0
        for module in self.modules():
            if isinstance(module, VariationalConv3d):
                total_kl += module.kl_divergence()
        return total_kl
    
    def kl_divergence(self):
        """Compute weighted KL divergence (legacy method for backward compatibility)."""
        return self.kl_weight * self.raw_kl_divergence()
    
    def sample_predictions(self, x, n_samples=100):
        """Generate multiple predictions by sampling from posterior."""
        self.train()  # Enable sampling
        predictions = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.forward(x)
                predictions.append(pred.cpu().numpy())
                
        return torch.stack([torch.tensor(p) for p in predictions])
