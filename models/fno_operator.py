"""
FNO-1D Operator Module for PhysMoP
Implements a Fourier Neural Operator along the time dimension for one-shot
motion prediction at arbitrary future query times.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SpectralConv1d(nn.Module):
    """1D Spectral Convolution Layer using FFT"""
    
    def __init__(self, in_channels, out_channels, modes, use_real_fft=True):
        super(SpectralConv1d, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes  # Number of Fourier modes to keep
        self.use_real_fft = use_real_fft
        
        # Fourier weights for lower modes
        self.scale = (1 / (in_channels * out_channels))
        if use_real_fft:
            # For real FFT, only need half the modes
            self.weights = nn.Parameter(
                self.scale * torch.rand(in_channels, out_channels, modes, 2, dtype=torch.float32)
            )
        else:
            self.weights = nn.Parameter(
                self.scale * torch.rand(in_channels, out_channels, modes, 2, dtype=torch.float32)
            )
    
    def complex_mul1d(self, input_fft, weights):
        """Complex multiplication in Fourier space"""
        # input_fft: (batch, in_channels, modes, 2) where last dim is [real, imag]
        # weights: (in_channels, out_channels, modes, 2)
        
        # Extract real and imaginary parts
        input_real = input_fft[..., 0]  # (B, in_c, modes)
        input_imag = input_fft[..., 1]  # (B, in_c, modes)
        weight_real = weights[..., 0]   # (in_c, out_c, modes)
        weight_imag = weights[..., 1]   # (in_c, out_c, modes)
        
        # Complex multiplication: (a + bi)(c + di) = (ac - bd) + (ad + bc)i
        # Einsum for matrix multiplication across channels
        out_real = torch.einsum('bim,iom->bom', input_real, weight_real) - \
                   torch.einsum('bim,iom->bom', input_imag, weight_imag)
        out_imag = torch.einsum('bim,iom->bom', input_real, weight_imag) + \
                   torch.einsum('bim,iom->bom', input_imag, weight_real)
        
        return torch.stack([out_real, out_imag], dim=-1)
    
    def forward(self, x):
        """
        Args:
            x: (batch, channels, time_steps)
        Returns:
            (batch, out_channels, time_steps)
        """
        batch_size, in_channels, time_steps = x.shape
        
        # Apply FFT
        if self.use_real_fft:
            x_ft = torch.fft.rfft(x, dim=-1, norm='ortho')
        else:
            x_ft = torch.fft.fft(x, dim=-1, norm='ortho')
        
        # Convert complex to real representation [real, imag]
        x_ft_real = torch.stack([x_ft.real, x_ft.imag], dim=-1)
        
        # Determine actual number of modes to use (minimum of specified modes and available frequencies)
        actual_modes = min(self.modes, x_ft_real.shape[2])
        
        # Truncate to the first 'actual_modes' frequencies
        x_ft_low = x_ft_real[..., :actual_modes, :]  # (B, in_c, actual_modes, 2)
        
        # Use only the corresponding weights
        weights_low = self.weights[..., :actual_modes, :]  # (in_c, out_c, actual_modes, 2)
        
        # Apply spectral weights
        out_ft_low = self.complex_mul1d(x_ft_low, weights_low)  # (B, out_c, actual_modes, 2)
        
        # Reconstruct full frequency spectrum (pad higher frequencies with zeros)
        if self.use_real_fft:
            num_freqs = time_steps // 2 + 1
        else:
            num_freqs = time_steps
            
        out_ft = torch.zeros(batch_size, self.out_channels, num_freqs, 2, 
                            dtype=x.dtype, device=x.device)
        out_ft[..., :actual_modes, :] = out_ft_low
        
        # Convert back to complex
        out_ft_complex = torch.complex(out_ft[..., 0], out_ft[..., 1])
        
        # Apply inverse FFT
        if self.use_real_fft:
            out = torch.fft.irfft(out_ft_complex, n=time_steps, dim=-1, norm='ortho')
        else:
            out = torch.fft.ifft(out_ft_complex, dim=-1, norm='ortho').real
        
        return out
        out_ft_complex = torch.complex(out_ft[..., 0], out_ft[..., 1])
        
        # Apply inverse FFT
        if self.use_real_fft:
            out = torch.fft.irfft(out_ft_complex, n=time_steps, dim=-1, norm='ortho')
        else:
            out = torch.fft.ifft(out_ft_complex, dim=-1, norm='ortho').real
        
        return out


class FNO1DBlock(nn.Module):
    """Single FNO layer with spectral and local operations"""
    
    def __init__(self, channels, modes, use_real_fft=True):
        super(FNO1DBlock, self).__init__()
        
        self.spectral_conv = SpectralConv1d(channels, channels, modes, use_real_fft)
        self.local_conv = nn.Conv1d(channels, channels, 1)  # Point-wise
        self.activation = nn.GELU()
        
    def forward(self, x):
        """
        Args:
            x: (batch, channels, time)
        """
        # Spectral path
        x1 = self.spectral_conv(x)
        
        # Local path
        x2 = self.local_conv(x)
        
        # Combine and activate
        out = self.activation(x1 + x2)
        
        # Residual connection
        out = out + x
        
        return out


class ContextEncoder(nn.Module):
    """Encode history window and conditions into context representation"""
    
    def __init__(self, input_dim, hidden_dim, hist_length, normalize=True):
        super(ContextEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.hist_length = hist_length
        self.normalize = normalize
        
        # Normalization statistics (will be computed during training)
        self.register_buffer('input_mean', torch.zeros(input_dim))
        self.register_buffer('input_std', torch.ones(input_dim))
        
        # Encoding layers
        self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def update_stats(self, x):
        """Update running statistics for normalization"""
        if self.training and self.normalize:
            # x: (B, T, D)
            mean = x.mean(dim=(0, 1))
            std = x.std(dim=(0, 1)) + 1e-8
            
            # Exponential moving average
            momentum = 0.1
            self.input_mean = (1 - momentum) * self.input_mean + momentum * mean
            self.input_std = (1 - momentum) * self.input_std + momentum * std
    
    def forward(self, hist, cond=None):
        """
        Args:
            hist: (batch, hist_length, input_dim) - historical poses
            cond: optional conditions (not used in v1)
        Returns:
            (batch, hist_length, hidden_dim)
        """
        # Update normalization statistics
        if self.normalize:
            self.update_stats(hist)
            hist_norm = (hist - self.input_mean) / self.input_std
        else:
            hist_norm = hist
        
        # Encode
        x = self.fc_in(hist_norm)
        x = self.layer_norm(x)
        x = F.gelu(x)
        
        return x


class TimeGridBuilder(nn.Module):
    """Build time grid with positional encoding for query times"""
    
    def __init__(self, embed_type='fourier', embed_dim=4):
        super(TimeGridBuilder, self).__init__()
        
        self.embed_type = embed_type
        self.embed_dim = embed_dim
        
        if embed_type == 'learned':
            # Learnable time embedding
            self.time_embed = nn.Linear(1, embed_dim)
    
    def forward(self, query_times, dt=1/25):
        """
        Args:
            query_times: (batch, K) - query time points in frames or seconds
            dt: time step in seconds (default 1/25 for 25 FPS)
        Returns:
            (batch, K, embed_dim) - time grid with positional encoding
        """
        batch_size, K = query_times.shape
        
        # Normalize to [0, 1]
        t_min = query_times.min(dim=1, keepdim=True)[0]
        t_max = query_times.max(dim=1, keepdim=True)[0]
        t_norm = (query_times - t_min) / (t_max - t_min + 1e-8)
        
        if self.embed_type == 'fourier':
            # Fourier positional encoding: [sin(t), cos(t), t, t^2]
            t_norm_expanded = t_norm.unsqueeze(-1)  # (B, K, 1)
            
            embeddings = []
            embeddings.append(torch.sin(2 * np.pi * t_norm_expanded))
            embeddings.append(torch.cos(2 * np.pi * t_norm_expanded))
            embeddings.append(t_norm_expanded)
            embeddings.append(t_norm_expanded ** 2)
            
            time_grid = torch.cat(embeddings[:self.embed_dim], dim=-1)
            
        elif self.embed_type == 'learned':
            t_norm_expanded = t_norm.unsqueeze(-1)
            time_grid = self.time_embed(t_norm_expanded)
            
        else:  # 'none'
            time_grid = t_norm.unsqueeze(-1)
        
        return time_grid


class FNOOperator(nn.Module):
    """
    Fourier Neural Operator for one-shot motion prediction.
    Maps history window to arbitrary future query times.
    """
    
    def __init__(
        self,
        input_dim=63,           # Pose parameter dimension
        hidden_dim=128,         # FNO hidden dimension
        num_layers=4,           # Number of FNO blocks
        modes=16,               # Fourier modes to keep
        hist_length=25,         # History window
        time_embed='fourier',   # Time encoding type
        time_embed_dim=4,       # Time encoding dimension
        use_real_fft=True,      # Use real FFT
        normalize=True          # Normalize inputs/outputs
    ):
        super(FNOOperator, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.modes = modes
        self.hist_length = hist_length
        self.normalize = normalize
        
        # Context encoder
        self.context_encoder = ContextEncoder(
            input_dim, hidden_dim, hist_length, normalize
        )
        
        # Time grid builder
        self.time_grid_builder = TimeGridBuilder(time_embed, time_embed_dim)
        
        # Lift: combine context and time information
        self.lift = nn.Linear(hidden_dim + time_embed_dim, hidden_dim)
        
        # FNO blocks
        self.fno_blocks = nn.ModuleList([
            FNO1DBlock(hidden_dim, modes, use_real_fft)
            for _ in range(num_layers)
        ])
        
        # Project to output
        self.project = nn.Linear(hidden_dim, input_dim)
        
        # Output normalization
        self.register_buffer('output_mean', torch.zeros(input_dim))
        self.register_buffer('output_std', torch.ones(input_dim))
    
    def update_output_stats(self, x):
        """Update output normalization statistics"""
        if self.training and self.normalize:
            mean = x.mean(dim=(0, 1))
            std = x.std(dim=(0, 1)) + 1e-8
            
            momentum = 0.1
            self.output_mean = (1 - momentum) * self.output_mean + momentum * mean
            self.output_std = (1 - momentum) * self.output_std + momentum * std
    
    def forward(self, hist, query_times, cond=None):
        """
        Args:
            hist: (batch, hist_length, input_dim) - historical poses
            query_times: (batch, K) - future query time points
            cond: optional conditions
        Returns:
            (batch, K, input_dim) - predicted poses at query times
        """
        batch_size = hist.shape[0]
        K = query_times.shape[1]
        
        # Encode context from history
        context = self.context_encoder(hist, cond)  # (B, H, hidden_dim)
        
        # Build time grid with positional encoding
        time_grid = self.time_grid_builder(query_times)  # (B, K, time_embed_dim)
        
        # Expand context to all query times using mean pooling
        # Simple approach: broadcast context statistics
        context_agg = context.mean(dim=1, keepdim=True)  # (B, 1, hidden_dim)
        context_expanded = context_agg.expand(-1, K, -1)  # (B, K, hidden_dim)
        
        # Combine context and time
        x = torch.cat([context_expanded, time_grid], dim=-1)  # (B, K, hidden_dim + time_embed_dim)
        x = self.lift(x)  # (B, K, hidden_dim)
        
        # Transpose for FNO (expects channels dimension)
        x = x.transpose(1, 2)  # (B, hidden_dim, K)
        
        # Apply FNO blocks along time dimension
        for fno_block in self.fno_blocks:
            x = fno_block(x)
        
        # Transpose back
        x = x.transpose(1, 2)  # (B, K, hidden_dim)
        
        # Project to output dimension
        output = self.project(x)  # (B, K, input_dim)
        
        return output
