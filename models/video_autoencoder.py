"""
Video Autoencoder with ConvLSTM for Temporal Anomaly Detection.
Developed by Kuldeep Choksi

Architecture:
    Input: Video sequence [B, T, C, H, W]
        ↓
    Encoder (per frame): Extracts spatial features
        ↓
    ConvLSTM: Captures temporal dependencies
        ↓
    Decoder (per frame): Reconstructs frames
        ↓
    Output: Reconstructed sequence [B, T, C, H, W]

The model learns normal temporal patterns. Anomalies cause high
reconstruction error because the model never learned those patterns.
"""

import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    """
    Convolutional LSTM Cell.
    
    Unlike regular LSTM that works on vectors, ConvLSTM works on 
    spatial feature maps. It preserves spatial structure while 
    learning temporal dependencies.
    
    Args:
        input_dim: Number of input channels
        hidden_dim: Number of hidden channels
        kernel_size: Size of convolutional kernel
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2
        
        # Combined convolution for all gates (more efficient)
        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,  # 4 gates: input, forget, cell, output
            kernel_size=kernel_size,
            padding=padding,
            bias=True
        )
    
    def forward(self, x, hidden_state):
        """
        Args:
            x: Input tensor [B, C, H, W]
            hidden_state: Tuple of (h, c) each [B, hidden_dim, H, W]
            
        Returns:
            h_next: Next hidden state
            c_next: Next cell state
        """
        h_cur, c_cur = hidden_state
        
        # Concatenate input and hidden state
        combined = torch.cat([x, h_cur], dim=1)
        
        # Compute all gates at once
        gates = self.conv(combined)
        
        # Split into individual gates
        i, f, g, o = torch.split(gates, self.hidden_dim, dim=1)
        
        # Apply activations
        i = torch.sigmoid(i)  # Input gate
        f = torch.sigmoid(f)  # Forget gate
        g = torch.tanh(g)     # Cell gate
        o = torch.sigmoid(o)  # Output gate
        
        # Update cell and hidden state
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next
    
    def init_hidden(self, batch_size, height, width, device):
        """Initialize hidden state with zeros."""
        h = torch.zeros(batch_size, self.hidden_dim, height, width, device=device)
        c = torch.zeros(batch_size, self.hidden_dim, height, width, device=device)
        return (h, c)


class ConvLSTM(nn.Module):
    """
    Multi-layer Convolutional LSTM.
    
    Processes a sequence of feature maps and outputs a sequence
    of hidden states capturing temporal information.
    """
    
    def __init__(
        self, 
        input_dim: int, 
        hidden_dims: list, 
        kernel_size: int = 3,
        num_layers: int = 2,
        batch_first: bool = True,
        return_all_layers: bool = False
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims if isinstance(hidden_dims, list) else [hidden_dims] * num_layers
        self.num_layers = len(self.hidden_dims)
        self.batch_first = batch_first
        self.return_all_layers = return_all_layers
        
        # Create ConvLSTM cells for each layer
        cells = []
        for i in range(self.num_layers):
            cur_input_dim = input_dim if i == 0 else self.hidden_dims[i - 1]
            cells.append(ConvLSTMCell(cur_input_dim, self.hidden_dims[i], kernel_size))
        
        self.cells = nn.ModuleList(cells)
    
    def forward(self, x, hidden_state=None):
        """
        Args:
            x: Input sequence [B, T, C, H, W] if batch_first else [T, B, C, H, W]
            hidden_state: Optional initial hidden state
            
        Returns:
            output: Output sequence from last layer
            hidden_states: List of final hidden states for each layer
        """
        if not self.batch_first:
            x = x.permute(1, 0, 2, 3, 4)  # [T, B, C, H, W] -> [B, T, C, H, W]
        
        b, t, c, h, w = x.size()
        device = x.device
        
        # Initialize hidden states if not provided
        if hidden_state is None:
            hidden_state = self._init_hidden(b, h, w, device)
        
        # Process through layers
        layer_output_list = []
        layer_hidden_list = []
        
        cur_input = x
        
        for layer_idx, cell in enumerate(self.cells):
            h, c = hidden_state[layer_idx]
            output_inner = []
            
            # Process each timestep
            for t_idx in range(t):
                h, c = cell(cur_input[:, t_idx], (h, c))
                output_inner.append(h)
            
            # Stack outputs [B, T, C, H, W]
            layer_output = torch.stack(output_inner, dim=1)
            cur_input = layer_output
            
            layer_output_list.append(layer_output)
            layer_hidden_list.append((h, c))
        
        if self.return_all_layers:
            return layer_output_list, layer_hidden_list
        else:
            return layer_output_list[-1], layer_hidden_list[-1]
    
    def _init_hidden(self, batch_size, height, width, device):
        """Initialize hidden states for all layers."""
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cells[i].init_hidden(batch_size, height, width, device))
        return init_states


class VideoEncoder(nn.Module):
    """
    Encoder for video frames - processes each frame independently.
    Same architecture as image encoder but applied per-frame.
    """
    
    def __init__(self, in_channels: int = 3, latent_dim: int = 128):
        super().__init__()
        
        self.encoder = nn.Sequential(
            # Block 1: 256 -> 128
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 2: 128 -> 64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 3: 64 -> 32
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 4: 32 -> 16
            nn.Conv2d(128, latent_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(latent_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, 2),
        )
    
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] single frame or [B, T, C, H, W] sequence
        """
        if x.dim() == 5:
            # Sequence: process each frame
            b, t, c, h, w = x.size()
            x = x.view(b * t, c, h, w)
            encoded = self.encoder(x)
            _, c_out, h_out, w_out = encoded.size()
            encoded = encoded.view(b, t, c_out, h_out, w_out)
            return encoded
        else:
            return self.encoder(x)


class VideoDecoder(nn.Module):
    """
    Decoder for video frames - reconstructs each frame independently.
    """
    
    def __init__(self, out_channels: int = 3, latent_dim: int = 128):
        super().__init__()
        
        self.decoder = nn.Sequential(
            # Block 1: 16 -> 32
            nn.ConvTranspose2d(latent_dim, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Block 2: 32 -> 64
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Block 3: 64 -> 128
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Block 4: 128 -> 256
            nn.ConvTranspose2d(32, out_channels, kernel_size=2, stride=2),
            nn.Tanh()
        )
    
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] single frame or [B, T, C, H, W] sequence
        """
        if x.dim() == 5:
            b, t, c, h, w = x.size()
            x = x.view(b * t, c, h, w)
            decoded = self.decoder(x)
            _, c_out, h_out, w_out = decoded.size()
            decoded = decoded.view(b, t, c_out, h_out, w_out)
            return decoded
        else:
            return self.decoder(x)


class VideoAutoencoder(nn.Module):
    """
    Complete Video Autoencoder with ConvLSTM for temporal modeling.
    
    Architecture:
        Encoder -> ConvLSTM -> Decoder
    
    The ConvLSTM learns temporal patterns in the latent space,
    enabling detection of temporal anomalies.
    """
    
    def __init__(
        self, 
        in_channels: int = 3, 
        latent_dim: int = 128,
        lstm_hidden_dim: int = 128,
        lstm_num_layers: int = 2
    ):
        super().__init__()
        
        self.encoder = VideoEncoder(in_channels, latent_dim)
        
        self.convlstm = ConvLSTM(
            input_dim=latent_dim,
            hidden_dims=[lstm_hidden_dim] * lstm_num_layers,
            kernel_size=3,
            num_layers=lstm_num_layers,
            batch_first=True,
            return_all_layers=False
        )
        
        # Project LSTM output back to latent dim if different
        self.proj = nn.Conv2d(lstm_hidden_dim, latent_dim, kernel_size=1) \
            if lstm_hidden_dim != latent_dim else nn.Identity()
        
        self.decoder = VideoDecoder(in_channels, latent_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        Forward pass for video sequence.
        
        Args:
            x: Input sequence [B, T, C, H, W]
            
        Returns:
            Reconstructed sequence [B, T, C, H, W]
        """
        # Encode each frame
        encoded = self.encoder(x)  # [B, T, latent_dim, H', W']
        
        # Process through ConvLSTM
        lstm_out, _ = self.convlstm(encoded)  # [B, T, lstm_hidden, H', W']
        
        # Project if needed
        b, t, c, h, w = lstm_out.size()
        lstm_out = lstm_out.view(b * t, c, h, w)
        projected = self.proj(lstm_out)
        projected = projected.view(b, t, -1, h, w)
        
        # Decode each frame
        reconstructed = self.decoder(projected)  # [B, T, C, H, W]
        
        return reconstructed
    
    def get_reconstruction_error(self, x, per_frame: bool = False, per_pixel: bool = False):
        """
        Compute reconstruction error for anomaly scoring.
        
        Args:
            x: Input sequence [B, T, C, H, W]
            per_frame: Return error per frame [B, T]
            per_pixel: Return error map per frame [B, T, 1, H, W]
            
        Returns:
            Anomaly scores
        """
        recon = self.forward(x)
        
        # MSE error
        error = (x - recon) ** 2
        
        if per_pixel:
            # Average across channels, keep spatial
            error = error.mean(dim=2, keepdim=True)  # [B, T, 1, H, W]
            return error
        elif per_frame:
            # Average across channels and spatial
            error = error.mean(dim=[2, 3, 4])  # [B, T]
            return error
        else:
            # Global score per sequence
            error = error.mean(dim=[1, 2, 3, 4])  # [B]
            return error


# Quick test
if __name__ == '__main__':
    print("Video Autoencoder with ConvLSTM")
    print("Developed by Kuldeep Choksi")
    print()
    
    model = VideoAutoencoder()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Test forward pass
    x = torch.randn(2, 16, 3, 256, 256)  # [B, T, C, H, W]
    print(f"Input shape: {x.shape}")
    
    recon = model(x)
    print(f"Output shape: {recon.shape}")
    
    # Test error computation
    error_seq = model.get_reconstruction_error(x)
    error_frame = model.get_reconstruction_error(x, per_frame=True)
    error_pixel = model.get_reconstruction_error(x, per_pixel=True)
    
    print(f"Sequence error shape: {error_seq.shape}")
    print(f"Per-frame error shape: {error_frame.shape}")
    print(f"Per-pixel error shape: {error_pixel.shape}")