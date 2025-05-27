import torch
import math
import torch.nn as nn
import torchvision.models as models
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class PositionalEncoding(nn.Module):
    """Adds temporal positional encoding to the input sequence."""
    def __init__(self, d_model):
        super().__init__()
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        self.register_buffer('div_term', div_term)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        batch_size, seq_len, d_model = x.shape
        position = torch.arange(seq_len, device=x.device).unsqueeze(1)
        pe = torch.zeros(seq_len, d_model, device=x.device)
        pe[:, 0::2] = torch.sin(position * self.div_term)
        pe[:, 1::2] = torch.cos(position * self.div_term)
        return x + pe

class DeepFakeDetector(nn.Module):
    """Hybrid CNN-Transformer model for DeepFake detection."""
    def __init__(self):
        super().__init__()
        # Use a smaller ResNet variant
        self.cnn = models.resnet18(weights='DEFAULT')
        self.cnn_out_dim = 512  # ResNet18 output dimension
        
        # Remove the final fully connected layer
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-1])
        
        # Transformer encoder with smaller dimensions
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.cnn_out_dim,
            nhead=8,
            dim_feedforward=512,  # Reduced from 1024
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)  # Reduced from 2
        
        # Final classification layer
        self.classifier = nn.Sequential(
            nn.Linear(self.cnn_out_dim, 128),  # Reduced from 256
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        self.pos_encoder = PositionalEncoding(self.cnn_out_dim)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, num_frames, height, width, channels)
        Returns:
            Tensor of shape (batch_size, 1) containing classification probabilities
        """
        # Input shape: (batch_size, num_frames, height, width, channels)
        batch_size, num_frames, h, w, c = x.shape
        
        # Reshape and permute for CNN: (batch_size * num_frames, channels, height, width)
        x = x.view(-1, h, w, c)  # First reshape to (batch_size * num_frames, h, w, c)
        x = x.permute(0, 3, 1, 2)  # Then permute to (batch_size * num_frames, c, h, w)
        
        # Get CNN features
        features = self.cnn(x)  # (batch_size * num_frames, 512, 1, 1)
        features = features.squeeze(-1).squeeze(-1)  # (batch_size * num_frames, 512)
        
        # Reshape for transformer: (batch_size, num_frames, 512)
        features = features.view(batch_size, num_frames, -1)
        
        # Add positional encoding
        features = self.pos_encoder(features)
        
        # Transformer
        features = self.transformer(features)
        
        # Global average pooling
        features = features.mean(dim=1)  # (batch_size, 512)
        
        # Classification
        output = self.classifier(features)
        return output 