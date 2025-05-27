import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class PositionalEncoding(nn.Module):
    """Adds temporal positional encoding to the input sequence."""
    def __init__(self, d_model, max_len=15):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return x

class DeepFakeDetector(nn.Module):
    """Hybrid CNN-Transformer model for DeepFake detection."""
    def __init__(self, num_frames=15, use_sr=False):
        super().__init__()
        # CNN Backbone (ResNet50)
        resnet = models.resnet50(pretrained=True)
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])  # Remove final FC layer
        self.cnn_out_dim = 2048  # ResNet50 feature dimension
        
        # Transformer components
        self.pos_encoder = PositionalEncoding(self.cnn_out_dim)
        encoder_layers = TransformerEncoderLayer(
            d_model=self.cnn_out_dim,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=3)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.cnn_out_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        self.use_sr = use_sr

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, num_frames, channels, height, width)
        Returns:
            Tensor of shape (batch_size, 1) containing classification probabilities
        """
        batch_size, num_frames, c, h, w = x.size()
        
        # Reshape for CNN processing
        x = x.view(-1, c, h, w)  # (batch_size * num_frames, c, h, w)
        
        # Extract features using CNN
        features = self.cnn(x)  # (batch_size * num_frames, 2048, 1, 1)
        features = features.squeeze(-1).squeeze(-1)  # (batch_size * num_frames, 2048)
        
        # Reshape for transformer
        features = features.view(batch_size, num_frames, -1)  # (batch_size, num_frames, 2048)
        features = features.permute(1, 0, 2)  # (num_frames, batch_size, 2048)
        
        # Add positional encoding
        features = self.pos_encoder(features)
        
        # Apply transformer
        transformer_out = self.transformer_encoder(features)
        
        # Use the output of the last frame for classification
        last_frame_features = transformer_out[-1]  # (batch_size, 2048)
        
        # Classify
        output = self.classifier(last_frame_features)
        return output 