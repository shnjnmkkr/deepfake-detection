import torch
import math
import torch.nn as nn
import torchvision.models as models
from einops import rearrange, repeat
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

class SpatialTemporalEncoder(nn.Module):
    def __init__(self, dim=512, depth=4, heads=8, dim_head=64, mlp_dim=1024, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(dim),
                TransformerEncoderLayer(
                    d_model=dim,
                    nhead=heads,
                    dim_feedforward=mlp_dim,
                    dropout=dropout,
                    batch_first=True
                ),
                nn.LayerNorm(dim),
                TransformerEncoderLayer(
                    d_model=dim,
                    nhead=heads,
                    dim_feedforward=mlp_dim,
                    dropout=dropout,
                    batch_first=True
                )
            ]))

    def forward(self, x):
        for spatial_norm, spatial_trans, temporal_norm, temporal_trans in self.layers:
            # Spatial attention
            x = spatial_norm(x)
            x = spatial_trans(x)
            # Temporal attention 
            x = temporal_norm(x)
            x = temporal_trans(x)
        return x

class DeepFakeDetector(nn.Module):
    """ViViT-based DeepFake detection model"""
    def __init__(self, 
                 num_frames=15,
                 patch_size=16,
                 dim=512,
                 depth=4,
                 heads=8,
                 mlp_dim=1024,
                 pool='cls',
                 channels=3,
                 dim_head=64,
                 dropout=0.1,
                 emb_dropout=0.1):
        super().__init__()
        
        # Image patches
        self.patch_size = patch_size
        self.num_frames = num_frames
        
        # Backbone CNN (ResNet50 with (2+1)D convolutions)
        self.backbone = models.video.r2plus1d_18(pretrained=True)
        self.backbone.fc = nn.Identity()  # Remove classifier
        
        # Patch embedding
        self.to_patch_embedding = nn.Sequential(
            nn.Linear(512, dim),
            nn.LayerNorm(dim),
        )

        # Position embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        # Transformer encoder
        self.transformer = SpatialTemporalEncoder(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            dropout=dropout
        )

        # MLP head
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, 1)
        )

        self.pool = pool

    def forward(self, x):
        # Input shape: (batch, frames, channels, height, width)
        b, f, c, h, w = x.shape
        
        # Extract features using backbone
        x = x.view(-1, c, h, w)  # Reshape for CNN
        x = self.backbone(x)
        
        # Create patches
        x = self.to_patch_embedding(x)
        x = x.view(b, f, -1)  # Reshape back to sequence
        
        # Add cls token
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add position embedding
        x += self.pos_embedding[:, :(x.shape[1])]
        x = self.dropout(x)
        
        # Apply transformer
        x = self.transformer(x)
        
        # Pool and classify
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        return self.mlp_head(x) 