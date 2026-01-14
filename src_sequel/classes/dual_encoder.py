# Cell 2: Dual-Encoder Model Definition

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CanineModel, CanineTokenizer
import os

class DualEncoderModel(nn.Module):
    """
    Dual-encoder model that learns both content and style representations.
    - Text encoder: for content/semantic understanding
    - Style encoder: learns to extract style patterns automatically
    """
    
    def __init__(self, 
                 text_dim=256,      # Content embedding dimension
                 style_dim=64,      # Style embedding dimension  
                 dropout=0.1):
        super().__init__()
        
        # Shared base encoder (CANINE)
        self.base_encoder = CanineModel.from_pretrained("google/canine-s")
        hidden_size = self.base_encoder.config.hidden_size  # 768
        
        # CONTENT projection head (for semantic similarity) 
        self.content_proj = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, text_dim)
        )
        
        # STYLE projection head (learns to extract style automatically)
        self.style_proj = nn.Sequential(
            nn.Linear(hidden_size, 384),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(384),
            nn.Linear(384, style_dim)
        )
        
        # Optional: Add small MLP to make style more expressive
        self.style_mlp = nn.Sequential(
            nn.Linear(style_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, style_dim)
        )
    
    def forward(self, input_ids, attention_mask=None):
        """
        Returns both content and style embeddings.
        """
        # Get base representations
        outputs = self.base_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token for both representations
        pooled_output = outputs.pooler_output  # (batch, hidden_size)
        
        # Extract content embedding (for semantic search)
        content_emb = self.content_proj(pooled_output)
        
        # Extract style embedding (learns style patterns automatically)
        style_emb = self.style_proj(pooled_output)
        style_emb = self.style_mlp(style_emb)  # Optional: make more expressive
        
        content_emb = F.normalize(content_emb, dim=-1)
        style_emb = F.normalize(style_emb, dim=-1)
        
        
        return content_emb, style_emb
    
    def encode_content(self, input_ids, attention_mask=None):
        """Get only content embedding (for retrieval)."""
        content_emb, _ = self.forward(input_ids, attention_mask)
        return F.normalize(content_emb, dim=-1)
    
    def encode_style(self, input_ids, attention_mask=None):
        """Get only style embedding (for style analysis)."""
        _, style_emb = self.forward(input_ids, attention_mask)
        return F.normalize(style_emb, dim=-1)