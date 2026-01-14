import torch.nn as nn

class DualEncoderAsSingle(nn.Module):
    """Wraps DualEncoder to return only style embeddings."""
    def __init__(self, dual_encoder):
        super().__init__()
        self.dual_encoder = dual_encoder
    
    def forward(self, input_ids, attention_mask=None):
        # Return only style embeddings
        _, style_emb = self.dual_encoder(input_ids, attention_mask)
        return style_emb
    