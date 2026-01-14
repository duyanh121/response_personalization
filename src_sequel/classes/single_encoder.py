from transformers import CanineModel
import torch.nn as nn
import torch.nn.functional as F

class SingleEncoder(nn.Module):
    def __init__(self, model_name="google/canine-s", proj_dim=128):
        super().__init__()
        self.encoder = CanineModel.from_pretrained(model_name)
        hidden_dim = self.encoder.config.hidden_size

        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, proj_dim)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls = outputs.last_hidden_state[:, 0]  # [CLS]
        z = self.proj(cls)
        z = F.normalize(z, dim=1)
        return z
