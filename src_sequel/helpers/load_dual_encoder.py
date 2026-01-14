import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CanineModel, CanineTokenizer
from classes.dual_encoder import DualEncoderModel
import os

def load_dual_encoder_model(model_dir):
    """
    Load a dual-encoder model saved with your format.
    
    Args:
        model_dir: Directory containing:
            - dual_encoder.pt (model state dict)
            - projection_head.pt (projection heads)
            - HuggingFace model files (config.json, pytorch_model.bin)
            - Tokenizer files
    
    Returns:
        model, tokenizer, device
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Create model instance (same architecture as training)
    model = DualEncoderModel(
        text_dim=256,  # Default, adjust if you changed it
        style_dim=64,   # Default, adjust if you changed it
        dropout=0.1
    )
    
    # 2. Load the main model weights
    model_path = os.path.join(model_dir, "dual_encoder.pt")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        print(f"Loaded model weights from {model_path}")
    else:
        print("Model weights not found, initializing fresh")
    
    # 3. Load projection heads (optional - they're already in model weights)
    proj_path = os.path.join(model_dir, "projection_head.pt")
    if os.path.exists(proj_path):
        proj_dict = torch.load(proj_path, map_location="cpu")
        model.content_proj.load_state_dict(proj_dict['content_proj'])
        model.style_proj.load_state_dict(proj_dict['style_proj'])
        model.style_mlp.load_state_dict(proj_dict['style_mlp'])
        print(f"Loaded projection heads from {proj_path}")
    
    # 4. Load tokenizer from the directory
    try:
        tokenizer = CanineTokenizer.from_pretrained(model_dir)
        print(f"Loaded tokenizer from {model_dir}")
    except:
        print("Could not load tokenizer from save dir, using default")
        tokenizer = CanineTokenizer.from_pretrained("google/canine-s")
    
    # 5. Move to device and set to eval mode
    model.to(device)
    model.eval()
    
    print(f"Model loaded successfully")
    print(f"Device: {device}")
    print(f"Model dir: {model_dir}")
    
    return model, tokenizer, device
