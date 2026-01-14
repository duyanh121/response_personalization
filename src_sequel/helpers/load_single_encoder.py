import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CanineModel, CanineTokenizer
from classes.single_encoder import SingleEncoder
import os

def load_single_encoder_model(model_dir):
    """
    Load a trained SingleEncoder model from your save directory.
    
    Args:
        model_dir: Directory containing saved model files
    
    Returns:
        model, tokenizer, device
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model checkpoint
    checkpoint_path = os.path.join(model_dir, "single_encoder.pt")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Get model configuration
    proj_dim = checkpoint.get('proj_dim', 128)
    model_name = checkpoint.get('model_name', 'google/canine-s')
    
    # Initialize model with correct configuration
    model = SingleEncoder(model_name=model_name, proj_dim=proj_dim)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load tokenizer
    tokenizer = CanineTokenizer.from_pretrained(model_dir)
    
    # Move to device
    model.to(device)
    model.eval()
    
    print(f"SingleEncoder loaded from {model_dir}")
    print(f"Base model: {model_name}")
    print(f"Projection dimension: {proj_dim}")
    print(f"Device: {device}")
    
    return model, tokenizer, device