import numpy as np
from numpy.linalg import norm
from style_features import extract_features

import os
import numpy as np
from numpy.linalg import norm
from style_features import extract_features

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STYLE_MEAN_PATH = os.path.join(BASE_DIR, "style_mean.npy")
STYLE_STD_PATH = os.path.join(BASE_DIR, "style_std.npy")

style_mean = np.load(STYLE_MEAN_PATH)
style_std = np.load(STYLE_STD_PATH)


# def SFS(text):
#     f = extract_features(text)
#     z = (f - style_mean) / (style_std + 1e-8)
#     return norm(z)


def SFS(text):
    f = extract_features(text)
    return np.dot(f, style_mean) / (norm(f) * norm(style_mean) + 1e-6)


def style_loss(text):
    f = extract_features(text)
    return np.mean(((f - style_mean) / (style_std + 1e-6)) ** 2)


# def style_loss(text):
#     """
#     Cosine distance between text's style and user's style.
#     Range: 0 (identical) to 2 (opposite)
#     """
#     f = extract_features(text)
    
#     # Cosine distance = 1 - cosine similarity
#     dot = np.dot(f, style_mean)
#     norm_f = norm(f)
#     norm_mean = norm(style_mean)
    
#     if norm_f == 0 or norm_mean == 0:
#         return 1.0  # Max distance if either vector is zero
    
#     cosine_sim = dot / (norm_f * norm_mean)
#     return 1.0 - cosine_sim  # Convert similarity to distance