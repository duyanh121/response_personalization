import numpy as np
from numpy.linalg import norm
from style_features import extract_features

style_mean = np.load("style_mean.npy")
style_std = np.load("style_std.npy")

def SFS(text):
    f = extract_features(text)
    return np.dot(f, style_mean) / (norm(f) * norm(style_mean) + 1e-6)

def style_loss(text):
    f = extract_features(text)
    return np.mean(((f - style_mean) / (style_std + 1e-6)) ** 2)
