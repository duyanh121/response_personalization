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


def SFS(text):
    f = extract_features(text)
    z = (f - style_mean) / (style_std + 1e-8)
    return norm(z)


def SFS(text):
    f = extract_features(text)
    return np.dot(f, style_mean) / (norm(f) * norm(style_mean) + 1e-6)

def style_loss(text):
    f = extract_features(text)
    return np.mean(((f - style_mean) / (style_std + 1e-6)) ** 2)
