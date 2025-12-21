import numpy as np
from load_data import load_texts
from style_features import extract_features_batch

# ================= CONFIG =================
DATA_PATH = "data/processed/mr_train.csv"
TEXT_COLUMN = "response"
MAX_TEXTS = 100000       # giới hạn cho nhanh
# =========================================


texts = load_texts(DATA_PATH, text_col=TEXT_COLUMN)

print(f"Loaded {len(texts)} texts")

if MAX_TEXTS is not None:
    texts = texts[:MAX_TEXTS]
    print(f"Using {len(texts)} texts for style computation")

F = extract_features_batch(texts)

style_mean = F.mean(axis=0)
style_std = F.std(axis=0)

np.save("style_mean.npy", style_mean)
np.save("style_std.npy", style_std)

print("Style mean:", style_mean)
print("Style std:", style_std)
