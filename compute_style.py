import numpy as np
from load_data import load_texts
from style_features import extract_features_batch

texts = load_texts(
    "reddit_conversations_v1.0_3turns.topical/"
    "reddit_conversations.3turns.train.topical.txt"
)

texts = texts[:100000]  # ⚠️ chỉ lấy 100k dòng cho nhanh

F = extract_features_batch(texts)

style_mean = F.mean(axis=0)
style_std = F.std(axis=0)

np.save("style_mean.npy", style_mean)
np.save("style_std.npy", style_std)

print("Style mean:", style_mean)
