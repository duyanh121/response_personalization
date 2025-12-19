from style_features import extract_features

samples = [
    "LOL bro this is insane 😂😂!!!",
    "Thank you for your consideration. I look forward to your response."
]

for s in samples:
    print(s)
    print(extract_features(s))
    print("-" * 40)
