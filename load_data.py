def load_texts(txt_path):
    texts = []
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line:
                texts.append(line)
    return texts


if __name__ == "__main__":
    txt_path = (
        "reddit_conversations_v1.0_3turns.topical/"
        "reddit_conversations.3turns.train.topical.txt"
    )

    texts = load_texts(txt_path)
    print(f"Loaded {len(texts)} texts")
    print(texts[:3])
