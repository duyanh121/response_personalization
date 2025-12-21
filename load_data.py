import pandas as pd


def load_texts(csv_path, text_col="target"):
    """
    Load texts from a CSV file.

    Args:
        csv_path (str): path to csv file
        text_col (str): column name containing text

    Returns:
        List[str]: list of texts
    """
    df = pd.read_csv(csv_path)

    if text_col not in df.columns:
        raise ValueError(
            f"Column '{text_col}' not found. Available columns: {list(df.columns)}"
        )

    texts = (
        df[text_col]
        .dropna()
        .astype(str)
        .str.strip()
        .tolist()
    )

    return texts


if __name__ == "__main__":
    csv_path = "data/sample/mr_train.csv"
    texts = load_texts(csv_path, text_col="target")

    print(f"Loaded {len(texts)} texts")
    print(texts[:3])
