import torch

class ConversationDataset(torch.utils.data.Dataset):
    def __init__(self, rows, tokenizer, max_length=512):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

        # map authors to integer labels
        authors = sorted(set(r["Author"] for r in rows))
        self.author2id = {a: i for i, a in enumerate(authors)}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        enc = self.tokenizer(
            row["Content"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "author": self.author2id[row["Author"]],
            "content": row["Content"]
        }
