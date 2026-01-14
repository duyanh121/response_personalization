import torch
from torch.nn.functional import normalize
from tqdm import tqdm


def generate_embeddings(
    texts,
    model,
    tokenizer,
    batch_size=8,
    max_length=512,
    device="cuda"
):
    all_embeddings = []

    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i + batch_size]

        enc = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )

        with torch.no_grad():
            z = model(
                enc["input_ids"].to(device),
                enc["attention_mask"].to(device)
            )
            z = normalize(z, dim=1).cpu()

        all_embeddings.append(z)

        del enc, z
        torch.cuda.empty_cache()

    return torch.cat(all_embeddings, dim=0)

