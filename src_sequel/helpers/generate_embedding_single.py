import torch
from torch.nn.functional import normalize

def generate_embedding_single(text, model, tokenizer, device, max_length=512):
    model.eval()

    enc = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    with torch.no_grad():
        z = model(
            enc["input_ids"].to(device),
            enc["attention_mask"].to(device)
        )
        z = normalize(z, dim=1)

    return z.cpu().numpy()
