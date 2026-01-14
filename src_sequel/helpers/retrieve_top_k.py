from rag.generate_embedding_single import generate_embedding_single

def retrieve_top_k(
    query_text,
    model,
    tokenizer,
    index,
    records,
    device,
    k=5
):
    q_emb = generate_embedding_single(query_text, model, tokenizer, device)

    scores, indices = index.search(q_emb, k)

    examples = [records[i]["content"] for i in indices[0]]
    return examples
