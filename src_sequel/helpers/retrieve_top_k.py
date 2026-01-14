from helpers.generate_embedding_single import generate_embedding_single

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

    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(records):
            results.append({
                "author": records[idx]["author"],
                "content": records[idx]["content"],
                "similarity": scores[0][i].item() if hasattr(scores[0][i], 'item') else float(scores[0][i]),
                "rank_by_similarity": i + 1,  # i=0 is rank 1 (most similar)
                "original_index": int(idx)
            })
    
    return results
