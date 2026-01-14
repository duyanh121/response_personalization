import torch
from helpers.generate_embedding_single import generate_embedding_single

def retrieve_weighted_samples(
    query_text,
    model,
    tokenizer,
    index,
    records,
    device,
    k=5,
    temperature=0.1
):
    """
    Retrieve k samples using distance-weighted random sampling.
    Closer candidates are more likely, but not guaranteed.
    
    Returns: List of dictionaries with:
        - author: author of the message
        - content: message content
        - similarity: cosine similarity score
        - rank_by_similarity: rank among ALL candidates (1 = most similar overall)
        - rank_by_probability: rank among ALL candidates (1 = highest probability)
        - probability: sampling probability used
    """
    # Generate query embedding
    q_emb = generate_embedding_single(query_text, model, tokenizer, device)
    
    # Search for more candidates than needed
    search_k = min(100, index.ntotal)
    distances, candidate_indices = index.search(q_emb, search_k)
    
    # Get similarities from FAISS (IndexFlatIP gives cosine similarity)
    similarities = torch.tensor(distances[0], device=device)
    
    # Apply temperature scaling and softmax to get probabilities
    probabilities = torch.softmax(similarities / temperature, dim=0)
    
    # Sample k unique indices
    sampled_indices = torch.multinomial(
        probabilities, 
        min(k, len(probabilities)), 
        replacement=False
    )
    
    # Get selected indices from FAISS results
    selected_indices = candidate_indices[0][sampled_indices.cpu().numpy()]
    selected_similarities = similarities[sampled_indices]
    selected_probabilities = probabilities[sampled_indices]
    
    # Sort ALL candidates by similarity to get global ranks
    # argsort gives indices in ascending order, so we reverse for descending
    all_sim_sorted_idx = torch.argsort(similarities, descending=True)
    
    # Sort ALL candidates by probability to get global probability ranks
    all_prob_sorted_idx = torch.argsort(probabilities, descending=True)
    
    # Create results with global ranking info
    results = []
    for i, idx in enumerate(selected_indices):
        if idx < len(records):
            # Find similarity rank among ALL candidates
            # where() returns a tuple, we take the first element
            sim_rank_tensor = torch.where(all_sim_sorted_idx == sampled_indices[i])[0]
            sim_rank = sim_rank_tensor.item() + 1 if len(sim_rank_tensor) > 0 else search_k + 1
            
            # Find probability rank among ALL candidates
            prob_rank_tensor = torch.where(all_prob_sorted_idx == sampled_indices[i])[0]
            prob_rank = prob_rank_tensor.item() + 1 if len(prob_rank_tensor) > 0 else search_k + 1
            
            results.append({
                "author": records[idx]["author"],
                "content": records[idx]["content"],
                "similarity": selected_similarities[i].item(),
                "probability": selected_probabilities[i].item(),
                "rank_by_similarity": sim_rank,  # e.g., could be 1, 3, 7, 15, 42
                "rank_by_probability": prob_rank,  # e.g., could be 1, 2, 5, 8, 12
                "original_index": int(idx)
            })
    
    # Sort results by similarity rank for cleaner output
    results.sort(key=lambda x: x["rank_by_similarity"])
    
    return results