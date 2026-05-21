import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ─────────────────────────────────────────
# 2. SEMANTIC SEARCH
# ─────────────────────────────────────────

def semantic_search(query, model, embeddings, clause_texts, clause_metadata, top_k=5):
    """
    Given a natural language query:
    - Encode query into embedding
    - Compute cosine similarity against all clause embeddings
    - Return top_k most relevant clauses with scores

    Args:
        query         : natural language question or keyword string
        model         : loaded SentenceTransformer model
        embeddings    : np.ndarray of shape (num_clauses, 384)
        clause_texts  : list of clause strings
        clause_metadata: list of clause metadata dicts
        top_k         : number of top results to return

    Returns:
        list of dicts with clause info and similarity score
    """
    # Encode the query into a vector
    query_embedding = model.encode([query], convert_to_numpy=True)
    # shape: (1, 384)

    # Compute cosine similarity between query and all clauses
    scores = cosine_similarity(query_embedding, embeddings)[0]
    # shape: (num_clauses,)

    # Get indices of top_k highest scores
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append({
            "clause_id"       : clause_metadata[idx]["clause_id"],
            "document"        : clause_metadata[idx]["document"],
            "section"         : clause_metadata[idx]["section"],
            "compliance_type" : clause_metadata[idx]["compliance_type"],
            "clause_text"     : clause_texts[idx],
            "similarity_score": round(float(scores[idx]), 4)
        })

    return results


# ─────────────────────────────────────────
# 3. CLAUSE-TO-CLAUSE SIMILARITY DETECTION
# ─────────────────────────────────────────

def detect_similar_clause_pairs(embeddings, clause_texts, clause_metadata,
                                 threshold=0.75, max_pairs=50):
    """
    Find all pairs of clauses with cosine similarity above threshold.
    Skips self-comparisons (same clause_id).
    Skips duplicate pairs (A,B) and (B,A).

    Args:
        threshold  : minimum similarity score to flag a pair
        max_pairs  : cap on number of pairs returned (avoids flooding)

    Returns:
        list of dicts, each representing a similar pair
    """
    # Compute full similarity matrix
    # shape: (num_clauses, num_clauses)
    similarity_matrix = cosine_similarity(embeddings)

    similar_pairs = []

    num_clauses = len(clause_texts)

    for i in range(num_clauses):
        for j in range(i + 1, num_clauses):   # j > i avoids duplicates and self
            score = similarity_matrix[i][j]

            if score >= threshold:
                similar_pairs.append({
                    "clause_a_id"     : clause_metadata[i]["clause_id"],
                    "clause_b_id"     : clause_metadata[j]["clause_id"],
                    "clause_a_text"   : clause_texts[i],
                    "clause_b_text"   : clause_texts[j],
                    "clause_a_type"   : clause_metadata[i]["compliance_type"],
                    "clause_b_type"   : clause_metadata[j]["compliance_type"],
                    "clause_a_section": clause_metadata[i]["section"],
                    "clause_b_section": clause_metadata[j]["section"],
                    "document_a"      : clause_metadata[i]["document"],
                    "document_b"      : clause_metadata[j]["document"],
                    "similarity_score": round(float(score), 4)
                })

        if len(similar_pairs) >= max_pairs:
            break

    # Sort by score descending
    similar_pairs = sorted(similar_pairs, key=lambda x: x["similarity_score"], reverse=True)

    return similar_pairs[:max_pairs]


# ─────────────────────────────────────────
# 4. SAVE SIMILARITY RESULTS
# ─────────────────────────────────────────

def save_similarity_results(similar_pairs, save_path="embeddings/similar_pairs.pkl"):
    """
    Save detected similar pairs to disk.
    Used by Step 7 (conflict detection).
    """
    with open(save_path, "wb") as f:
        pickle.dump(similar_pairs, f)
    print(f"\nSaved {len(similar_pairs)} similar pairs to '{save_path}'")


def load_similarity_results(save_path="embeddings/similar_pairs.pkl"):
    """
    Load similar pairs from disk.
    """
    with open(save_path, "rb") as f:
        similar_pairs = pickle.load(f)
    print(f"Loaded {len(similar_pairs)} similar pairs")
    return similar_pairs


if __name__ == "__main__":
    from embeddings import load_embeddings, load_embedding_model

    embeddings, clause_texts, clause_metadata = load_embeddings("embeddings/")
    model = load_embedding_model()

    print(f"Loaded {len(clause_texts)} clauses with embeddings of shape {embeddings.shape}")

    # Semantic Search Demo
    query = "data retention and storage limitation"
    print("\n--- SEMANTIC SEARCH RESULTS ---")
    print(f"Query: '{query}'\n")

    search_results = semantic_search(
        query, model, embeddings, clause_texts, clause_metadata, top_k=5
    )

    for rank, result in enumerate(search_results, 1):
        print(f"Rank {rank} | Score: {result['similarity_score']}")
        print(f"  Document : {result['document']}")
        print(f"  Section  : {result['section']}")
        print(f"  Type     : {result['compliance_type']}")
        print(f"  Text     : {result['clause_text'][:150]}...")
        print()

    print("\n--- SIMILAR CLAUSE PAIRS (threshold=0.75) ---")
    similar_pairs = detect_similar_clause_pairs(
        embeddings, clause_texts, clause_metadata,
        threshold=0.75,
        max_pairs=50
    )
    print(f"Total similar pairs found: {len(similar_pairs)}\n")
    save_similarity_results(similar_pairs)
