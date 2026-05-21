import numpy as np
import pickle
import os
from sentence_transformers import SentenceTransformer

# ─────────────────────────────────────────
# 1. LOAD MODEL
# ─────────────────────────────────────────

# all-MiniLM-L6-v2 is lightweight, fast, and well suited
# for semantic similarity tasks
MODEL_NAME = "all-MiniLM-L6-v2"


def load_embedding_model(model_name=MODEL_NAME):
    """Load and return the sentence-transformer model."""
    return SentenceTransformer(model_name)


# ─────────────────────────────────────────
# 2. GENERATE EMBEDDINGS
# ─────────────────────────────────────────

def generate_embeddings(enriched_dataset, model):
    """
    Generate a semantic embedding for each clause.
    Returns:
    - embeddings      : np.ndarray of shape (num_clauses, 384)
    - clause_texts    : list of clause strings (parallel to embeddings)
    - clause_metadata : list of clause dicts without text (for reference)
    """
    clause_texts    = [clause["clause_text"] for clause in enriched_dataset]
    clause_metadata = [
        {
            "clause_id"       : clause["clause_id"],
            "document"        : clause["document"],
            "section"         : clause["section"],
            "compliance_type" : clause["compliance_type"],
            "entities"        : clause["entities"]
        }
        for clause in enriched_dataset
    ]

    print(f"\nGenerating embeddings for {len(clause_texts)} clauses...")

    # batch_size=32 is safe for most machines
    # show_progress_bar gives visibility on large documents
    embeddings = model.encode(
        clause_texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    print(f"Embeddings shape: {embeddings.shape}")
    # Expected: (num_clauses, 384)

    return embeddings, clause_texts, clause_metadata


# ─────────────────────────────────────────
# 3. STORE EMBEDDINGS
# ─────────────────────────────────────────

def save_embeddings(embeddings, clause_texts, clause_metadata, save_path="embeddings/"):
    """
    Save embeddings and associated data to disk.
    Three files:
    - embeddings.npy        : numpy array of vectors
    - clause_texts.pkl      : list of raw clause strings
    - clause_metadata.pkl   : list of clause metadata dicts
    """
    os.makedirs(save_path, exist_ok=True)

    np.save(os.path.join(save_path, "embeddings.npy"), embeddings)

    with open(os.path.join(save_path, "clause_texts.pkl"), "wb") as f:
        pickle.dump(clause_texts, f)

    with open(os.path.join(save_path, "clause_metadata.pkl"), "wb") as f:
        pickle.dump(clause_metadata, f)

    print(f"\nSaved to '{save_path}':")
    print(f"  → embeddings.npy      : {embeddings.shape}")
    print(f"  → clause_texts.pkl    : {len(clause_texts)} entries")
    print(f"  → clause_metadata.pkl : {len(clause_metadata)} entries")


def load_embeddings(save_path="embeddings/"):
    """
    Load embeddings and associated data back from disk.
    Use this in Step 5 onwards instead of recomputing.
    """
    embeddings = np.load(os.path.join(save_path, "embeddings.npy"))

    with open(os.path.join(save_path, "clause_texts.pkl"), "rb") as f:
        clause_texts = pickle.load(f)

    with open(os.path.join(save_path, "clause_metadata.pkl"), "rb") as f:
        clause_metadata = pickle.load(f)

    print(f"Loaded embeddings: {embeddings.shape}")
    return embeddings, clause_texts, clause_metadata


# ─────────────────────────────────────────
# 4. QUICK SANITY CHECK
# ─────────────────────────────────────────

def sanity_check(embeddings, clause_texts):
    """
    Basic checks to confirm embeddings are valid.
    """
    print("\n--- SANITY CHECK ---")

    # Shape check
    assert embeddings.shape[0] == len(clause_texts), \
        "Mismatch: number of embeddings != number of clauses"
    print(f"Shape check passed  : {embeddings.shape[0]} embeddings for {len(clause_texts)} clauses")

    # Dimension check
    assert embeddings.shape[1] == 384, \
        f"Unexpected embedding dimension: {embeddings.shape[1]}"
    print(f"Dimension check passed : 384 dimensions per embedding")

    # NaN check
    assert not np.isnan(embeddings).any(), \
        "NaN values found in embeddings"
    print(f"NaN check passed    : No NaN values")

    # Sample values
    print(f"\nSample embedding (clause 0, first 5 values):")
    print(f"  {embeddings[0][:5]}")


if __name__ == "__main__":
    from step1_preprocessing import preprocess_documents
    from step2_segmentation import segment_all_documents
    from knowledge_extraction import extract_knowledge

    dataset = preprocess_documents("documents/")
    clause_dataset = segment_all_documents(dataset)
    enriched_dataset = extract_knowledge(clause_dataset)
    model = load_embedding_model()
    print(f"Model loaded: {MODEL_NAME}")

    embeddings, clause_texts, clause_metadata = generate_embeddings(
        enriched_dataset, model
    )

    sanity_check(embeddings, clause_texts)
    save_embeddings(embeddings, clause_texts, clause_metadata)
