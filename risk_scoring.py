import pickle
import numpy as np

# ─────────────────────────────────────────
# 1. KEYWORD LISTS FOR RULE-BASED SCORING
# ─────────────────────────────────────────

HIGH_RISK_KEYWORDS = [
    "criminal", "imprisonment", "prosecution",
    "breach", "violation", "infringement",
    "revocation", "suspension", "termination",
    "liable", "unlawful", "illegal"
]

MEDIUM_RISK_KEYWORDS = [
    "notify", "disclose", "restrict", "limitation",
    "consent", "authorize", "review", "audit",
    "monitor", "report", "ensure", "comply"
]


# ─────────────────────────────────────────
# 2. RULE-BASED BASE RISK
# ─────────────────────────────────────────

def rule_based_risk(clause):
    """
    Assign base risk level using:
    - compliance_type from Step 3
    - keyword presence in clause text

    Returns:
        risk_level : "low" | "medium" | "high"
        reason     : explanation string
    """
    text_lower       = clause["clause_text"].lower()
    compliance_type  = clause["compliance_type"]

    # Check high-risk keywords first
    for keyword in HIGH_RISK_KEYWORDS:
        if keyword in text_lower:
            return "high", f"Contains high-risk keyword: '{keyword}'"

    # Check medium-risk keywords
    for keyword in MEDIUM_RISK_KEYWORDS:
        if keyword in text_lower:
            return "medium", f"Contains medium-risk keyword: '{keyword}'"

    # Fall back to compliance type
    if compliance_type == "penalty":
        return "high", "Clause is classified as a penalty"

    if compliance_type == "prohibition":
        return "medium", "Clause is classified as a prohibition"

    if compliance_type == "obligation":
        return "medium", "Clause is classified as an obligation"

    return "low", "No high-risk indicators found"


# ─────────────────────────────────────────
# 3. BUILD SIMILARITY LOOKUP PER CLAUSE
# ─────────────────────────────────────────

def build_similarity_lookup(similar_pairs):
    """
    From the similar_pairs list (Step 5), build a lookup:
    clause_id → highest similarity score with any other clause

    Used by AI-assisted layer to check if a clause
    has high overlap with others.
    """
    lookup = {}  # clause_id: best_score

    for pair in similar_pairs:
        id_a  = pair["clause_a_id"]
        id_b  = pair["clause_b_id"]
        score = pair["similarity_score"]

        if id_a not in lookup or lookup[id_a] < score:
            lookup[id_a] = score

        if id_b not in lookup or lookup[id_b] < score:
            lookup[id_b] = score

    return lookup


# ─────────────────────────────────────────
# 4. AI-ASSISTED RISK ADJUSTMENT
# ─────────────────────────────────────────

RISK_ORDER = {"low": 0, "medium": 1, "high": 2}
RISK_LABEL = {0: "low", 1: "medium", 2: "high"}

def ai_assisted_adjustment(clause_id, base_risk, similarity_lookup):
    """
    Adjust base risk upward if the clause has high semantic
    similarity with other clauses (potential overlap/conflict).

    Similarity >= 0.85 → escalate by 2 levels max
    Similarity 0.75–0.84 → escalate by 1 level max

    Returns:
        adjusted_risk  : "low" | "medium" | "high"
        adjustment_reason : explanation string or None
    """
    best_score = similarity_lookup.get(clause_id, 0.0)

    base_level = RISK_ORDER[base_risk]

    if best_score >= 0.85:
        new_level = min(base_level + 2, 2)
        if new_level > base_level:
            return (
                RISK_LABEL[new_level],
                f"AI adjustment: high semantic overlap detected "
                f"(similarity={round(best_score, 4)}) — risk escalated"
            )

    elif best_score >= 0.75:
        new_level = min(base_level + 1, 2)
        if new_level > base_level:
            return (
                RISK_LABEL[new_level],
                f"AI adjustment: moderate semantic overlap detected "
                f"(similarity={round(best_score, 4)}) — risk escalated"
            )

    return base_risk, None


# ─────────────────────────────────────────
# 5. FULL RISK SCORING PIPELINE
# ─────────────────────────────────────────

def score_risks(enriched_dataset, similar_pairs):
    """
    For each clause:
    1. Apply rule-based scoring → base risk + reason
    2. Apply AI-assisted adjustment → final risk + adjustment reason
    3. Produce explainable output

    Returns risk_dataset: enriched_dataset + risk fields
    """
    similarity_lookup = build_similarity_lookup(similar_pairs)
    risk_dataset      = []

    for clause in enriched_dataset:
        clause_id = clause["clause_id"]

        # Layer 1: Rule-based
        base_risk, rule_reason = rule_based_risk(clause)

        # Layer 2: AI-assisted
        final_risk, ai_reason = ai_assisted_adjustment(
            clause_id, base_risk, similarity_lookup
        )

        # Build explainable output
        explanation = rule_reason
        if ai_reason:
            explanation = explanation + " | " + ai_reason

        risk_dataset.append({
            # carry forward all Step 3 fields
            "clause_id"       : clause_id,
            "document"        : clause["document"],
            "section"         : clause["section"],
            "clause_text"     : clause["clause_text"],
            "compliance_type" : clause["compliance_type"],
            "entities"        : clause["entities"],
            # new fields added in Step 6
            "base_risk"       : base_risk,
            "final_risk"      : final_risk,
            "risk_explanation": explanation
        })

    return risk_dataset


# ─────────────────────────────────────────
# 6. SUMMARY STATS
# ─────────────────────────────────────────

def print_risk_summary(risk_dataset):
    """
    Print count breakdown of final risk levels.
    """
    from collections import Counter

    risk_counts = Counter(c["final_risk"] for c in risk_dataset)

    print("\n--- RISK SCORING SUMMARY ---")
    print(f"Total clauses : {len(risk_dataset)}")
    print(f"High risk     : {risk_counts.get('high', 0)}")
    print(f"Medium risk   : {risk_counts.get('medium', 0)}")
    print(f"Low risk      : {risk_counts.get('low', 0)}")


# ─────────────────────────────────────────
# 7. SAVE RISK DATASET
# ─────────────────────────────────────────

def save_risk_dataset(risk_dataset, save_path="embeddings/risk_dataset.pkl"):
    with open(save_path, "wb") as f:
        pickle.dump(risk_dataset, f)
    print(f"\nRisk dataset saved to '{save_path}'")


def load_risk_dataset(save_path="embeddings/risk_dataset.pkl"):
    with open(save_path, "rb") as f:
        risk_dataset = pickle.load(f)
    print(f"Loaded risk dataset: {len(risk_dataset)} clauses")
    return risk_dataset


if __name__ == "__main__":
    from knowledge_extraction import extract_knowledge
    from step2_segmentation import segment_all_documents
    from step1_preprocessing import preprocess_documents
    from similarity import load_similarity_results

    dataset = preprocess_documents("documents/")
    clause_dataset = segment_all_documents(dataset)
    enriched_dataset = extract_knowledge(clause_dataset)
    similar_pairs = load_similarity_results("embeddings/similar_pairs.pkl")

    risk_dataset = score_risks(enriched_dataset, similar_pairs)
    print_risk_summary(risk_dataset)
    save_risk_dataset(risk_dataset)
