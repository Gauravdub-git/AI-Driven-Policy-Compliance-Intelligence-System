import pickle
from collections import defaultdict


# ─────────────────────────────────────────
# 1. BUILD CLAUSE LOOKUP BY ID
# ─────────────────────────────────────────

def build_clause_lookup(risk_dataset):
    """
    Build a dict: clause_id → full clause dict
    Allows fast access to any clause by its ID.
    """
    return {clause["clause_id"]: clause for clause in risk_dataset}


# ─────────────────────────────────────────
# 2. DIRECT CONTRADICTION DETECTION
# (obligation vs prohibition on same topic)
# ─────────────────────────────────────────

def detect_direct_contradictions(similar_pairs, clause_lookup, threshold=0.75):
    """
    Flag pairs where:
    - One clause is an obligation AND the other is a prohibition
    - Similarity score >= threshold

    These are the clearest conflicts:
    "must do X" vs "must not do X"

    Returns list of conflict dicts.
    """
    conflicts = []

    for pair in similar_pairs:
        if pair["similarity_score"] < threshold:
            continue

        type_a = pair["clause_a_type"]
        type_b = pair["clause_b_type"]

        is_contradiction = (
            (type_a == "obligation" and type_b == "prohibition") or
            (type_a == "prohibition" and type_b == "obligation")
        )

        if is_contradiction:
            conflicts.append({
                "conflict_type"    : "direct_contradiction",
                "severity"         : "high",
                "clause_a_id"      : pair["clause_a_id"],
                "clause_b_id"      : pair["clause_b_id"],
                "clause_a_text"    : pair["clause_a_text"],
                "clause_b_text"    : pair["clause_b_text"],
                "clause_a_type"    : type_a,
                "clause_b_type"    : type_b,
                "clause_a_section" : pair["clause_a_section"],
                "clause_b_section" : pair["clause_b_section"],
                "document_a"       : pair["document_a"],
                "document_b"       : pair["document_b"],
                "similarity_score" : pair["similarity_score"],
                "explanation"      : (
                    f"Direct contradiction: one clause is an obligation "
                    f"and the other is a prohibition on a similar topic "
                    f"(similarity={pair['similarity_score']})"
                )
            })

    return conflicts


# ─────────────────────────────────────────
# 3. OVERLAPPING OBLIGATION DETECTION
# (two obligations on same topic — possible conflict)
# ─────────────────────────────────────────

def detect_overlapping_obligations(similar_pairs, clause_lookup, threshold=0.80):
    """
    Flag pairs where:
    - Both clauses are obligations
    - Similarity score >= threshold (higher bar than contradiction)
    - They come from different sections or documents

    These may define conflicting requirements on the same topic.

    Returns list of conflict dicts.
    """
    conflicts = []

    for pair in similar_pairs:
        if pair["similarity_score"] < threshold:
            continue

        type_a = pair["clause_a_type"]
        type_b = pair["clause_b_type"]

        if type_a != "obligation" or type_b != "obligation":
            continue

        # Only flag if from different sections or different documents
        different_source = (
            pair["clause_a_section"] != pair["clause_b_section"] or
            pair["document_a"]       != pair["document_b"]
        )

        if different_source:
            conflicts.append({
                "conflict_type"    : "overlapping_obligation",
                "severity"         : "medium",
                "clause_a_id"      : pair["clause_a_id"],
                "clause_b_id"      : pair["clause_b_id"],
                "clause_a_text"    : pair["clause_a_text"],
                "clause_b_text"    : pair["clause_b_text"],
                "clause_a_type"    : type_a,
                "clause_b_type"    : type_b,
                "clause_a_section" : pair["clause_a_section"],
                "clause_b_section" : pair["clause_b_section"],
                "document_a"       : pair["document_a"],
                "document_b"       : pair["document_b"],
                "similarity_score" : pair["similarity_score"],
                "explanation"      : (
                    f"Overlapping obligations: two obligation clauses from "
                    f"different sources address the same topic "
                    f"(similarity={pair['similarity_score']}) — "
                    f"may impose conflicting requirements"
                )
            })

    return conflicts


# ─────────────────────────────────────────
# 4. PENALTY WITHOUT OBLIGATION DETECTION
# ─────────────────────────────────────────

def detect_penalty_without_obligation(risk_dataset):
    """
    Flag sections where a penalty clause exists
    but no obligation clause exists in the same section
    of the same document.

    Logic:
    - Group clauses by (document, section)
    - For each group, check if penalty exists without obligation

    Returns list of conflict dicts.
    """
    # Group clauses by (document, section)
    section_groups = defaultdict(list)

    for clause in risk_dataset:
        key = (clause["document"], clause["section"])
        section_groups[key].append(clause)

    conflicts = []

    for (document, section), clauses in section_groups.items():
        types_in_section = set(c["compliance_type"] for c in clauses)

        has_penalty    = "penalty"    in types_in_section
        has_obligation = "obligation" in types_in_section

        if has_penalty and not has_obligation:
            # Find the penalty clause to attach to conflict
            penalty_clauses = [
                c for c in clauses if c["compliance_type"] == "penalty"
            ]

            for penalty_clause in penalty_clauses:
                conflicts.append({
                    "conflict_type"    : "penalty_without_obligation",
                    "severity"         : "medium",
                    "clause_a_id"      : penalty_clause["clause_id"],
                    "clause_b_id"      : None,
                    "clause_a_text"    : penalty_clause["clause_text"],
                    "clause_b_text"    : None,
                    "clause_a_type"    : "penalty",
                    "clause_b_type"    : None,
                    "clause_a_section" : section,
                    "clause_b_section" : None,
                    "document_a"       : document,
                    "document_b"       : None,
                    "similarity_score" : None,
                    "explanation"      : (
                        f"Penalty defined in '{section}' of '{document}' "
                        f"but no corresponding obligation found in the same section — "
                        f"unclear what rule is being enforced"
                    )
                })

    return conflicts


# ─────────────────────────────────────────
# 5. FULL CONFLICT DETECTION PIPELINE
# ─────────────────────────────────────────

def detect_all_conflicts(similar_pairs, risk_dataset):
    """
    Run all three conflict detectors and combine results.
    Returns full conflict list sorted by severity.
    """
    clause_lookup = build_clause_lookup(risk_dataset)

    contradictions = detect_direct_contradictions(
        similar_pairs, clause_lookup, threshold=0.75
    )

    overlaps = detect_overlapping_obligations(
        similar_pairs, clause_lookup, threshold=0.80
    )

    penalty_gaps = detect_penalty_without_obligation(risk_dataset)

    all_conflicts = contradictions + overlaps + penalty_gaps

    # Sort: high severity first
    severity_order = {"high": 0, "medium": 1}
    all_conflicts  = sorted(
        all_conflicts,
        key=lambda x: severity_order.get(x["severity"], 2)
    )

    return all_conflicts


# ─────────────────────────────────────────
# 6. SUMMARY STATS
# ─────────────────────────────────────────

def print_conflict_summary(all_conflicts):
    """
    Print breakdown of detected conflicts by type and severity.
    """
    from collections import Counter

    type_counts     = Counter(c["conflict_type"] for c in all_conflicts)
    severity_counts = Counter(c["severity"]      for c in all_conflicts)

    print("\n--- CONFLICT DETECTION SUMMARY ---")
    print(f"Total conflicts detected  : {len(all_conflicts)}")
    print(f"Direct contradictions     : {type_counts.get('direct_contradiction', 0)}")
    print(f"Overlapping obligations   : {type_counts.get('overlapping_obligation', 0)}")
    print(f"Penalty without obligation: {type_counts.get('penalty_without_obligation', 0)}")
    print(f"High severity             : {severity_counts.get('high', 0)}")
    print(f"Medium severity           : {severity_counts.get('medium', 0)}")


# ─────────────────────────────────────────
# 7. SAVE CONFLICT RESULTS
# ─────────────────────────────────────────

def save_conflicts(all_conflicts, save_path="embeddings/conflicts.pkl"):
    with open(save_path, "wb") as f:
        pickle.dump(all_conflicts, f)
    print(f"\nConflicts saved to '{save_path}'")


def load_conflicts(save_path="embeddings/conflicts.pkl"):
    with open(save_path, "rb") as f:
        all_conflicts = pickle.load(f)
    print(f"Loaded {len(all_conflicts)} conflicts")
    return all_conflicts


if __name__ == "__main__":
    from similarity import load_similarity_results
    from risk_scoring import load_risk_dataset

    similar_pairs = load_similarity_results("embeddings/similar_pairs.pkl")
    risk_dataset = load_risk_dataset("embeddings/risk_dataset.pkl")

    all_conflicts = detect_all_conflicts(similar_pairs, risk_dataset)
    print_conflict_summary(all_conflicts)
    save_conflicts(all_conflicts)
