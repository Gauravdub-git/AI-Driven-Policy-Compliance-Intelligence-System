import spacy
import re

_nlp = None


def _load_spacy_model():
    """Load spaCy model if available; otherwise fall back to blank English pipeline."""
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        return spacy.blank("en")


def get_nlp():
    """Lazily initialize spaCy model to avoid import-time overhead."""
    global _nlp
    if _nlp is None:
        _nlp = _load_spacy_model()
    return _nlp

# ─────────────────────────────────────────
# 1. COMPLIANCE ELEMENT KEYWORDS
# ─────────────────────────────────────────

OBLIGATION_KEYWORDS = [
    "shall", "must", "required to", "is obliged",
    "has to", "have to", "obligated to", "is required"
]

PROHIBITION_KEYWORDS = [
    "must not", "shall not", "prohibited",
    "is forbidden", "may not", "cannot", "not permitted",
    "not allowed"
]

PENALTY_KEYWORDS = [
    "fine", "penalty", "sanction", "liable",
    "imprisonment", "punishable", "subject to",
    "prosecution", "infringement"
]


# ─────────────────────────────────────────
# 2. COMPLIANCE ELEMENT CLASSIFIER
# ─────────────────────────────────────────

def classify_compliance_element(clause_text):
    """
    Classify a clause as obligation, prohibition, penalty, or unclassified.
    Checks prohibition before obligation to avoid misclassifying
    'shall not' as an obligation due to 'shall' match.
    Returns label string.
    """
    text_lower = clause_text.lower()

    # Check prohibition first (must not / shall not before must / shall)
    for keyword in PROHIBITION_KEYWORDS:
        if keyword in text_lower:
            return "prohibition"

    for keyword in PENALTY_KEYWORDS:
        if keyword in text_lower:
            return "penalty"

    for keyword in OBLIGATION_KEYWORDS:
        if keyword in text_lower:
            return "obligation"

    return "unclassified"


# ─────────────────────────────────────────
# 3. NAMED ENTITY RECOGNITION
# ─────────────────────────────────────────

# spaCy entity labels we care about per module requirements:
# ORG   → organizations
# DATE  → dates
# LAW   → laws, acts, regulations

RELEVANT_ENTITY_TYPES = {"ORG", "DATE", "LAW"}

def extract_entities(clause_text):
    """
    Run spaCy NER on a clause.
    Returns dict with lists of organizations, dates, and laws found.
    """
    doc = get_nlp()(clause_text)

    entities = {
        "organizations" : [],
        "dates"         : [],
        "laws"          : []
    }

    label_map = {
        "ORG"  : "organizations",
        "DATE" : "dates",
        "LAW"  : "laws"
    }

    for ent in doc.ents:
        if ent.label_ in RELEVANT_ENTITY_TYPES:
            key = label_map[ent.label_]
            # Avoid duplicates within same clause
            if ent.text.strip() not in entities[key]:
                entities[key].append(ent.text.strip())

    return entities


# ─────────────────────────────────────────
# 4. FULL KNOWLEDGE EXTRACTION PIPELINE
# ─────────────────────────────────────────

def extract_knowledge(clause_dataset):
    """
    For each clause:
    - Classify compliance element type
    - Extract named entities
    Returns enriched clause dataset.
    """
    enriched_dataset = []

    for clause in clause_dataset:
        clause_text = clause["clause_text"]

        compliance_type = classify_compliance_element(clause_text)
        entities        = extract_entities(clause_text)

        enriched_clause = {
            # carry forward all fields from Step 2
            "clause_id"       : clause["clause_id"],
            "document"        : clause["document"],
            "section"         : clause["section"],
            "clause_text"     : clause_text,
            # new fields added in Step 3
            "compliance_type" : compliance_type,
            "entities"        : entities
        }

        enriched_dataset.append(enriched_clause)

    return enriched_dataset


# ─────────────────────────────────────────
# 5. SUMMARY STATS
# ─────────────────────────────────────────

def print_extraction_summary(enriched_dataset):
    """
    Print count breakdown of compliance types across all clauses.
    """
    from collections import Counter

    type_counts = Counter(c["compliance_type"] for c in enriched_dataset)

    print("\n--- EXTRACTION SUMMARY ---")
    print(f"Total clauses     : {len(enriched_dataset)}")
    print(f"Obligations       : {type_counts.get('obligation', 0)}")
    print(f"Prohibitions      : {type_counts.get('prohibition', 0)}")
    print(f"Penalties         : {type_counts.get('penalty', 0)}")
    print(f"Unclassified      : {type_counts.get('unclassified', 0)}")


if __name__ == "__main__":
    from step1_preprocessing import preprocess_documents
    from step2_segmentation import segment_all_documents

    dataset = preprocess_documents("documents/")
    clause_dataset = segment_all_documents(dataset)
    enriched_dataset = extract_knowledge(clause_dataset)
    print_extraction_summary(enriched_dataset)

    # Preview
    print("\n--- PREVIEW (first 5 enriched clauses) ---")
    for clause in enriched_dataset[:5]:
        print(f"\nClause ID       : {clause['clause_id']}")
        print(f"Document        : {clause['document']}")
        print(f"Section         : {clause['section']}")
        print(f"Compliance Type : {clause['compliance_type']}")
        print(f"Entities        : {clause['entities']}")
        print(f"Text            : {clause['clause_text'][:150]}...")
