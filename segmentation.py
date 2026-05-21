import re

# ─────────────────────────────────────────
# 1. PATTERN DEFINITIONS
# ─────────────────────────────────────────

# Matches: "Article 5", "Section 3.1", "SECTION 2", "1.", "1.1", "Chapter 4"
SECTION_PATTERN = re.compile(
    r'((?:Article|Section|Chapter|ARTICLE|SECTION|CHAPTER)\s+\d+[\.\d]*'
    r'|\b\d+\.\d*\s+[A-Z]'   # e.g., "3.1 Data Retention"
    r'|\b\d+\.\s+[A-Z])',     # e.g., "1. Purpose"
    re.IGNORECASE
)

# Matches list items: "(a)", "a)", "i.", "(i)", "(1)"
LIST_ITEM_PATTERN = re.compile(
    r'(\([a-z]\)|\([ivxlcdm]+\)|[a-z]\)|\([0-9]+\))',
    re.IGNORECASE
)


# ─────────────────────────────────────────
# 2. SECTION-LEVEL SPLIT
# ─────────────────────────────────────────

def split_into_sections(cleaned_text):
    """
    Split document into sections based on article/section headers.
    Returns list of (section_header, section_body) tuples.
    """
    parts = SECTION_PATTERN.split(cleaned_text)

    sections = []

    # parts alternates: [pre-header text, header, body, header, body ...]
    if len(parts) == 1:
        # No headers found — treat entire text as one section
        sections.append(("General", parts[0].strip()))
        return sections

    # First chunk before any header
    if parts[0].strip():
        sections.append(("Preamble", parts[0].strip()))

    # Pair headers with their bodies
    for i in range(1, len(parts) - 1, 2):
        header = parts[i].strip()
        body = parts[i + 1].strip() if i + 1 < len(parts) else ""
        if body:
            sections.append((header, body))

    return sections


# ─────────────────────────────────────────
# 3. CLAUSE-LEVEL SPLIT WITHIN SECTION
# ─────────────────────────────────────────

def split_into_clauses(section_body):
    """
    Split a section body into individual clauses.
    Priority:
      1. List item markers (a), b), i. etc.)
      2. Sentence boundaries (fallback)
    Returns list of clause strings.
    """
    clauses = []

    # Try list-item split first
    list_parts = LIST_ITEM_PATTERN.split(section_body)

    if len(list_parts) > 1:
        # Merge markers with their content
        i = 0
        while i < len(list_parts):
            part = list_parts[i].strip()
            if LIST_ITEM_PATTERN.fullmatch(part):
                # This is a marker — merge with next chunk
                if i + 1 < len(list_parts):
                    clause = part + " " + list_parts[i + 1].strip()
                    if clause.strip():
                        clauses.append(clause.strip())
                    i += 2
                else:
                    i += 1
            else:
                # Regular text before first list item
                if part:
                    # Fall back to sentence split for this part
                    from nltk.tokenize import sent_tokenize
                    for sent in sent_tokenize(part):
                        if len(sent.strip()) > 20:  # filter noise
                            clauses.append(sent.strip())
                i += 1
    else:
        # No list items — use sentence tokenization
        from nltk.tokenize import sent_tokenize
        for sent in sent_tokenize(section_body):
            if len(sent.strip()) > 20:
                clauses.append(sent.strip())

    return clauses


# ─────────────────────────────────────────
# 4. FULL SEGMENTATION PIPELINE
# ─────────────────────────────────────────

def segment_document(doc_name, cleaned_text):
    """
    Full pipeline for one document:
    cleaned_text → sections → clauses
    Returns list of clause dicts with metadata.
    """
    sections = split_into_sections(cleaned_text)
    all_clauses = []
    clause_id = 0

    for section_header, section_body in sections:
        clauses = split_into_clauses(section_body)

        for clause_text in clauses:
            all_clauses.append({
                "clause_id"      : clause_id,
                "document"       : doc_name,
                "section"        : section_header,
                "clause_text"    : clause_text
            })
            clause_id += 1

    return all_clauses


def segment_all_documents(processed_dataset):
    """
    Run segmentation over all documents in the dataset.
    Returns combined clause list across all documents.
    """
    all_clauses = []

    for doc_name, doc_data in processed_dataset.items():
        print(f"\nSegmenting: {doc_name}")
        clauses = segment_document(doc_name, doc_data["cleaned_text"])
        all_clauses.extend(clauses)
        print(f"  → Sections found : {len(split_into_sections(doc_data['cleaned_text']))}")
        print(f"  → Clauses found  : {len(clauses)}")

    return all_clauses


if __name__ == "__main__":
    from step1_preprocessing import preprocess_documents

    dataset = preprocess_documents("documents/")
    clause_dataset = segment_all_documents(dataset)

    # Preview
    print("\n--- PREVIEW (first 5 clauses) ---")
    for clause in clause_dataset[:5]:
        print(f"\nClause ID : {clause['clause_id']}")
        print(f"Document  : {clause['document']}")
        print(f"Section   : {clause['section']}")
        print(f"Text      : {clause['clause_text'][:150]}...")