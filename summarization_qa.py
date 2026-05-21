import pickle
import numpy as np
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity

_summarizer = None
_qa_model = None
_summarizer_error = None
_qa_model_error = None


def _extractive_summary(text, max_sentences=4):
    """Fallback summary when transformer summarization pipeline is unavailable."""
    parts = [s.strip() for s in text.split(".") if len(s.strip().split()) >= 8]
    if not parts:
        return text[:600] if text else "Document too short to summarize."
    summary = ". ".join(parts[:max_sentences]).strip()
    if summary and not summary.endswith("."):
        summary += "."
    return summary


def get_summarizer():
    """Lazily initialize summarization model."""
    global _summarizer, _summarizer_error
    if _summarizer is None:
        try:
            _summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        except Exception as exc:
            _summarizer_error = str(exc)
            _summarizer = False
    return _summarizer


def get_qa_model():
    """Lazily initialize Q&A model."""
    global _qa_model, _qa_model_error
    if _qa_model is None:
        try:
            _qa_model = pipeline("question-answering", model="deepset/roberta-base-squad2")
        except Exception as exc:
            _qa_model_error = str(exc)
            _qa_model = False
    return _qa_model


# ─────────────────────────────────────────
# 2. SUMMARIZATION
# ─────────────────────────────────────────

# BART has a token limit — chunk large documents before summarizing
CHUNK_SIZE      = 800   # words per chunk
MAX_SUMMARY_LEN = 150   # tokens
MIN_SUMMARY_LEN = 50    # tokens


def chunk_text(text, chunk_size=CHUNK_SIZE):
    """
    Split text into chunks of approximately chunk_size words.
    BART cannot process very long inputs in one pass.
    """
    words  = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i: i + chunk_size])
        chunks.append(chunk)

    return chunks


def summarize_document(document_name, risk_dataset):
    """
    Summarize all clauses belonging to a document.
    Steps:
    1. Collect all clause texts for the document
    2. Join into full document text
    3. Chunk to stay within BART token limit
    4. Summarize each chunk
    5. Combine chunk summaries into final summary

    Returns summary string.
    """
    # Collect clauses for this document
    doc_clauses = [
        c["clause_text"]
        for c in risk_dataset
        if c["document"] == document_name
    ]

    if not doc_clauses:
        return f"No clauses found for document: {document_name}"

    full_text = " ".join(doc_clauses)
    chunks    = chunk_text(full_text)

    chunk_summaries = []

    for i, chunk in enumerate(chunks):
        # Skip very short chunks — not worth summarizing
        if len(chunk.split()) < 30:
            continue

        summarizer = get_summarizer()
        if summarizer is False:
            chunk_summaries.append(_extractive_summary(chunk))
        else:
            result = summarizer(
                chunk,
                max_length=MAX_SUMMARY_LEN,
                min_length=MIN_SUMMARY_LEN,
                do_sample=False
            )
            chunk_summaries.append(result[0]["summary_text"])

    if not chunk_summaries:
        return "Document too short to summarize."

    # If multiple chunks, summarize the combined chunk summaries
    if len(chunk_summaries) > 1:
        combined = " ".join(chunk_summaries)
        combined_chunks = chunk_text(combined)

        final_summaries = []
        for chunk in combined_chunks:
            if len(chunk.split()) < 30:
                continue
            summarizer = get_summarizer()
            if summarizer is False:
                final_summaries.append(_extractive_summary(chunk))
            else:
                result = summarizer(
                    chunk,
                    max_length=MAX_SUMMARY_LEN,
                    min_length=MIN_SUMMARY_LEN,
                    do_sample=False
                )
                final_summaries.append(result[0]["summary_text"])

        return " ".join(final_summaries)

    return chunk_summaries[0]


def summarize_all_documents(risk_dataset):
    """
    Generate summaries for all documents in the risk dataset.
    Returns dict: {document_name: summary_text}
    """
    document_names = list(set(c["document"] for c in risk_dataset))
    summaries      = {}

    for doc_name in document_names:
        print(f"\nSummarizing: {doc_name}")
        summary = summarize_document(doc_name, risk_dataset)
        summaries[doc_name] = summary
        print(f"  → Done ({len(summary.split())} words)")

    return summaries


# ─────────────────────────────────────────
# 3. RECOMMENDATIONS
# ─────────────────────────────────────────

def generate_recommendations(risk_dataset, all_conflicts):
    """
    Generate actionable compliance recommendations based on:
    - High risk clauses from Step 6
    - Detected conflicts from Step 7

    Returns list of recommendation strings.
    """
    recommendations = []

    # --- Recommendations from high risk clauses ---
    high_risk_clauses = [
        c for c in risk_dataset if c["final_risk"] == "high"
    ]

    if high_risk_clauses:
        recommendations.append(
            f"REVIEW REQUIRED: {len(high_risk_clauses)} high-risk clause(s) detected. "
            f"These involve penalties, prohibitions, or semantic overlaps that require "
            f"immediate compliance review."
        )

        # Group by document
        from collections import defaultdict
        doc_high_risk = defaultdict(list)
        for c in high_risk_clauses:
            doc_high_risk[c["document"]].append(c)

        for doc, clauses in doc_high_risk.items():
            recommendations.append(
                f"Document '{doc}' contains {len(clauses)} high-risk clause(s) — "
                f"prioritize review of sections: "
                f"{', '.join(set(c['section'] for c in clauses))}"
            )

    # --- Recommendations from direct contradictions ---
    contradictions = [
        c for c in all_conflicts
        if c["conflict_type"] == "direct_contradiction"
    ]

    if contradictions:
        recommendations.append(
            f"CONFLICT RESOLUTION REQUIRED: {len(contradictions)} direct contradiction(s) "
            f"detected between obligation and prohibition clauses. "
            f"These must be resolved to avoid compliance violations."
        )

        for conflict in contradictions:
            recommendations.append(
                f"Resolve conflict between '{conflict['clause_a_section']}' "
                f"({conflict['document_a']}) and '{conflict['clause_b_section']}' "
                f"({conflict['document_b']}): {conflict['explanation']}"
            )

    # --- Recommendations from overlapping obligations ---
    overlaps = [
        c for c in all_conflicts
        if c["conflict_type"] == "overlapping_obligation"
    ]

    if overlaps:
        recommendations.append(
            f"OVERLAP REVIEW: {len(overlaps)} overlapping obligation(s) detected across "
            f"documents. Review for redundancy or conflicting requirements."
        )

    # --- Recommendations from penalty without obligation ---
    penalty_gaps = [
        c for c in all_conflicts
        if c["conflict_type"] == "penalty_without_obligation"
    ]

    if penalty_gaps:
        recommendations.append(
            f"POLICY GAP: {len(penalty_gaps)} section(s) define penalties without a "
            f"corresponding obligation. Add clear obligation clauses to these sections."
        )

    if not recommendations:
        recommendations.append(
            "No critical compliance issues detected. "
            "Routine review of medium and low risk clauses is advised."
        )

    return recommendations


# ─────────────────────────────────────────
# 4. Q&A
# ─────────────────────────────────────────

def retrieve_relevant_clauses(question, embedding_model, embeddings,
                               clause_texts, clause_metadata, top_k=3):
    """
    Step 1 of Q&A:
    Use semantic search to retrieve top_k most relevant
    clauses for the question using cosine similarity.

    Returns list of (clause_text, metadata) tuples.
    """
    question_embedding = embedding_model.encode(
        [question], convert_to_numpy=True
    )

    scores      = cosine_similarity(question_embedding, embeddings)[0]
    top_indices = np.argsort(scores)[::-1][:top_k]

    retrieved = []
    for idx in top_indices:
        retrieved.append({
            "clause_text"     : clause_texts[idx],
            "clause_id"       : clause_metadata[idx]["clause_id"],
            "document"        : clause_metadata[idx]["document"],
            "section"         : clause_metadata[idx]["section"],
            "compliance_type" : clause_metadata[idx]["compliance_type"],
            "similarity_score": round(float(scores[idx]), 4)
        })

    return retrieved


def answer_question(question, embedding_model, embeddings,
                    clause_texts, clause_metadata, top_k=3):
    """
    Full Q&A pipeline:
    1. Retrieve top_k relevant clauses via semantic search
    2. Combine retrieved clauses into a context passage
    3. Run deepset/roberta-base-squad2 extractive Q&A over context
    4. Return answer with source clause info

    Returns dict with answer and supporting clauses.
    """
    # Step 1: Retrieve relevant clauses
    retrieved_clauses = retrieve_relevant_clauses(
        question, embedding_model, embeddings,
        clause_texts, clause_metadata, top_k=top_k
    )

    if not retrieved_clauses:
        return {
            "question"          : question,
            "answer"            : "No relevant clauses found.",
            "confidence"        : 0.0,
            "supporting_clauses": []
        }

    # Step 2: Build context from retrieved clauses
    context = " ".join([c["clause_text"] for c in retrieved_clauses])

    # Step 3: Run extractive Q&A
    qa_input = {
        "question": question,
        "context" : context
    }

    qa_model = get_qa_model()
    if qa_model is False:
        best = retrieved_clauses[0]
        fallback_answer = best["clause_text"][:300].strip()
        if fallback_answer and not fallback_answer.endswith("."):
            fallback_answer += "..."
        return {
            "question"          : question,
            "answer"            : fallback_answer or "No answer available.",
            "confidence"        : round(float(best.get("similarity_score", 0.0)), 4),
            "supporting_clauses": retrieved_clauses
        }

    result = qa_model(qa_input)

    return {
        "question"          : question,
        "answer"            : result["answer"],
        "confidence"        : round(result["score"], 4),
        "supporting_clauses": retrieved_clauses
    }


# ─────────────────────────────────────────
# 5. SAVE OUTPUTS
# ─────────────────────────────────────────

def save_summaries(summaries, save_path="embeddings/summaries.pkl"):
    with open(save_path, "wb") as f:
        pickle.dump(summaries, f)
    print(f"\nSummaries saved to '{save_path}'")


def load_summaries(save_path="embeddings/summaries.pkl"):
    with open(save_path, "rb") as f:
        summaries = pickle.load(f)
    return summaries


def save_recommendations(recommendations, save_path="embeddings/recommendations.pkl"):
    with open(save_path, "wb") as f:
        pickle.dump(recommendations, f)
    print(f"Recommendations saved to '{save_path}'")


def load_recommendations(save_path="embeddings/recommendations.pkl"):
    with open(save_path, "rb") as f:
        recommendations = pickle.load(f)
    return recommendations


if __name__ == "__main__":
    from embeddings import load_embeddings, load_embedding_model
    from risk_scoring import load_risk_dataset
    from conflict_detection import load_conflicts

    embeddings, clause_texts, clause_metadata = load_embeddings("embeddings/")
    risk_dataset = load_risk_dataset("embeddings/risk_dataset.pkl")
    all_conflicts = load_conflicts("embeddings/conflicts.pkl")
    embedding_model = load_embedding_model()

    summaries = summarize_all_documents(risk_dataset)
    save_summaries(summaries)

    recommendations = generate_recommendations(risk_dataset, all_conflicts)
    save_recommendations(recommendations)
