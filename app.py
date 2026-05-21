# app.py — AI-Driven Policy & Compliance Intelligence System
# Performance-optimised version: pagination, caching, lazy rendering

import streamlit as st
import os
import pickle
import numpy as np
from collections import Counter

# ─────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────

st.set_page_config(
    page_title="AI Compliance Intelligence System",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────
# IMPORTS FROM PIPELINE MODULES
# ─────────────────────────────────────────

from preprocessing      import preprocess_documents
from segmentation       import segment_document
from knowledge_extraction import extract_knowledge
from embeddings         import generate_embeddings, save_embeddings, load_embeddings
from similarity         import semantic_search, detect_similar_clause_pairs, save_similarity_results
from risk_scoring       import score_risks, save_risk_dataset, load_risk_dataset
from conflict_detection import detect_all_conflicts, save_conflicts, load_conflicts
from summarization_qa   import (
    summarize_all_documents, generate_recommendations, answer_question,
    save_summaries, load_summaries, save_recommendations, load_recommendations,
)
from sentence_transformers import SentenceTransformer

# ─────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────

CLAUSES_PER_PAGE   = 25   # ← was unlimited; this is the #1 lag fix
CONFLICTS_PER_PAGE = 15

# ─────────────────────────────────────────
# CACHED RESOURCES  (loaded once per session)
# ─────────────────────────────────────────

@st.cache_resource(show_spinner="Loading embedding model…")
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


# ─────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────

def run_full_pipeline(uploaded_files, embedding_model):
    os.makedirs("documents",  exist_ok=True)
    os.makedirs("embeddings", exist_ok=True)

    for uploaded_file in uploaded_files:
        with open(os.path.join("documents", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.read())

    steps = [
        "Preprocessing documents…",
        "Segmenting clauses…",
        "Extracting compliance knowledge…",
        "Generating embeddings…",
        "Running similarity detection…",
        "Scoring risks…",
        "Detecting conflicts…",
        "Generating summaries & recommendations…",
    ]
    progress = st.progress(0, text=steps[0])

    def tick(i):
        progress.progress((i + 1) / len(steps), text=steps[i])

    tick(0)
    processed_dataset = preprocess_documents("documents/")

    tick(1)
    clause_dataset = []
    for doc_name, doc_data in processed_dataset.items():
        clause_dataset.extend(segment_document(doc_name, doc_data["cleaned_text"]))

    tick(2)
    enriched_dataset = extract_knowledge(clause_dataset)

    tick(3)
    embeddings, clause_texts, clause_metadata = generate_embeddings(
        enriched_dataset, embedding_model
    )
    save_embeddings(embeddings, clause_texts, clause_metadata)

    tick(4)
    similar_pairs = detect_similar_clause_pairs(
        embeddings, clause_texts, clause_metadata, threshold=0.75, max_pairs=50
    )
    save_similarity_results(similar_pairs)

    tick(5)
    risk_dataset = score_risks(enriched_dataset, similar_pairs)
    save_risk_dataset(risk_dataset)

    tick(6)
    all_conflicts = detect_all_conflicts(similar_pairs, risk_dataset)
    save_conflicts(all_conflicts)

    tick(7)
    summaries       = summarize_all_documents(risk_dataset)
    recommendations = generate_recommendations(risk_dataset, all_conflicts)
    save_summaries(summaries)
    save_recommendations(recommendations)

    progress.empty()
    return (risk_dataset, all_conflicts, summaries, recommendations,
            embeddings, clause_texts, clause_metadata)


def load_existing_results():
    embeddings, clause_texts, clause_metadata = load_embeddings()
    risk_dataset    = load_risk_dataset()
    all_conflicts   = load_conflicts()
    summaries       = load_summaries()
    recommendations = load_recommendations()
    return (risk_dataset, all_conflicts, summaries, recommendations,
            embeddings, clause_texts, clause_metadata)


def results_exist():
    return all(os.path.exists(p) for p in [
        "embeddings/embeddings.npy",
        "embeddings/risk_dataset.pkl",
        "embeddings/conflicts.pkl",
        "embeddings/summaries.pkl",
        "embeddings/recommendations.pkl",
    ])


# ─────────────────────────────────────────
# CACHED FILTER HELPERS
# Using st.cache_data so filters don't re-scan lists on every widget interaction
# ─────────────────────────────────────────

@st.cache_data(show_spinner=False)
def filter_clauses(_risk_dataset, doc_filter, risk_filter, type_filter):
    """Returns filtered slice.  Leading underscore tells Streamlit not to hash the list."""
    result = _risk_dataset
    if doc_filter  != "All": result = [c for c in result if c["document"]        == doc_filter]
    if risk_filter != "All": result = [c for c in result if c["final_risk"]      == risk_filter]
    if type_filter != "All": result = [c for c in result if c["compliance_type"] == type_filter]
    return result


@st.cache_data(show_spinner=False)
def filter_conflicts(_all_conflicts, severity_filter, ctype_filter):
    result = _all_conflicts
    if severity_filter != "All": result = [c for c in result if c["severity"]      == severity_filter]
    if ctype_filter    != "All": result = [c for c in result if c["conflict_type"] == ctype_filter]
    return result


# ─────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────

RISK_ICON = {"high": "🔴", "medium": "🟡", "low": "🟢"}
SEV_ICON  = {"high": "🔴", "medium": "🟡"}

def _paginator(key, total, per_page):
    """Render prev/next buttons and return (start, end) slice indices."""
    total_pages = max(1, (total + per_page - 1) // per_page)
    page_key = f"page_{key}"
    if page_key not in st.session_state:
        st.session_state[page_key] = 0

    col_l, col_m, col_r = st.columns([1, 4, 1])
    with col_l:
        if st.button("← Prev", key=f"prev_{key}", disabled=st.session_state[page_key] == 0):
            st.session_state[page_key] -= 1
    with col_m:
        page = st.session_state[page_key]
        st.markdown(
            f"<div style='text-align:center;padding-top:6px;color:gray;font-size:0.85rem'>"
            f"Page {page + 1} of {total_pages} &nbsp;·&nbsp; {total} items</div>",
            unsafe_allow_html=True,
        )
    with col_r:
        if st.button("Next →", key=f"next_{key}", disabled=st.session_state[page_key] >= total_pages - 1):
            st.session_state[page_key] += 1

    page = st.session_state[page_key]
    start = page * per_page
    end   = min(start + per_page, total)
    return start, end


def _reset_page(key):
    st.session_state[f"page_{key}"] = 0


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

def main():
    # ── CUSTOM CSS ────────────────────────
    st.markdown("""
    <style>
      .block-container { padding-top: 1.5rem; }
      .stExpander > summary { font-size: 0.92rem; }

      /* Sidebar dark theme */
      div[data-testid="stSidebarContent"] { background: #0f172a; color: #e2e8f0; }
      div[data-testid="stSidebarContent"] * { color: #e2e8f0; }
      div[data-testid="stSidebarContent"] .stButton>button {
        background: #2563eb; color: white; border: none; width: 100%;
        border-radius: 6px; padding: 0.5rem;
      }

      /* Stat cards — explicit dark palette, visible on both themes */
      .stat-card {
        background: #1e293b;
        border-radius: 10px;
        padding: 18px 20px 14px;
        border-left: 4px solid #3b82f6;
        min-height: 90px;
      }
      .stat-card.red   { border-left-color: #ef4444; }
      .stat-card.amber { border-left-color: #f59e0b; }
      .stat-card.green { border-left-color: #22c55e; }
      .stat-card.indigo{ border-left-color: #6366f1; }
      .stat-card.teal  { border-left-color: #14b8a6; }
      .stat-card.rose  { border-left-color: #f43f5e; }
      .stat-card.sky   { border-left-color: #0ea5e9; }

      .stat-label {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #94a3b8;
        margin-bottom: 6px;
      }
      .stat-value {
        font-size: 2rem;
        font-weight: 700;
        color: #f1f5f9;
        line-height: 1;
      }
    </style>
    """, unsafe_allow_html=True)

    def stat_card(label, value, colour=""):
        """Render a single stat card as HTML."""
        st.markdown(
            f'<div class="stat-card {colour}">'
            f'  <div class="stat-label">{label}</div>'
            f'  <div class="stat-value">{value}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── HEADER ────────────────────────────
    st.markdown("## 📋 AI-Driven Policy & Compliance Intelligence System")
    st.caption("Upload policy documents to extract compliance insights, assess risks, detect conflicts, and ask natural language questions.")
    st.divider()

    # ── SIDEBAR ───────────────────────────
    with st.sidebar:
        st.markdown("### 📁 Document Upload")
        uploaded_files = st.file_uploader(
            "Upload policy documents (PDF or TXT)",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        process_button = st.button("▶ Process Documents", type="primary")
        st.divider()

        load_button = False
        if results_exist():
            st.success("✅ Processed results found on disk.")
            load_button = st.button("📂 Load Existing Results")

        st.divider()
        st.caption("AI Compliance Intelligence System v2.0")

    # ── SESSION STATE INIT ────────────────
    if "results_loaded" not in st.session_state:
        st.session_state.results_loaded = False

    embedding_model = load_embedding_model()   # cached — no-op after first call

    # ── PROCESS ───────────────────────────
    if process_button:
        if not uploaded_files:
            st.warning("Please upload at least one document before processing.")
        else:
            (
                st.session_state.risk_dataset,
                st.session_state.all_conflicts,
                st.session_state.summaries,
                st.session_state.recommendations,
                st.session_state.embeddings,
                st.session_state.clause_texts,
                st.session_state.clause_metadata,
            ) = run_full_pipeline(uploaded_files, embedding_model)
            st.session_state.results_loaded = True
            # Reset all pagination state when new docs are processed
            for k in list(st.session_state.keys()):
                if k.startswith("page_"):
                    del st.session_state[k]
            st.success("✅ Processing complete.")

    # ── LOAD EXISTING ─────────────────────
    if load_button:
        (
            st.session_state.risk_dataset,
            st.session_state.all_conflicts,
            st.session_state.summaries,
            st.session_state.recommendations,
            st.session_state.embeddings,
            st.session_state.clause_texts,
            st.session_state.clause_metadata,
        ) = load_existing_results()
        st.session_state.results_loaded = True
        st.success("✅ Existing results loaded.")

    # ── TABS ──────────────────────────────
    if not st.session_state.results_loaded:
        st.info("👈 Upload documents using the sidebar and click **Process Documents** to begin.")
        return

    risk_dataset    = st.session_state.risk_dataset
    all_conflicts   = st.session_state.all_conflicts
    summaries       = st.session_state.summaries
    recommendations = st.session_state.recommendations
    embeddings      = st.session_state.embeddings
    clause_texts    = st.session_state.clause_texts
    clause_metadata = st.session_state.clause_metadata

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview",
        "📄 Clauses & Risk",
        "⚠️  Conflicts",
        "📝 Summaries & Recommendations",
        "💬 Q&A",
    ])

    # ══════════════════════════════════════
    # TAB 1 — OVERVIEW
    # ══════════════════════════════════════
    with tab1:
        st.subheader("System Overview")

        risk_counts     = Counter(c["final_risk"]      for c in risk_dataset)
        type_counts     = Counter(c["compliance_type"] for c in risk_dataset)
        conflict_counts = Counter(c["conflict_type"]   for c in all_conflicts)

        c1, c2, c3, c4 = st.columns(4)
        with c1: stat_card("Total Clauses",  len(risk_dataset),             "sky")
        with c2: stat_card("High Risk",      risk_counts.get("high",   0),  "red")
        with c3: stat_card("Medium Risk",    risk_counts.get("medium", 0),  "amber")
        with c4: stat_card("Low Risk",       risk_counts.get("low",    0),  "green")

        st.markdown("<div style='margin-top:14px'></div>", unsafe_allow_html=True)

        c5, c6, c7, c8 = st.columns(4)
        with c5: stat_card("Obligations",  type_counts.get("obligation",  0), "indigo")
        with c6: stat_card("Prohibitions", type_counts.get("prohibition", 0), "rose")
        with c7: stat_card("Penalties",    type_counts.get("penalty",     0), "red")
        with c8: stat_card("Conflicts",    len(all_conflicts),                "amber")

        st.divider()
        st.subheader("Documents Processed")
        doc_names = sorted(set(c["document"] for c in risk_dataset))
        for doc in doc_names:
            doc_clauses   = [c for c in risk_dataset if c["document"] == doc]
            doc_high_risk = sum(1 for c in doc_clauses if c["final_risk"] == "high")
            risk_bar = "🔴" * doc_high_risk + "🟢" * (len(doc_clauses) - doc_high_risk)
            st.markdown(
                f"**{doc}** — {len(doc_clauses)} clauses &nbsp;|&nbsp; "
                f"{doc_high_risk} high-risk"
            )

    # ══════════════════════════════════════
    # TAB 2 — CLAUSES & RISK  (PAGINATED)
    # ══════════════════════════════════════
    with tab2:
        st.subheader("Clause-Level Risk Assessment")

        cf1, cf2, cf3 = st.columns(3)
        with cf1:
            doc_filter = st.selectbox(
                "Filter by Document",
                ["All"] + sorted(set(c["document"] for c in risk_dataset)),
                key="t2_doc",
            )
        with cf2:
            risk_filter = st.selectbox(
                "Filter by Risk Level",
                ["All", "high", "medium", "low"],
                key="t2_risk",
            )
        with cf3:
            type_filter = st.selectbox(
                "Filter by Compliance Type",
                ["All", "obligation", "prohibition", "penalty", "unclassified"],
                key="t2_type",
            )

        # Cached filter — won't re-run on unrelated widget interactions
        filtered = filter_clauses(
            tuple(
                (c["document"], c["final_risk"], c["compliance_type"],
                 c["clause_text"], c["base_risk"], c["risk_explanation"],
                 c["section"],
                 tuple(c["entities"].get("organizations", [])),
                 tuple(c["entities"].get("dates", [])),
                 tuple(c["entities"].get("laws", [])))
                for c in risk_dataset
            ),
            doc_filter, risk_filter, type_filter,
        )
        # filter_clauses receives tuples; reconstruct dicts for display
        # (simpler: just filter the list directly, cache_data won't help with mutable dicts)
        # So we do the filter inline but limit rendering with pagination:
        filtered_clauses = risk_dataset
        if doc_filter  != "All": filtered_clauses = [c for c in filtered_clauses if c["document"]        == doc_filter]
        if risk_filter != "All": filtered_clauses = [c for c in filtered_clauses if c["final_risk"]      == risk_filter]
        if type_filter != "All": filtered_clauses = [c for c in filtered_clauses if c["compliance_type"] == type_filter]

        st.markdown(f"**{len(filtered_clauses)}** clauses match your filters")
        st.divider()

        # PAGINATION — this is the core performance fix
        start, end = _paginator("clauses", len(filtered_clauses), CLAUSES_PER_PAGE)
        st.divider()

        # Only render the current page's worth of expanders
        for clause in filtered_clauses[start:end]:
            label = (
                f"{RISK_ICON.get(clause['final_risk'], '⚪')} "
                f"[{clause['compliance_type'].upper()}] "
                f"{clause['section']} — {clause['document']}"
            )
            with st.expander(label):
                st.markdown("**Clause Text:**")
                st.write(clause["clause_text"])
                col_a, col_b = st.columns(2)
                col_a.markdown(f"**Risk Level:** {RISK_ICON.get(clause['final_risk'], '')} `{clause['final_risk'].upper()}`")
                col_b.markdown(f"**Base Risk:** `{clause['base_risk']}`")
                st.markdown(f"**Explanation:** {clause['risk_explanation']}")
                ents = clause["entities"]
                if any(ents.values()):
                    st.markdown("**Named Entities:**")
                    if ents["organizations"]: st.write(f"Organizations: {', '.join(ents['organizations'])}")
                    if ents["dates"]:         st.write(f"Dates: {', '.join(ents['dates'])}")
                    if ents["laws"]:          st.write(f"Laws: {', '.join(ents['laws'])}")

    # ══════════════════════════════════════
    # TAB 3 — CONFLICTS  (PAGINATED)
    # ══════════════════════════════════════
    with tab3:
        st.subheader("Detected Compliance Conflicts")

        if not all_conflicts:
            st.info("No conflicts detected across the uploaded documents.")
        else:
            sf1, sf2 = st.columns(2)
            with sf1:
                sev_filter = st.selectbox(
                    "Filter by Severity",
                    ["All", "high", "medium"],
                    key="t3_sev",
                )
            with sf2:
                ctype_filter = st.selectbox(
                    "Filter by Conflict Type",
                    ["All", "direct_contradiction", "overlapping_obligation",
                     "penalty_without_obligation"],
                    key="t3_type",
                    format_func=lambda x: x.replace("_", " ").title() if x != "All" else x,
                )

            filtered_conflicts = all_conflicts
            if sev_filter   != "All": filtered_conflicts = [c for c in filtered_conflicts if c["severity"]      == sev_filter]
            if ctype_filter != "All": filtered_conflicts = [c for c in filtered_conflicts if c["conflict_type"] == ctype_filter]

            st.markdown(f"**{len(filtered_conflicts)}** conflicts match your filters")
            st.divider()

            # PAGINATION
            start_c, end_c = _paginator("conflicts", len(filtered_conflicts), CONFLICTS_PER_PAGE)
            st.divider()

            for conflict in filtered_conflicts[start_c:end_c]:
                label = (
                    f"{SEV_ICON.get(conflict['severity'], '⚪')} "
                    f"[{conflict['conflict_type'].replace('_', ' ').upper()}] "
                    f"{conflict['document_a']} — {conflict['clause_a_section']}"
                )
                with st.expander(label):
                    st.markdown(f"**Conflict Type:** `{conflict['conflict_type']}`")
                    st.markdown(
                        f"**Severity:** {SEV_ICON.get(conflict['severity'], '')} "
                        f"`{conflict['severity'].upper()}`"
                    )
                    st.markdown(f"**Explanation:** {conflict['explanation']}")
                    st.markdown("**Clause A:**")
                    st.info(
                        f"[{conflict['document_a']} | {conflict['clause_a_section']}]\n\n"
                        f"{conflict['clause_a_text']}"
                    )
                    if conflict["clause_b_text"]:
                        st.markdown("**Clause B:**")
                        st.warning(
                            f"[{conflict['document_b']} | {conflict['clause_b_section']}]\n\n"
                            f"{conflict['clause_b_text']}"
                        )
                    if conflict["similarity_score"]:
                        st.markdown(f"**Similarity Score:** `{conflict['similarity_score']}`")

    # ══════════════════════════════════════
    # TAB 4 — SUMMARIES & RECOMMENDATIONS
    # ══════════════════════════════════════
    with tab4:
        st.subheader("Document Summaries")
        if summaries:
            for doc_name, summary in summaries.items():
                with st.expander(f"📄 {doc_name}"):
                    st.write(summary)
        else:
            st.info("No summaries available.")

        st.divider()
        st.subheader("Compliance Recommendations")

        for i, rec in enumerate(recommendations, 1):
            if rec.startswith(("REVIEW REQUIRED", "CONFLICT")):
                st.error(f"**{i}.** {rec}")
            elif rec.startswith(("POLICY GAP", "OVERLAP")):
                st.warning(f"**{i}.** {rec}")
            else:
                st.success(f"**{i}.** {rec}")

    # ══════════════════════════════════════
    # TAB 5 — Q&A
    # ══════════════════════════════════════
    with tab5:
        st.subheader("Natural Language Q&A")
        st.markdown(
            "Ask a question about the uploaded policy documents. "
            "The system retrieves the most relevant clauses and extracts a direct answer."
        )

        question   = st.text_input(
            "Your question",
            placeholder="e.g. What are the data retention requirements?",
            label_visibility="collapsed",
        )
        ask_button = st.button("🔍 Ask", type="primary")

        if ask_button and question.strip():
            with st.spinner("Searching policy documents…"):
                result = answer_question(
                    question, embedding_model,
                    embeddings, clause_texts, clause_metadata, top_k=3,
                )

            st.divider()
            st.markdown("### Answer")
            st.success(result["answer"])
            st.markdown(f"**Confidence:** `{result['confidence']:.4f}`")

            if result["confidence"] < 0.3:
                st.warning(
                    "Low confidence — the model could not find a precise answer. "
                    "Review the supporting clauses below for full context."
                )

            st.divider()
            st.markdown("**Supporting Clauses (top 3 by semantic similarity):**")
            for i, clause in enumerate(result["supporting_clauses"], 1):
                with st.expander(
                    f"Clause {i} — {clause['document']} | "
                    f"{clause['section']} (sim: {clause['similarity_score']})"
                ):
                    st.write(clause["clause_text"])
                    st.markdown(f"**Compliance Type:** `{clause['compliance_type']}`")

        elif ask_button:
            st.warning("Please enter a question before clicking Ask.")


if __name__ == "__main__":
    main()