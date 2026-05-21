# AI-Driven-Policy-Compliance-Intelligence-System

Step 10 — Final Documentation and Project Report
What This Step Does
The module requires two deliverables for this step:

Explainability and usage documentation — explains what each component does, how to use the system, and how AI decisions are made
Final project report — documents the full project from objective to outcomes

Both are produced as structured documents here. No new code is written in this step — this step consolidates everything built across Steps 1–9.

Deliverable 1 — Explainability and Usage Documentation
================================================================
AI-DRIVEN POLICY & COMPLIANCE INTELLIGENCE SYSTEM
Explainability and Usage Documentation
================================================================

----------------------------------------------------------------
1. SYSTEM OVERVIEW
----------------------------------------------------------------

This system processes policy and regulatory documents and
produces structured compliance intelligence using AI and NLP.
It is designed to support compliance teams in reviewing large
volumes of policy documents faster and more consistently than
manual review.

----------------------------------------------------------------
2. HOW TO USE THE SYSTEM
----------------------------------------------------------------

REQUIREMENTS
	- Python 3.9 or above
	- All dependencies installed (see Section 3)
	- At least one policy document in PDF or TXT format

INSTALLATION
	Run the following in your terminal:

		pip install pdfplumber spacy nltk sentence-transformers
		pip install transformers torch scikit-learn streamlit
		python -m spacy download en_core_web_sm

RUNNING THE APPLICATION
	From the project root directory:

		streamlit run app.py

	This opens the application in your browser at:
		http://localhost:8501

STEP-BY-STEP USAGE
	1. Open the application in your browser
	2. Use the sidebar to upload one or more PDF or TXT files
	3. Click "Process Documents"
		 - The system runs all 8 processing steps automatically
		 - Progress is shown via on-screen spinners
	4. Once complete, results appear across five tabs:
		 - Overview        : summary metrics across all documents
		 - Clauses & Risk  : clause-level risk with filters
		 - Conflicts       : detected contradictions and overlaps
		 - Summaries & Recs: AI summaries and recommendations
		 - Q&A             : ask questions in plain English
	5. On subsequent visits, click "Load Existing Results"
		 to reload without reprocessing

----------------------------------------------------------------
3. COMPONENT EXPLAINABILITY
----------------------------------------------------------------

Each AI decision made by the system is explained below.

STEP 1 — DOCUMENT INGESTION & PREPROCESSING
	What it does  : Reads PDFs and text files, removes noise,
									normalizes whitespace, and tokenizes content.
	Tool used     : pdfplumber (extraction), NLTK (tokenization)
	Why it matters: Raw PDF text contains headers, footers, and
									encoding artifacts. Cleaning ensures downstream
									NLP operates on clean input.

STEP 2 — CLAUSE SEGMENTATION
	What it does  : Splits cleaned text into clause-level units
									using section headers and list item patterns.
	Tool used     : Python regex, NLTK sent_tokenize
	Why it matters: Embeddings and classification work best on
									focused, clause-level text rather than full
									document text.

STEP 3 — KNOWLEDGE EXTRACTION
	What it does  : Labels each clause as obligation, prohibition,
									penalty, or unclassified using keyword matching.
									Extracts organizations, dates, and laws using
									spaCy NER.
	Tool used     : spaCy (NER), keyword pattern matching
	How decisions are made:
		- Prohibition keywords are checked before obligation keywords
			to prevent "shall not" being misclassified as obligation.
		- NER uses spaCy's en_core_web_sm model trained on general
			English text — LAW entity coverage may be limited on
			domain-specific documents.

STEP 4 — EMBEDDING GENERATION
	What it does  : Converts each clause into a 384-dimensional
									numerical vector representing its meaning.
	Model used    : all-MiniLM-L6-v2 (Sentence Transformers)
	How it works  : The model encodes semantic meaning — clauses
									with similar meaning produce similar vectors
									regardless of exact wording.

STEP 5 — SIMILARITY AND SEMANTIC SEARCH
	What it does  : Finds semantically similar clause pairs and
									retrieves relevant clauses for user queries.
	Tool used     : cosine_similarity from scikit-learn
	How scores work:
		- Score of 1.0 means identical meaning
		- Score >= 0.75 flags clauses as similar
		- Score >= 0.85 triggers elevated risk in Step 6
	Limitation    : Cosine similarity measures semantic closeness,
									not legal equivalence. Domain review is still
									required for final compliance decisions.

STEP 6 — RISK SCORING
	What it does  : Assigns low, medium, or high risk to each
									clause using two layers of reasoning.
	Layer 1 — Rule-based:
		- Keyword presence (e.g., "criminal", "breach") → high risk
		- Compliance type (penalty → high, prohibition → medium,
			obligation → medium, unclassified → low)
	Layer 2 — AI-assisted:
		- If a clause has high semantic overlap with another clause
			(similarity >= 0.75), its risk is escalated upward.
		- This reflects that overlapping clauses may impose
			conflicting or compounded compliance requirements.
	Explainability: Every clause carries a risk_explanation field
									describing exactly why its risk level was set.

STEP 7 — CONFLICT DETECTION
	What it does  : Identifies three types of logical conflict
									between clauses.
	Conflict types:
		Direct Contradiction  : obligation paired with prohibition
														on the same topic (similarity >= 0.75)
		Overlapping Obligation: two obligations on same topic from
														different sources (similarity >= 0.80)
		Penalty Without Rule  : penalty clause exists in a section
														with no corresponding obligation
	Limitation    : Conflict detection is based on compliance type
									labels and similarity scores. It does not
									perform full legal reasoning.

STEP 8 — SUMMARIZATION, RECOMMENDATIONS AND Q&A
	Summarization:
		Model used  : facebook/bart-large-cnn
		How it works: Document clauses are joined and chunked to
									fit BART's token limit, then summarized.
									Multiple chunk summaries are merged into one
									final document summary.
	Recommendations:
		How they are generated: Rule-based logic over risk_dataset
									and conflict list. Each recommendation maps
									directly to a detected risk or conflict.
	Q&A:
		Step 1      : Semantic retrieval using all-MiniLM-L6-v2
									finds the top 3 most relevant clauses for
									the question.
		Step 2      : deepset/roberta-base-squad2 extracts a
									precise answer span from the retrieved context.
		Confidence  : The model returns a confidence score.
									Scores below 0.3 are flagged as unreliable.
		Limitation  : The QA model is extractive — it can only
									return text present in the document. It cannot
									infer or reason beyond what is written.

----------------------------------------------------------------
4. KNOWN LIMITATIONS
----------------------------------------------------------------

	- spaCy's en_core_web_sm has limited coverage of legal LAW
		entities. A domain-specific model would improve NER accuracy.

	- Clause segmentation relies on regex patterns. Documents
		without clear headers or numbering may produce fewer
		well-defined clauses.

	- BART summarization requires chunking for long documents.
		Very short documents (under 30 words) are skipped.

	- All compliance decisions produced by this system are
		AI-assisted and must be reviewed by a qualified compliance
		professional before acting on them.

----------------------------------------------------------------
5. FILE AND FOLDER STRUCTURE
----------------------------------------------------------------

	project/
	├── documents/              ← uploaded policy documents
	├── embeddings/             ← all saved processing outputs
	│   ├── embeddings.npy
	│   ├── clause_texts.pkl
	│   ├── clause_metadata.pkl
	│   ├── similar_pairs.pkl
	│   ├── risk_dataset.pkl
	│   ├── conflicts.pkl
	│   ├── summaries.pkl
	│   └── recommendations.pkl
	├── step1_preprocessing.py
	├── step2_segmentation.py
	├── step3_knowledge_extraction.py
	├── step4_embeddings.py
	├── step5_similarity.py
	├── step6_risk_scoring.py
	├── step7_conflict_detection.py
	├── step8_summarization_qa.py
	└── app.py

================================================================

Deliverable 2 — Final Project Report
================================================================
AI-DRIVEN POLICY & COMPLIANCE INTELLIGENCE SYSTEM
Final Project Report
================================================================

----------------------------------------------------------------
1. PROJECT OBJECTIVE
----------------------------------------------------------------

To develop an end-to-end AI-driven compliance intelligence
system that automatically processes policy and regulatory
documents, transforms unstructured text into structured
compliance knowledge, applies AI-based reasoning to identify
obligations, violations, overlaps, and risk levels, and
delivers explainable insights, summaries, and natural language
Q&A through an interactive interface.

----------------------------------------------------------------
2. BACKGROUND AND CONTEXT
----------------------------------------------------------------

Organisations across banking, healthcare, insurance, and
technology must comply with a growing set of internal policies
and external regulations. These documents are lengthy, complex,
and frequently updated, making manual review slow and prone to
error. This project builds an AI-powered solution that
automates compliance analysis, reduces operational overhead,
and improves governance by applying NLP and semantic
embeddings to policy documents.

----------------------------------------------------------------
3. TECHNICAL APPROACH
----------------------------------------------------------------

The system was developed across four phases aligned to the
module requirements:

PHASE 1 — Document Ingestion and Preprocessing (Steps 1–2)
	Documents were ingested using pdfplumber for PDF extraction
	and standard file I/O for text files. Text was cleaned by
	removing URLs, page numbers, and encoding artifacts, then
	normalized and tokenized using NLTK. Cleaned text was
	segmented into clause-level units using regex patterns that
	detect section headers, article numbers, and list markers,
	with sentence tokenization as fallback.

PHASE 2 — Semantic Understanding and Knowledge Extraction
(Steps 3–4)
	Each clause was classified as an obligation, prohibition,
	penalty, or unclassified using keyword matching. Named
	entities including organizations, dates, and laws were
	extracted using spaCy's en_core_web_sm model. Semantic
	embeddings were generated for each clause using the
	all-MiniLM-L6-v2 Sentence Transformer model, producing
	384-dimensional vectors stored as numpy arrays for
	downstream use.

PHASE 3 — AI Reasoning and Risk Assessment (Steps 5–7)
	Cosine similarity via scikit-learn was used for semantic
	search and similar clause pair detection. Risk levels were
	assigned using two layers: a rule-based layer using keyword
	and compliance type patterns, and an AI-assisted layer that
	escalates risk based on detected semantic overlap scores.
	Every risk assignment produces an explanation string.
	Conflict detection identified direct contradictions between
	obligation and prohibition clauses, overlapping obligations
	across documents, and sections where penalties exist without
	corresponding obligation clauses.

PHASE 4 — Interface, Explainability and Delivery (Steps 8–9)
	AI-based document summaries were generated using
	facebook/bart-large-cnn with chunking to handle token limits.
	Compliance recommendations were generated using rule-based
	logic over the risk dataset and conflict list. Natural
	language Q&A was implemented using semantic retrieval
	followed by extractive answer generation using
	deepset/roberta-base-squad2. The full system was deployed
	as an interactive Streamlit application with five tabs
	covering overview metrics, clause-level risk, conflict
	detection, summaries and recommendations, and Q&A.

----------------------------------------------------------------
4. TOOLS AND TECHNOLOGIES USED
----------------------------------------------------------------

	Category              Tool / Library
	──────────────────────────────────────────────────────
	Programming Language  Python 3.9+
	PDF Extraction        pdfplumber
	NLP                   spaCy, NLTK
	Embeddings            Sentence Transformers
												(all-MiniLM-L6-v2)
	ML / Similarity       scikit-learn (cosine_similarity)
	Summarization         Hugging Face Transformers
												(facebook/bart-large-cnn)
	Q&A                   Hugging Face Transformers
												(deepset/roberta-base-squad2)
	Interface             Streamlit
	Storage               NumPy (.npy), Pickle (.pkl)

----------------------------------------------------------------
5. DELIVERABLES PRODUCED
----------------------------------------------------------------

	✅ Preprocessed and structured policy document dataset
	✅ Clause-level semantic embeddings
	✅ Extracted compliance rule dataset
		 (obligations, prohibitions, penalties, NER)
	✅ AI-based compliance risk assessment engine
		 (rule-based + AI-assisted, with explainability)
	✅ Conflict detection engine
		 (contradictions, overlaps, penalty gaps)
	✅ Explainable compliance insights and summaries
	✅ Compliance recommendations
	✅ Interactive AI compliance assistant (Streamlit app)
	✅ Explainability and usage documentation
	✅ Final project report

----------------------------------------------------------------
6. PROJECT OUTCOMES
----------------------------------------------------------------

	1. The system is capable of interpreting complex policy and
		 regulatory text at clause level, classifying compliance
		 elements, and extracting named entities automatically.

	2. AI-driven identification and explanation of compliance
		 risks is achieved through the two-layer risk scoring
		 engine, with every decision accompanied by a human-
		 readable explanation.

	3. A deployable compliance intelligence solution is
		 delivered as a Streamlit application, enabling faster
		 document review and supporting governance decisions
		 through summaries, conflict alerts, recommendations,
		 and natural language Q&A.

----------------------------------------------------------------
7. LIMITATIONS AND FUTURE SCOPE
----------------------------------------------------------------

	Limitations:
	- NER accuracy is limited by the general-purpose spaCy model
	- Clause segmentation depends on document formatting quality
	- All outputs require review by a compliance professional

	Future Scope (beyond current module requirements):
	- Integration with live regulatory update feeds
	- Support for multilingual policy documents
	- Fine-tuned domain-specific NER and classification models

================================================================
END OF REPORT
================================================================

Common Mistakes
MistakeFixWriting documentation after submissionWrite documentation alongside each step — this report is only possible because every step produced structured, traceable outputsListing tools not used in the projectEvery tool listed in Section 4 maps directly to a step where it was usedOmitting limitationsThe module requires explainability — that includes being honest about what the system cannot doTreating the report as a separate document from the codeThe report directly references step outputs — keep them consistent

All 10 Steps Complete
StepDeliverable1Preprocessing notebook + cleaned dataset2Clause segmentation notebook + clause dataset3Knowledge extraction notebook + enriched dataset4Embedding generation notebook + stored embeddings5Similarity + search notebook + similar pairs6Risk scoring notebook + risk dataset7Conflict detection notebook + conflict list8Summarization + Q&A notebook + saved outputs9Streamlit application10Explainability documentation + final project report