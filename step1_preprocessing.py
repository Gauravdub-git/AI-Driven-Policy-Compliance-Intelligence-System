import pdfplumber
import os
import re
import nltk
import spacy

nltk.download('punkt')
nltk.download('stopwords')

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

nlp = spacy.load("en_core_web_sm")

# ─────────────────────────────────────────
# 1. TEXT EXTRACTION
# ─────────────────────────────────────────

def extract_text_from_pdf(pdf_path):
    """Extract raw text from a PDF file using pdfplumber."""
    full_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                full_text += page_text + "\n"
    return full_text


def extract_text_from_txt(txt_path):
    """Extract raw text from a plain text file."""
    with open(txt_path, "r", encoding="utf-8") as f:
        return f.read()


def load_documents(folder_path):
    """
    Load all PDFs and text files from a folder.
    Returns a dict: {filename: raw_text}
    """
    documents = {}
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        if filename.endswith(".pdf"):
            documents[filename] = extract_text_from_pdf(filepath)
        elif filename.endswith(".txt"):
            documents[filename] = extract_text_from_txt(filepath)
        else:
            print(f"Skipped (unsupported format): {filename}")
    return documents


# ─────────────────────────────────────────
# 2. TEXT CLEANING & NORMALIZATION
# ─────────────────────────────────────────

def clean_text(text):
    """
    Clean and normalize raw extracted text.
    - Remove extra whitespace and newlines
    - Remove special characters (keep punctuation for sentence boundaries)
    - Lowercase
    - Normalize unicode
    """
    # Remove headers/footers patterns (page numbers, URLs)
    text = re.sub(r'http\S+', '', text)               # remove URLs
    text = re.sub(r'\bPage\s+\d+\b', '', text)        # remove "Page 1" etc.

    # Normalize whitespace
    text = re.sub(r'\n+', ' ', text)                  # newlines to space
    text = re.sub(r'\s+', ' ', text)                  # multiple spaces to one

    # Remove non-ASCII characters
    text = text.encode('ascii', 'ignore').decode('ascii')

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


# ─────────────────────────────────────────
# 3. TOKENIZATION
# ─────────────────────────────────────────

def tokenize_text(text):
    """
    Tokenize cleaned text into:
    - sentences
    - words (with stopword removal for analysis use)
    """
    sentences = sent_tokenize(text)

    words = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    filtered_words = [w for w in words if w.isalpha() and w not in stop_words]

    return {
        "sentences": sentences,
        "words": filtered_words
    }


# ─────────────────────────────────────────
# 4. FULL PREPROCESSING PIPELINE
# ─────────────────────────────────────────

def preprocess_documents(folder_path):
    """
    Full pipeline:
    Load → Extract → Clean → Tokenize
    Returns structured dataset.
    """
    raw_documents = load_documents(folder_path)
    processed_dataset = {}

    for doc_name, raw_text in raw_documents.items():
        print(f"\nProcessing: {doc_name}")

        cleaned = clean_text(raw_text)
        tokens = tokenize_text(cleaned)

        processed_dataset[doc_name] = {
            "raw_text": raw_text,
            "cleaned_text": cleaned,
            "sentences": tokens["sentences"],
            "words": tokens["words"]
        }

        print(f"  → Sentences extracted : {len(tokens['sentences'])}")
        print(f"  → Words (filtered)    : {len(tokens['words'])}")

    return processed_dataset


# ─────────────────────────────────────────
# 5. RUN
# ─────────────────────────────────────────

DOCUMENTS_FOLDER = "documents/"
dataset = preprocess_documents(DOCUMENTS_FOLDER)

# Preview output for first document
first_doc = list(dataset.keys())[0]
print("\n--- PREVIEW ---")
print(f"Document     : {first_doc}")
print(f"Cleaned text : {dataset[first_doc]['cleaned_text'][:300]}...")
print(f"First 3 sentences:")
for s in dataset[first_doc]['sentences'][:3]:
    print(f"  - {s}")
