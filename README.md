# AI Detective — AI Knowledge Investigation Assistant

> Evidence-grounded investigation, with natural language. Ask questions about incidents, reports, and structured records, and the assistant retrieves the most relevant evidence, reasons across documents, keeps conversation context, and answers without inventing facts.

This is the **RAG / knowledge-investigation edition** built and tested in a Kaggle notebook. It combines PDF incident reports with Excel/CSV investigation records, converts the knowledge base into multilingual embeddings, indexes it with FAISS, and uses a local Mistral-Nemo instruction model to generate evidence-grounded answers.

**Author:** Raouf Ahmed

**Live Demo:** [AI Detective — Streamlit](https://ai-detective-ai-knowledge-investigation-assistant-axdrc6gzpqws.streamlit.app/)

**GitHub:** [AI Detective — Repository](https://github.com/raoufahmed15/AI-Detective-AI-Knowledge-Investigation-Assistant-full)

---

## Table of contents

1. [What's in the box](#whats-in-the-box)
2. [Quick start](#quick-start)
3. [Project structure](#project-structure)
4. [Data & knowledge base](#data--knowledge-base)
5. [Configuration & models](#configuration--models)
6. [Architecture](#architecture)
7. [RAG pipeline](#rag-pipeline)
8. [Evidence & reasoning rules](#evidence--reasoning-rules)
9. [Evaluation](#evaluation)
10. [Saved artifacts](#saved-artifacts)
11. [Current scope](#current-scope)
12. [Demo](#demo)

---

## What's in the box

**Fully working in the notebook:**
- PDF ingestion using `PyPDF2`
- Excel / CSV ingestion using `pandas`
- Text cleaning and word-based chunking (`chunk_size=120`, `overlap=20`)
- Incident metadata tracking: source, page/row, and `incident_id`
- Multilingual embeddings with **`intfloat/multilingual-e5-base`**
- Vector indexing and similarity search with **FAISS `IndexFlatIP`** after L2 normalization
- Local LLM generation with **`mistralai/Mistral-Nemo-Instruct-2407`** via `AutoModelForCausalLM` + `AutoTokenizer`
- Evidence-grounded RAG answering
- Conversation buffer memory
- Query contextualization for ambiguous follow-up questions
- Cross-document reasoning across incidents
- Source / evidence tracking on every answer
- Explicit hallucination-control instructions when evidence is insufficient
- Optional conversation-memory summarization
- Artifact save + reload testing
- Automated retrieval and answer-fidelity evaluation
- Final test suite covering direct retrieval, PDF evidence, cross-document reasoning, reasoning, and hallucination control

**Example knowledge base run:**
- **10 PDF files**
- **2 structured table files**
- **32 knowledge items** after extraction, cleaning, and chunking
- **32 vectors** in the FAISS index

---

## Quick start

### 0. Prerequisites

The notebook was built for a Kaggle-style environment and uses GPU acceleration when CUDA is available.

Recommended:
- Python 3.12
- Kaggle Notebook with GPU enabled for the local LLM
- A PDF dataset directory
- A CSV / Excel investigation-record directory

### 1. Install dependencies

```bash
pip install -q PyPDF2 sentence-transformers faiss-cpu transformers accelerate torch
```

### 2. Set the dataset paths

The notebook expects these Kaggle input locations:

```python
PDF_DIR = "/kaggle/input/datasets/raouf158/pdf-reports"
SHEET_DIR = "/kaggle/input/datasets/raouf158/crime-recored-ttext"
```

The loader recursively discovers:
- `*.pdf` files under `PDF_DIR`
- Excel / CSV files under `SHEET_DIR`

### 3. Build the knowledge base

The ingestion stage:

1. Extracts PDF text page by page.
2. Converts each structured row into a readable text record.
3. Cleans repeated whitespace.
4. Splits long text into overlapping chunks.
5. Stores metadata alongside every knowledge item.

A successful notebook run produced:

```text
10 PDFs, 2 table file(s) -> 32 knowledge items
```

### 4. Create embeddings and the FAISS index

The notebook uses:

```python
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-base"
```

Queries are encoded as `query: ...` and passages as `passage: ...`, then normalized before FAISS inner-product search.

### 5. Load the LLM

The generation model is:

```python
LLM_MODEL_NAME = "mistralai/Mistral-Nemo-Instruct-2407"
```

The notebook uses `AutoTokenizer`, `AutoModelForCausalLM`, and `model.generate()` through a shared `generate_text()` function.

### 6. Run the RAG pipeline

```python
result = rag_answer(
    "Tell me about INC-001.",
    [],
    index,
    embedding_model,
    knowledge_items,
)
```

Each result contains:

```text
answer
original_question
contextualized_question
sources
```

### 7. Run the evaluation and test suite

The notebook includes both:

```text
Automated evaluation
Final AI Detective test suite
Memory + follow-up test
```

---

## Project structure

```text
AI-Detective-AI-Knowledge-Investigation-Assistant/
├── notebook / Kaggle workflow
│   ├── 01. Install & Imports
│   ├── 02. Extract the Knowledge Base
│   ├── 03. Embeddings + FAISS Index
│   ├── 04. Semantic Search Test
│   ├── 05. LLM Setup
│   ├── 06. Conversation Memory, RAG Prompt & Pipeline
│   ├── 07. Follow-up Questions & Cross-Document Reasoning
│   ├── 08. Optional: Buffer Summary Memory
│   ├── 09. Save & Reload Artifacts
│   ├── 10. Evaluation
│   ├── 11. Summary
│   ├── 12. Final AI Detective Test Suite
│   └── 13. Memory + Follow-up Test
├── artifacts/
│   ├── index.faiss
│   ├── metadata.pkl
│   ├── config.json
│   └── final_test_results.csv
└── README.md
```

The exact runtime layout depends on the Kaggle / deployment environment; the notebook itself is the reference implementation for the current build.

---

## Data & knowledge base

The system works with two kinds of evidence.

### Unstructured evidence — PDF

Each PDF page is extracted, cleaned, and chunked. The metadata stored with every chunk includes:

```text
document
page
source_type = pdf
incident_id
text
```

Incident IDs are inferred from the beginning of the PDF filename, following the notebook's current naming convention.

### Structured evidence — Excel / CSV

Each row becomes an independent knowledge item. The row is converted to text in the form:

```text
column: value
column: value
...
```

Metadata includes:

```text
source_name
row_number
source_type = structured
incident_id
text
```

This lets the retriever search across narrative reports and structured incident records through the same interface.

---

## Configuration & models

### Embedding model

**`intfloat/multilingual-e5-base`**

Used for semantic retrieval over both structured and unstructured evidence.

### Generation model

**`mistralai/Mistral-Nemo-Instruct-2407`**

Used for:
- final RAG answers
- query contextualization
- optional memory summarization

The notebook relies on pretrained models only. **Nothing is trained in this project.**

### Retrieval settings

```text
Chunk size: 120 words
Chunk overlap: 20 words
Top-k retrieval: 5 by default
FAISS index: IndexFlatIP (cosine-style search after L2 normalization)
```

---

## Architecture

```text
                     ┌─────────────────────────────┐
                     │      Investigation Data     │
                     │                             │
                     │  PDF reports + Excel/CSV    │
                     └──────────────┬──────────────┘
                                    │
                                    ▼
                     ┌─────────────────────────────┐
                     │       Ingestion Layer       │
                     │                             │
                     │ extract → clean → chunk     │
                     │ attach incident metadata    │
                     └──────────────┬──────────────┘
                                    │
                                    ▼
                     ┌─────────────────────────────┐
                     │       Embedding Layer       │
                     │                             │
                     │ multilingual-e5-base        │
                     └──────────────┬──────────────┘
                                    │
                                    ▼
                     ┌─────────────────────────────┐
                     │        FAISS Index          │
                     │                             │
                     │ similarity search + top-k   │
                     └──────────────┬──────────────┘
                                    │
                      ┌─────────────▼─────────────┐
                      │   Conversation Context    │
                      │                           │
                      │ memory + query rewrite   │
                      └─────────────┬─────────────┘
                                    │
                                    ▼
                     ┌─────────────────────────────┐
                     │       RAG Prompt            │
                     │                             │
                     │ evidence + history +       │
                     │ investigation rules         │
                     └──────────────┬──────────────┘
                                    │
                                    ▼
                     ┌─────────────────────────────┐
                     │   Mistral-Nemo-Instruct     │
                     │                             │
                     │ evidence-grounded answer   │
                     └──────────────┬──────────────┘
                                    │
                                    ▼
                     ┌─────────────────────────────┐
                     │      Answer + Sources       │
                     │                             │
                     │ answer / rewritten query / │
                     │ source metadata / scores    │
                     └─────────────────────────────┘
```

---

## RAG pipeline

The main `rag_answer(...)` function chains the complete workflow:

1. **Contextualize the question** — follow-ups such as “was it mentioned elsewhere?” are rewritten into standalone questions using the conversation history.
2. **Retrieve evidence** — the contextualized query is embedded and matched against the FAISS index.
3. **Build the RAG prompt** — retrieved evidence is included with source name, page/row, incident ID, and similarity score.
4. **Generate the answer** — the local Mistral-Nemo model answers using the supplied evidence and conversation context.
5. **Update memory** — the new user question and assistant answer are appended to the conversation history.
6. **Return traceable output** — the response keeps both the original and contextualized questions plus the retrieved source metadata.

---

## Evidence & reasoning rules

The RAG prompt contains explicit safeguards designed for investigation-style reasoning:

- **Use only the provided evidence and conversation.**
- **Never invent missing facts.** When the evidence is insufficient, the assistant should say so.
- **A shared detail is a lead, not proof.** For example, seeing a similar vehicle description across incidents does not establish that the same vehicle or person was responsible.
- **Mention relevant incident IDs** when combining evidence from multiple records.

This distinction is central to the project: the assistant is intended to surface evidence and possible relationships without converting weak correlations into confirmed conclusions.

---

## Conversation memory & follow-ups

The notebook supports multi-turn investigation questions.

For example:

```text
User: Tell me about INC-001.
User: What vehicle was mentioned?
User: Was it mentioned in any other incident?
User: Does that prove the incidents are connected?
User: Why not?
```

The query contextualization step turns ambiguous follow-ups into standalone questions before retrieval. The notebook also contains an optional memory-summary function for compressing older conversation history while keeping recent turns.

---

## Evaluation

The notebook evaluates retrieval and generation using real system runs rather than hard-coded metrics.

### Automated evaluation

For the sampled incidents `INC-001`, `INC-003`, and `INC-004`, the notebook reported:

```text
Retrieval accuracy: 100.0%
Answer ID-mention rate: 100.0%
```

These metrics answer two narrow questions:

- Did retrieval surface evidence tagged with the requested incident ID?
- Did the generated answer explicitly mention that incident ID?

They should be interpreted as notebook-level checks on the sampled evaluation set, not as a general performance benchmark.

### Final test suite

The notebook runs **19 test questions** across these categories:

```text
Direct Retrieval
PDF Evidence
Cross-Document
Reasoning
Hallucination Control
```

Examples include:

```text
What happened in INC-001?
Where did INC-003 happen?
What did the CCTV report for INC-004 show?
Which incidents are associated with a black sedan?
Which incidents occurred on River Road and involved equipment theft?
Does the CCTV evidence confirm that INC-004 and INC-001 involve the same vehicle?
Who was driving the black sedan in INC-001?
What is the suspect's phone number in INC-001?
```

The tests are designed to check both retrieval and the assistant's ability to avoid unsupported conclusions.

---

## Saved artifacts

The notebook exports three core artifacts under `artifacts/`:

| File | Purpose |
|---|---|
| `index.faiss` | Persisted FAISS vector index |
| `metadata.pkl` | Stored knowledge items and their source metadata |
| `config.json` | Model names, top-k, chunk settings, index type, item count, and creation timestamp |
| `final_test_results.csv` | Results from the final 19-question test suite |

The notebook also performs a reload test by loading `index.faiss` and `metadata.pkl` back from disk and running a search again.

---

## Current scope

**Implemented in this notebook:**
- Knowledge-base construction
- Semantic retrieval
- Local LLM generation
- RAG answering
- Follow-up contextualization
- Conversation memory
- Cross-document reasoning
- Evidence tracking
- Hallucination-control prompting
- Evaluation and artifact persistence

**Out of scope for this notebook:**
- Training or fine-tuning a custom model
- A production database layer
- The deployment/backend layer itself
- Advanced reranking or a separate graph database

The notebook explicitly treats the saved artifacts as the handoff point that a later **Streamlit or FastAPI backend** can load.

---

## Demo

Try the deployed application:

**[Open the Live Demo](https://ai-detective-ai-knowledge-investigation-assistant-axdrc6gzpqws.streamlit.app/)**

Browse the source code:

**[Open the GitHub Repository](https://github.com/raoufahmed15/AI-Detective-AI-Knowledge-Investigation-Assistant-full)**

---

## Author

**Raouf Ahmed**

AI Detective — AI Knowledge Investigation Assistant
