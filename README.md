AI Detective — AI Knowledge Investigation Assistant

Author: Raouf Ahmed
Project Type: RAG-based AI Knowledge & Investigation Assistant

AI Detective is an evidence-grounded investigation assistant built around a Retrieval-Augmented Generation (RAG) pipeline. The system works over a knowledge base composed of PDF investigation reports and structured Excel/CSV records, retrieves the most relevant evidence for a user question, and uses a local instruction-tuned language model to generate an answer based only on the retrieved evidence and conversation context.

A central design principle is that a shared detail between incidents is treated as a possible investigative lead, not as proof that the incidents are connected. The prompt explicitly instructs the model to avoid inventing facts and to state when the available evidence is insufficient.

What the Project Does

The notebook builds the core knowledge and reasoning pipeline for an AI investigation assistant that can:

Ingest unstructured PDF reports and structured Excel/CSV records.

Clean and chunk text while preserving source metadata such as document name, page, row, and incident ID.

Create multilingual semantic embeddings using intfloat/multilingual-e5-base.

Store and search the embeddings with FAISS using normalized inner-product similarity (cosine-style search).

Generate answers with mistralai/Mistral-Nemo-Instruct-2407 through Hugging Face Transformers.

Maintain conversation memory for multi-turn investigation questions.

Rewrite follow-up questions into standalone questions before retrieval (query contextualization).

Perform cross-document reasoning across multiple incidents and evidence sources.

Track the sources used for every generated answer.

Reduce hallucination risk by explicitly grounding answers in retrieved evidence.

Save and reload the FAISS index, metadata, and experiment configuration as reusable artifacts.

Knowledge Base

The notebook processes real files from the configured Kaggle dataset directories and produces:

10 PDF files

2 structured table files (Excel/CSV)

32 knowledge items after extraction, cleaning, chunking, and record conversion

Each knowledge item keeps provenance information including:

source_type

source_name

page or row_number

incident_id

retrieval score

RAG Pipeline

PDF Reports + Excel/CSV Records
                │
                ▼
      Extraction & Cleaning
                │
                ▼
      Chunking / Record-to-Text
                │
                ▼
 Multilingual E5 Embeddings
                │
                ▼
          FAISS Index
                │
User Question + Conversation Memory
                │
                ▼
      Query Contextualization
                │
                ▼
       Semantic Retrieval
                │
                ▼
        Evidence + Metadata
                │
                ▼
   RAG Prompt with Guardrails
                │
                ▼
     Mistral-Nemo Generation
                │
                ▼
 Answer + Evidence Sources

Conversation Memory & Reasoning

AI Detective supports follow-up questions by keeping a conversation buffer and using the previous turns to contextualize ambiguous questions.

For example, a follow-up such as “Was it mentioned in another case?” can be rewritten into a standalone question that identifies what “it” refers to before semantic retrieval is performed.

The system also supports cross-document reasoning. When the same vehicle description appears in multiple incidents, the assistant is instructed to describe this as a possible connection rather than confirmed proof unless the evidence establishes more.

Evidence-Grounded Answers

The final RAG prompt instructs the assistant to:

Use only the provided evidence and conversation context.

Never invent unsupported facts.

Say when the evidence is insufficient.

Mention relevant incident IDs when synthesizing multiple sources.

Avoid treating a repeated detail across incidents as definitive proof of a connection.

Every answer is accompanied by retrieved source metadata so the reasoning can be traced back to the underlying records.

Evaluation

The notebook includes automated evaluation and a final test suite covering:

Direct Retrieval

PDF Evidence

Cross-Document Reasoning

Hallucination Control

Conversation Memory / Follow-up Questions

The automated evaluation in the notebook checks whether the correct incident evidence is retrieved and whether the generated answer mentions the relevant incident ID. On the sampled evaluation cases in the notebook, both reported metrics reached 100.0%:

Retrieval accuracy: 100.0%

Answer ID-mention rate: 100.0%

The final test suite is broader and is included for detailed inspection of individual answers and retrieved sources.

Saved Artifacts

The notebook saves reusable artifacts in the artifacts/ directory:

artifacts/
├── index.faiss
├── metadata.pkl
└── config.json

The configuration file stores the embedding model, LLM model, retrieval settings, chunk size/overlap, index type, knowledge-item count, and creation timestamp.

Technology Stack

Python

PyPDF2 — PDF text extraction

Pandas — structured data processing

Sentence Transformers — text embeddings

intfloat/multilingual-e5-base — multilingual embedding model

FAISS — vector similarity search

Hugging Face Transformers — LLM loading and generation

mistralai/Mistral-Nemo-Instruct-2407 — local instruction-tuned LLM

PyTorch — model execution and GPU support

Kaggle Notebook — knowledge-base preparation and experimentation environment

Streamlit — live demo interface

Important Scope Note

This notebook uses pretrained models only. No model is trained or fine-tuned in the notebook itself.

The notebook focuses on knowledge-base preparation, retrieval, RAG reasoning, conversation memory, evaluation, and artifact persistence. A later application layer can load the saved artifacts for a service or user interface.

Live Demo

👉 Open the AI Detective Live Demo

GitHub Repository

👉 View the Project on GitHub

Author

Raouf Ahmed
