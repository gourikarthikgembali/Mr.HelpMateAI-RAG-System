# Mr.HelpMateAI – Insurance Policy RAG System

A lightweight Retrieval‑Augmented Generation (RAG) workflow that ingests an insurance policy PDF, chunks pages, embeds text with **SentenceTransformer (all‑MiniLM‑L6‑v2)**, stores vectors in **ChromaDB**, and answers user questions using **semantic search + cache + (optional) re‑ranking**. 

---

## What this project does
- **Parse & normalize PDF** pages/tables using `pdfplumber`; build a dataframe of page text + metadata (page no., policy name). 
- **Fixed‑size chunking** (≈500 char/word budget) per page to create retrieval units with per‑chunk metadata. 
- **Embeddings** via `SentenceTransformer('all-MiniLM-L6-v2')`; attach vectors to each chunk. 
- **Vector store** with `chromadb.PersistentClient()`; create a main collection (`RAG_on_Insurance`) and a **query cache** collection (`Insurance_Cache`). 
- **Semantic search with cache**: check cache first; if miss, query main collection, then write the query & top‑k hits back to cache for fast repeats. 
- **Re‑ranking (optional)** using a cross‑encoder to boost relevance of retrieved chunks. 

---

## Files
- `Mr.HelpMateAI_RAG_System_Project_.ipynb` – end‑to‑end notebook/script: PDF parsing, chunking, embeddings, ChromaDB collections, cache, search, and cross‑encoder re‑rank. 
- `Principal-Sample-Life-Insurance-Policy.pdf` - a text document which describes various insurance policies

---

## Quick start
1. **Place the policy PDF** (e.g., `Principal-Sample-Life-Insurance-Policy.pdf`) beside the script. 
2. Run the notebook/script to:
   - build `insurance_pdf_data` (page text + metadata),
   - generate embeddings with MiniLM,
   - populate ChromaDB collections,
   - call `search(query)` for semantic results with cache. 

---

## Requirements
- Python 3.9+
- `pdfplumber`, `pandas`, `sentence-transformers`, `chromadb`, `tiktoken`, `openai` (if you wire LLM steps), and standard libs (`json`, `os`, `pathlib`). 

Install (example):
```bash
pip install pdfplumber pandas sentence-transformers chromadb tiktoken openai
```

---

## Notes
- Tables vs text are separated using bounding‑box checks; clustered to preserve reading order. 
- Cache logic stores the **query** as a document and the retrieved ids/docs/distances as metadata for quick recall. 
- Telemetry warnings from ChromaDB are non‑blocking in local runs. 

Maintainer: Gouri Karthik Gembali
