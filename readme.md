# RAG Summarizer Project

This repository contains a Retrieval-Augmented Generation (RAG) based document summarization tool implemented in Python. The tool ingests documents (PDF, TXT, or Markdown), splits them into chunks, creates vector embeddings, retrieves relevant content using a FAISS vector database, and generates concise summaries using a pre-trained BART model. This project is designed to meet the requirements of a coursework assignment, summarizing at least three documents and providing a detailed report.

## Table of Contents
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Sample Documents](#sample-documents)
- [Output](#output)
- [Troubleshooting](#troubleshooting)
- [Project Structure](#project-structure)
- [Visual Pipeline Overview](#visual-pipeline-overview)
- [Contributing](#contributing)
- [License](#license)

## Features
- Supports PDF, TXT, and Markdown document formats.
- Uses `SentenceTransformers` with `all-MiniLM-L6-v2` for embedding generation.
- Implements FAISS for efficient vector storage and retrieval.
- Employs `facebook/bart-large-cnn` (or `distilbart-cnn-12-6` as an alternative) for summarization.
- Provides debug output for chunking, retrieval, and summarization steps.
- Generates metadata including similarity scores and latency.

## Requirements
- Python 3.11 or 3.12
- Required packages (listed in `requirements.txt`):
  - `PyPDF2==3.0.1`
  - `langchain==0.0.348`
  - `sentence-transformers==2.2.2`
  - `faiss-cpu==1.7.4`
  - `transformers==4.36.2`
  - `numpy==1.26.2`
  - `huggingface_hub==0.19.4`

