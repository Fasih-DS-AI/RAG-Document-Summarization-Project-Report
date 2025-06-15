import os
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline
from typing import List, Tuple
import time

class RAGSummarizer:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", llm_model: str = "facebook/bart-large-cnn"):
        """Initialize the RAG summarizer with embedding and LLM models."""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        self.embedder = SentenceTransformer(model_name)
        self.dimension = self.embedder.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(self.dimension)
        self.chunks = []
        self.llm = pipeline("summarization", model=llm_model, device=-1)

    def ingest_document(self, file_path: str) -> List[str]:
        """Ingest and split document into chunks."""
        if file_path.endswith('.pdf'):
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        elif file_path.endswith(('.txt', '.md')):
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
        else:
            raise ValueError("Unsupported file format. Use PDF, TXT, or Markdown.")

        # Split text into semantic chunks
        self.chunks = self.text_splitter.split_text(text)
        return self.chunks

    def create_embeddings(self) -> None:
        """Create and store embeddings for document chunks in FAISS."""
        embeddings = self.embedder.encode(self.chunks, convert_to_numpy=True)
        self.index.add(embeddings)

    def retrieve_chunks(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Retrieve top-k relevant chunks for the query."""
        query_embedding = self.embedder.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_embedding, top_k)
        results = [(self.chunks[idx], distances[0][i]) for i, idx in enumerate(indices[0])]
        return results

    def generate_summary(self, retrieved_chunks: List[Tuple[str, float]], max_length: int = 200) -> dict:
        """Generate summary from retrieved chunks."""
        start_time = time.time()
        context = " ".join([chunk for chunk, _ in retrieved_chunks])
        summary = self.llm(context, max_length=max_length, min_length=50, do_sample=False)[0]['summary_text']
        latency = time.time() - start_time
        
        return {
            "summary": summary,
            "retrieved_context": [chunk for chunk, _ in retrieved_chunks],
            "similarity_scores": [float(score) for _, score in retrieved_chunks],
            "latency": latency
        }

    def process_document(self, file_path: str, query: str = "Summarize this document") -> dict:
        """Process document and generate summary."""
        self.chunks = self.ingest_document(file_path)
        self.create_embeddings()
        retrieved_chunks = self.retrieve_chunks(query)
        result = self.generate_summary(retrieved_chunks)
        return result

def main():
    # Initialize summarizer
    summarizer = RAGSummarizer()
    
    # Process sample document
    sample_pdf = "sample_document.pdf"  # Replace with actual document path
    if not os.path.exists(sample_pdf):
        print("Sample document not found. Please provide a valid PDF, TXT, or Markdown file.")
        return
    
    result = summarizer.process_document(sample_pdf)
    
    # Display results
    print("\n=== Summary ===")
    print(result["summary"])
    print("\n=== Retrieved Context ===")
    for i, chunk in enumerate(result["retrieved_context"], 1):
        print(f"Chunk {i}:\n{chunk}\n")
    print("=== Metadata ===")
    print(f"Similarity Scores: {result['similarity_scores']}")
    print(f"Latency: {result['latency']:.2f} seconds")

if __name__ == "__main__":
    main()