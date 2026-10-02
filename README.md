# Agentic RAG using CrewAI

A local Agentic RAG application built with CrewAI, Qdrant, Chonkie, and Ollama.

Users can upload a PDF and ask questions about it. The application:
- extracts and chunks the document
- creates semantic embeddings
- stores vectors in an in-memory Qdrant database
- retrieves relevant content
- uses CrewAI agents to generate the final response
- runs locally with Ollama using `qwen2.5-coder:7b`

## Tech Stack

- Python
- CrewAI
- Qdrant
- Chonkie
- MarkItDown
- Ollama
- Qwen 2.5 Coder
- Streamlit