# RAG with NVIDIA NIM

A retrieval-augmented generation app that swaps in NVIDIA's NIM (NVIDIA Inference Microservices) for embeddings and LLM inference, comparing performance against other RAG backends.

## How it works
- Loads PDF documents (including "Attention Is All You Need" and a robotics paper) and splits them into chunks
- - Embeds chunks using NVIDIA NIM embedding models
  - - Retrieves relevant context and generates answers via an NVIDIA NIM-hosted LLM
    - - Interactive UI built with Streamlit
     
      - ## Tech stack
      - Python, LangChain, NVIDIA NIM, Streamlit
     
      - ## Run locally
      - pip install -r requirements.txt
      - streamlit run st_app.py
     
      - Requires an NVIDIA NIM API key.
      - 
