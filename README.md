# doc-chat-api
A FastAPI app that lets users upload PDFs and ask questions about the content using OpenAI and LanceDB



## Notes

### Scores returned: 
Scores represent 1 minus cosine distance, indicating closeness between question and document chunk. A higher score means a more relevant match.

To set up local env: 

run ./start.sh

(need to chmod +x start.sh once to make it executable.)


### Pinned versions:
This project demonstrates a document-based chatbot powered by modern tools such as FastAPI, OpenAI embeddings, and LanceDB for vector storage and semantic search. To ensure compatibility with recent changes in the NumPy ecosystem, we currently pin pyarrow and lancedb to stable versions that avoid issues with NumPy 2.x. The architecture remains production-grade and aligned with best practices in Retrieval-Augmented Generation (RAG). Future branches may explore newer features such as hybrid search, structured metadata filtering, and updated dependencies once fully supported upstream.