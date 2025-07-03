# doc-chat-api
A FastAPI chatbot app that lets users upload PDFs and ask questions about the content using OpenAI and LanceDB

## Project structure

root/
├── app/
│   ├── main.py
│   ├── __init__.py
│   ├── core/
│   │   ├── vector_store.py
│   │   ├── pdf_parser.py
│   │   └── embedding.py
│   ├── ui/
│   │   └── streamlit_app.py
│   ├── api/
│   │   ├── upload.py
│   │   └── chat.py
├── data/
│   ├── pdfs/
│   └── lancedb/
│       └── document_embeddings.lance/
├── .git/
├── .mypy_cache/
├── .ruff_cache/
├── venv/
├── README.md
├── requirements.txt
├── pyproject.toml
├── .pre-commit-config.yaml
├── .gitignore
├── start.sh
└── Makefile

## 🚀 Usage

### Start FastAPI backend:

```make run-backend```

Launch Streamlit UI:

```make run-ui```

🛠️ Developer Tools

    ```make all``` — run linter, formatter, mypy, pre-commit

    Pre-commit hooks: ruff, mypy, trailing whitespace cleanup

## 📦 Tech Stack

| Component         | Tool/Library   | Purpose/Notes                        |
|-------------------|----------------|--------------------------------------|
| API               | FastAPI        | REST API framework                   |
| Data Validation   | Pydantic       | Request/response models              |
| Embeddings        | OpenAI         | Text embeddings                      |
| Tokenization      | tiktoken       | Token counting for chunking          |
| Vector Search     | LanceDB        | Vector database                      |
| PDF Parsing       | PyMuPDF (fitz) | PDF text extraction                  |
| Env Management    | python-dotenv  | Load secrets/config from .env        |
| UI                | Streamlit      | Web UI                               |
| Linting           | Ruff           | Code linting (dev tool)              |
| Type Checking     | MyPy           | Static type checking (dev tool)      |

uvicorn is used as the ASGI server (used in scripts, not app code)


## Notes

### Similarity Scores

The app returns a relevance score for each text chunk, indicating how well it matches the user's question.

When available, the score column is used directly (a value between 0 and 1, where 1 = most relevant). If only _distance is returned, the app converts it using 1 - distance (for cosine similarity).

### To set up local env:

```run ./start.sh```

(need to chmod +x start.sh once to make it executable.)


### Pinned versions:
To ensure compatibility and avoid ABI conflicts in the Python data ecosystem, this project currently uses:

- `pyarrow==16.0.0` — for full support of `fixed_size_list()` in LanceDB schemas
- `lancedb==0.4.4` — a stable release compatible with fixed-size vector types
- `numpy==1.24.4` — pinned to avoid compatibility issues with NumPy 2.x and older PyArrow/LanceDB binaries

This configuration supports production-grade vector search with clean schema definitions. Future branches may upgrade to `lancedb>=0.6` and `numpy>=2.0` to take advantage of hybrid search and metadata filtering when upstream compatibility is guaranteed.
