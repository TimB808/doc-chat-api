# doc-chat-api
A FastAPI chatbot app that lets users upload PDFs and ask questions about the content using OpenAI and LanceDB

## Project structure

```
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
├── venv/
├── README.md
├── requirements.txt
├── pyproject.toml
├── .pre-commit-config.yaml
├── .gitignore
├── start.sh
└── Makefile
```

## 🚀 Usage

### Start FastAPI backend:

```make run-backend```

Starts uvicorn at http://localhost:8000.

### Launch Streamlit UI:

```make run-ui```

Opens the chat interface at http://localhost:8501.

### Build Docker image for local testing

```make docker-build-local```

Creates a Docker image tagged as doc-chat-api. Useful for quick local container testing.

```make docker-run-local```

Runs the local testing image on http://localhost:8000.


### Build Cloud-ready Docker image

```make docker-build-cloud```

Builds the production Docker image used for deployment to Cloud Run (tagged as europe-west3-docker.pkg.dev/le-wagon-data-science-376310/doc-chat-api/doc-chat-api:latest).

```make docker-run-cloud```

Runs the Cloud Run production image locally at http://localhost:8000 for testing before deployment.


### Deploy Backend to Google Cloud Run

Before the first deployment:

```gcloud auth login```
```gcloud auth configure-docker```
```chmod +x deploy.sh```

Then deploy with:

```make deploy```

Builds the image, pushes it to Google Container Registry, and deploys it as a Cloud Run service.

Default backend URL (after deployment): https://doc-chat-api-<PROJECT_ID>.europe-west3.run.app

### Deploying the Streamlit UI

1. Local Testing

Build and run the Streamlit UI container locally:

```make docker-build-ui```
``make docker-run-ui```

This serves the UI at http://localhost:8501.

It points to http://localhost:8000 by default for the backend (or uses DOC_CHAT_API_URL if set).

2. Cloud Deployment

a. Build the UI Docker Image

```make docker-build-ui-cloud```

b. Deploy the UI to Cloud Run

```make deploy-ui```

Deploys the Streamlit UI as a Cloud Run service doc-chat-ui.

Sets DOC_CHAT_API_URL to the backend's public URL.

3. Environment Variable

The UI dynamically determines the backend URL:

```API_URL = os.getenv("DOC_CHAT_API_URL", "http://localhost:8000")```

When running locally, it defaults to http://localhost:8000.

On Cloud Run, the Makefile sets this to the backend URL automatically.

4. Accessing the UI

After deployment:

```gcloud run services describe doc-chat-ui --platform managed --region europe-west3 --format "value(status.url)"```

Or visit the public URL, e.g.:

```https://doc-chat-ui-<PROJECT_ID>.europe-west3.run.app```


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

### To set up local environment:

To run locally with a virtual environment:

```run ./start.sh```

(need to run ```chmod +x start.sh``` once to make it executable.)


### Pinned versions:
To ensure compatibility and avoid ABI conflicts in the Python data ecosystem, this project currently uses:

- `pyarrow==16.0.0` — for full support of `fixed_size_list()` in LanceDB schemas
- `lancedb==0.4.4` — a stable release compatible with fixed-size vector types
- `numpy==1.24.4` — pinned to avoid compatibility issues with NumPy 2.x and older PyArrow/LanceDB binaries

This configuration supports production-grade vector search with clean schema definitions. Future branches may upgrade to `lancedb>=0.6` and `numpy>=2.0` to take advantage of hybrid search and advanced filtering when upstream compatibility is guaranteed.
