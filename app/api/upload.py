from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from app.core.pdf_parser import extract_text_from_pdf
from app.core.embedding import chunk_text, get_embeddings
from app.core.vector_store import store_embeddings
import os
import uuid

router = APIRouter()

# Create a temporary folder for uploaded PDFs
UPLOAD_DIR = "data/pdfs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Save uploaded file to disk with a unique name
    file_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}.pdf")

    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Extract text from the PDF
    text = extract_text_from_pdf(file_path)
    if not text.strip():
        raise HTTPException(status_code=422, detail="No readable text found in PDF.")

    # Generate embeddings for the text
    chunks = chunk_text(text)
    embeddings = get_embeddings(chunks)
    
    # Store embeddings in vector database
    store_embeddings(file_id, embeddings)

    return JSONResponse(
        content={
            "message": "PDF uploaded and processed successfully",
            "file_id": file_id,
            "char_count": len(text)
        },
        status_code=200
    )
