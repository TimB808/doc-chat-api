import os
from typing import List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, confloat
import openai
from openai.error import OpenAIError

from app.core.vector_store import query_embeddings, SearchResult

# Configure OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

router = APIRouter()

class ChatRequest(BaseModel):
    question: str = Field(..., description="The question to ask about the document")
    file_id: str = Field(..., description="ID of the document to query")

class ContextChunk(BaseModel):
    text: str = Field(..., description="Text chunk from the document")
    score: confloat(ge=0, le=1) = Field(..., description="Similarity score (0-1) where 1 is most similar")

class ChatResponse(BaseModel):
    answer: str = Field(..., description="AI-generated answer to the question")
    context: List[ContextChunk] = Field(..., description="Relevant document chunks used for context with their similarity scores")

def get_embedding(text: str) -> List[float]:
    """Get embedding from OpenAI API."""
    try:
        response = openai.Embedding.create(
            input=text,
            model="text-embedding-ada-002"
        )
        return response["data"][0]["embedding"]
    except OpenAIError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting embedding from OpenAI: {str(e)}"
        )

def get_chat_completion(context: str, question: str) -> str:
    """Get chat completion from OpenAI API."""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system", 
                    "content": "You are a helpful assistant answering questions based on the provided document."
                },
                {
                    "role": "user", 
                    "content": f"{context}\n\nQuestion: {question}"
                }
            ],
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content
    except OpenAIError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting chat completion from OpenAI: {str(e)}"
        )

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Answer questions about a previously uploaded document."""
    try:
        # Validate inputs
        if not request.question.strip():
            raise HTTPException(
                status_code=400,
                detail="Question cannot be empty"
            )
        if not request.file_id.strip():
            raise HTTPException(
                status_code=400,
                detail="file_id cannot be empty"
            )
            
        # 1. Get embedding for the question
        question_embedding = get_embedding(request.question)
        
        # 2. Query vector store for relevant chunks
        context_chunks = query_embeddings(
            question_embedding=question_embedding,
            file_id=request.file_id,
            top_k=5
        )
        
        if not context_chunks:
            raise HTTPException(
                status_code=404,
                detail=f"No relevant content found for file_id: {request.file_id}"
            )
        
        # 3. Build context string
        context_text = "\n\n".join(chunk["text"] for chunk in context_chunks)
        
        # 4. Get completion from OpenAI
        answer = get_chat_completion(context_text, request.question)
        
        # 5. Convert context chunks to Pydantic models
        context = [ContextChunk(**chunk) for chunk in context_chunks]
        
        return ChatResponse(
            answer=answer,
            context=context
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        # Log unexpected errors and return 500
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )
