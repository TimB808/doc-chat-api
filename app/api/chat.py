import os
from typing import Annotated

from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from openai import OpenAI, OpenAIError
from pydantic import BaseModel, Field

from app.core.vector_store import query_embeddings

# Configure OpenAI
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

client = OpenAI(api_key=api_key)

router = APIRouter()


class ChatRequest(BaseModel):
    question: str = Field(..., description="The question to ask about the document")
    file_id: str = Field(..., description="ID of the document to query")


class ContextChunk(BaseModel):
    text: str = Field(..., description="Text chunk from the document")
    score: Annotated[
        float,
        Field(ge=0, le=1, description="Similarity score (0-1) where 1 is most similar"),
    ]


class ChatResponse(BaseModel):
    answer: str = Field(..., description="AI-generated answer to the question")
    context: list[ContextChunk] = Field(
        ...,
        description="Relevant document chunks used for context with their similarity scores",
    )


def get_embedding(text: str) -> list[float]:
    """Get embedding from OpenAI API."""
    try:
        response = client.embeddings.create(input=text, model="text-embedding-ada-002")
        embedding = response.data[0].embedding
        return embedding
    except OpenAIError as e:
        raise HTTPException(
            status_code=500, detail=f"Error getting embedding from OpenAI: {str(e)}"
        ) from e


def get_chat_completion(context: str, question: str) -> str:
    """Get chat completion from OpenAI API based on document context."""
    try:
        system_prompt = (
            "You are a concise and helpful assistant answering questions about a document. "
            "You are given excerpts from the document and a user question. "
            "Use the information to answer naturally, as if you had read the full document yourself. "
            "If the answer isn't clearly stated, respond honestly and say so — don't make assumptions. "
            "Do not refer to 'text chunks' or 'context' in your reply."
        )

        user_prompt = f"Context:\n{context}\n\n" f"Question:\n{question}"

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=500,
        )

        if not response.choices:
            return (
                "Sorry, I couldn't generate a response — the model returned no answer."
            )

        content = response.choices[0].message.content
        return (
            content.strip()
            if content
            else "Sorry, I couldn't generate a valid answer as the model returned no content. Please try again."
        )

    except OpenAIError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting chat completion from OpenAI: {str(e)}",
        ) from e


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Answer questions about a previously uploaded document."""
    try:
        # Validate inputs
        if request.question is None or not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        if request.file_id is None or not request.file_id.strip():
            raise HTTPException(status_code=400, detail="file_id cannot be empty")

        # 1. Get embedding for the question
        question_embedding = get_embedding(request.question)

        # 2. Query vector store for relevant chunks
        # Note: To use hybrid search (semantic + keyword BM25), replace query_embeddings with:
        #   context_chunks = await hybrid_search(
        #       query=request.question,
        #       question_embedding=question_embedding,
        #       file_id=request.file_id,
        #       k=5,
        #       alpha=0.5  # 0.5 = equal weight, higher = more semantic, lower = more keyword
        #   )
        context_chunks = query_embeddings(
            question_embedding=question_embedding, file_id=request.file_id, top_k=5
        )

        if not context_chunks:
            raise HTTPException(
                status_code=404,
                detail=f"No relevant content found for file_id: {request.file_id}",
            )

        # 3. Build context string (truncate to fit GPT token limits)
        context_text = "\n\n".join(chunk["text"] for chunk in context_chunks)[:12000]

        # 4. Get completion from OpenAI
        answer = get_chat_completion(context_text, request.question)

        # 5. Convert context chunks to Pydantic models
        context = [ContextChunk(**chunk) for chunk in context_chunks]

        return ChatResponse(answer=answer, context=context)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
