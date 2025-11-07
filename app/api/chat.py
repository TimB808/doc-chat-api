import logging
import os
from typing import Annotated, Optional

from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from openai import OpenAI, OpenAIError
from pydantic import BaseModel, Field, field_validator

from app.core.vector_store import hybrid_search, query_embeddings

logger = logging.getLogger("docchat")


# Configure OpenAI
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

client = OpenAI(api_key=api_key)

router = APIRouter()


class ChatRequest(BaseModel):
    question: str = Field(..., description="The question to ask about the document")
    # if omitted/blank, search across ALL documents
    file_id: Optional[str] = Field(
        None,
        description="ID of the document to query; if omitted, search across all documents",
    )
    alpha: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="0=BM25-only, 1=semantic-only (default 0.5 = hybrid)",
    )

    # normalise "" or whitespace to None
    @field_validator("file_id", mode="before")
    @classmethod
    def _empty_to_none(cls, v):
        if v is None:
            return None
        if isinstance(v, str):
            v = v.strip()
            return v or None
        return v


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
        response = client.embeddings.create(
            input=text,
            model="text-embedding-3-small",  # previously "text-embedding-ada-002"
        )
        return response.data[0].embedding
    except OpenAIError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting embedding from OpenAI: {str(e)}",
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
    try:
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")

        # 1) Embed the question
        question_embedding = get_embedding(request.question)

        # 2) Retrieval
        context_chunks = []  # type: ignore[var-annotated]

        if request.file_id:
            # STRICT file-scoped search
            try:
                context_chunks = await hybrid_search(
                    query=request.question,
                    question_embedding=question_embedding,
                    file_id=request.file_id,
                    k=5,
                    alpha=request.alpha,
                )
            except Exception as e:
                logger.exception(
                    "hybrid_search (scoped) failed; falling back to vector-only: %s", e
                )
                context_chunks = query_embeddings(
                    question_embedding=question_embedding,
                    file_id=request.file_id,
                    top_k=5,
                )

            if not context_chunks:
                # strict: do NOT search across all docs when a file_id was given
                raise HTTPException(
                    status_code=404,
                    detail=f"No relevant content found for file_id: {request.file_id}",
                )
        else:
            # Global search across ALL docs (only when file_id is empty/None)
            try:
                context_chunks = await hybrid_search(
                    query=request.question,
                    question_embedding=question_embedding,
                    file_id=None,
                    k=5,
                    alpha=request.alpha,
                )
            except Exception as e:
                logger.exception(
                    "hybrid_search (global) failed; falling back to vector-only: %s", e
                )
                context_chunks = query_embeddings(
                    question_embedding=question_embedding,
                    file_id=None,
                    top_k=5,
                )

            if not context_chunks:
                raise HTTPException(
                    status_code=404,
                    detail="No relevant content found across all documents.",
                )

        # Optional: debug top-5 scores while testing
        if os.getenv("DEBUG_HYBRID") == "1" and context_chunks:
            top = [
                (
                    i.get("id"),
                    round(float(i.get("score", i.get("final_score", 0.0))), 3),
                    i.get("title") or i.get("page"),
                )
                for i in context_chunks[:5]
            ]
            logger.debug("TOP5 (id, score, title/page): %s", top)

        # 3) Build context string
        context_text = "\n\n".join(chunk["text"] for chunk in context_chunks)[:12000]

        # 4) LLM answer
        answer = get_chat_completion(context_text, request.question)

        # 5) Response
        def _clamp01(x: float) -> float:
            return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x

        context = [
            ContextChunk(
                text=chunk["text"],
                score=_clamp01(
                    float(chunk.get("final_score") or chunk.get("score", 0.0))
                ),
            )
            for chunk in context_chunks
        ]

        return ChatResponse(answer=answer, context=context)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
