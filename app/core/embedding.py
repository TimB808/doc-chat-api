from dotenv import load_dotenv
load_dotenv()
import os
import tiktoken
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Use tiktoken tokenizer (for OpenAI models)
ENCODING_NAME = "cl100k_base"  # suitable for text-embedding-ada-002
MAX_TOKENS = 500  # chunk size
TOKEN_OVERLAP = 50  # to retain some context

def chunk_text(text: str, max_tokens=MAX_TOKENS, overlap=TOKEN_OVERLAP):
    """
    Splits long text into overlapping chunks based on token count.
    """
    encoding = tiktoken.get_encoding(ENCODING_NAME)
    tokens = encoding.encode(text)

    chunks = []
    start = 0
    while start < len(tokens):
        end = start + max_tokens
        chunk_tokens = tokens[start:end]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)
        start += max_tokens - overlap  # move forward with overlap

    return chunks

def get_embeddings(chunks: list[str]) -> list[dict]:
    """
    Calls OpenAI to get embeddings for each text chunk.
    Returns a list of dicts: {'text': ..., 'embedding': [...]}
    """
    results = []
    for chunk in chunks:
        response = client.embeddings.create(
            input=chunk,
            model="text-embedding-ada-002"
        )
        embedding = response.data[0].embedding
        results.append({
            "text": chunk,
            "embedding": embedding
        })
    return results
