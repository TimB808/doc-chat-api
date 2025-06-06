# doc-chat-api
A FastAPI app that lets users upload PDFs and ask questions about the content using OpenAI and LanceDB



## Notes

Scores returned: 
Scores represent 1 minus cosine distance, indicating closeness between question and document chunk. A higher score means a more relevant match.