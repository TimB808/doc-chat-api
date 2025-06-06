from fastapi import FastAPI
from app.api import upload, chat

app = FastAPI()

app.include_router(upload.router, prefix="/api")
app.include_router(chat.router, prefix="/api")


@app.get("/")
def read_root():
    return {"message": "Ask My PDF is up and running!"}
