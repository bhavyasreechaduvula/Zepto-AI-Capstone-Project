from fastapi import FastAPI
from pydantic import BaseModel

from support_assistant.graph import ask_question
from support_assistant.schemas import AskResponse


app = FastAPI(
    title="Zepto Policy Support Assistant"
)


class AskRequest(BaseModel):
    query: str


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    result = ask_question(request.query)

    return AskResponse(
        answer=result["answer"],
        sources=result["sources"],
        confidence=result["confidence"],
    )