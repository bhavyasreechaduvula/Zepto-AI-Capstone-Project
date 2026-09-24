FROM python:3.12-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    pydantic \
    chromadb \
    sentence-transformers \
    langgraph \
    langchain-core

EXPOSE 8000

CMD ["uvicorn", "support_assistant.api:app", "--host", "0.0.0.0", "--port", "8000"]

