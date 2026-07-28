FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src
COPY config ./config

RUN pip install --no-cache-dir -e .

# Hugging Face Spaces는 기본적으로 7860 포트를 사용한다.
EXPOSE 7860

CMD ["uvicorn", "accounting_rag.api.app:app", "--host", "0.0.0.0", "--port", "7860"]
