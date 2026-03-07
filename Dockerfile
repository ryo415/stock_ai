FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
COPY docs ./docs
COPY scripts ./scripts
COPY AGENTS.md USAGE.md ./

RUN pip install --upgrade pip && pip install -e .

EXPOSE 8501

CMD ["streamlit", "run", "src/stock_ai/web/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
