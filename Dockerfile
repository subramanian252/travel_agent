FROM python:3.13-slim

WORKDIR /app

RUN pip install uv

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-install-project

COPY . .

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
