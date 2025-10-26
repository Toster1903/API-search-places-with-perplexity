FROM ml_edition

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc g++ curl git \
    && rm -rf /var/lib/apt/lists/*

# ENV POETRY_VERSION=1.8.2
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"

COPY pyproject.toml poetry.lock ./

RUN poetry config virtualenvs.create false \
 && poetry install --only main --no-interaction --no-ansi

COPY . .

EXPOSE 8000
CMD ["uvicorn", "perplexity_seaarch:app", "--host", "0.0.0.0", "--port", "8000"]
