FROM ml_edition

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc g++ curl git \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/root/.local/bin:$PATH"

COPY requirements.txt ./

RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "perplexity_search:app", "--host", "0.0.0.0", "--port", "8000"]
