
## Описание

Smart Place Search API — это сервис на FastAPI для поиска интересных мест Ростова-на-Дону и области командно, с использованием:
- локальной семантической базы на ruBERT-эмбеддингах,
- генеративного до-поиска через Perplexity API (если не найдено релевантных результатов).

Система фильтрует нерелевантные и технические запросы по смыслу.

***

## Как запустить

### 1. Клонируй репозиторий и установи зависимости

```bash
git clone https://github.com/Toster1903/hahaton.git
cd hahaton
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Создай `.env`

Добавь файл `.env` с переменной:

```
PPLX_API_KEY=ваш_ключ_от_perplexity
```

### 3. Запусти FastAPI сервер

```bash
uvicorn perplexity_search:app --host 0.0.0.0 --port 8000 --reload      
```

### 4. Про примеры запросов

POST `/search_v2`
```json

{
  "user_query": "string",
  "places": [
    {"title": "string"},
    {"title": "string"}
  ],
  "max_results": 2
}

```

Ответ — структура:

- Если есть релевантные места:  
```json
{
  "results": [
    {
      "title": "string",
      "description": "string",
      "price": 0,
      "time": 0,
      "similarity_score": 0
    }
  ],
  "best_similarity": 0,
  "worst_similarity": 0
}

```



***

## Структура проекта


- `get_response_user.py` — FastAPI сервер и логика поиска
- `places_meta.json`, `embeddings.npy` — база мест и эмбеддинги
- `test_RUBERT.py`, `ruBERT_recognize.py` — скрипты генерации/теста эмбеддингов
- `rostov_all_places_full.json` — полный JSON со всеми местами
- `requirements.txt` — зависимости для повторного запуска

***

## Что внутри

- **Semantic Search:** ruBERT находит наиболее похожие записи по смыслу среди всех мест Ростова.
- **Perplexity Fallback:** если совпадения в базе "слабые", системой будет дополнен ответ через Perplexity API с инструкцией вернуть только релевантные места/пустой массив при off-topic.
- **Фильтр темы:** если ваш запрос не по теме туризма или города — сервис корректно сообщит об этом.

***


## Лицензия

MIT License


