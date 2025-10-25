import os
import json
import numpy as np
import torch
import re
from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from perplexity import Perplexity
from dotenv import load_dotenv


load_dotenv()


PERPLEXITY_API_KEY = os.getenv("PPLX_API_KEY")
METADATA_FILE = "places_meta.json"
EMBEDDINGS_FILE = "embeddings.npy"
MODEL_NAME = "cointegrated/rubert-tiny"
#MODEL_NAME = "sberbank-ai/sbert_large_nlu_ru"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SIMILARITY_THRESHOLD = 0.6
OFF_TOPIC_THRESHOLD = 0.3

# Категории
CATEGORIES = {
    0: "Культурно-развлекательный отдых",
    1: "Гастрономический и социальный досуг",
    2: "Активный и спортивный отдых",
    3: "Природный и оздоровительный отдых",
    4: "Развлечения и аттракционы",
    5: "Коммерческие мероприятия"
}

# Ключевые слова для каждой категории
CATEGORY_KEYWORDS = {
    0: ["театр", "опера", "балет", "концерт", "фестиваль", "музей", "выставка", "кинотеатр", "галерея", "библиотека", "лекция", "искусство", "культура", "драма", "мозаика", "памятник", "история", "храм", "собор", "монастырь", "мемориал", "архитектура"],
    1: ["ресторан", "кафе", "кофейня", "бар", "гастробар", "дегустация", "пикник", "барбекю", "вечеринка", "клуб", "кухня", "еда", "десерты", "выпечка", "вино", "пиво", "виски", "стейк", "мясо", "паста", "пицца"],
    2: ["фитнес", "бассейн", "аквапарк", "лыжи", "коньки", "велосипед", "футбол", "теннис", "пейнтбол", "спорт", "стадион", "аквакомплекс", "тренажерный зал", "единоборства", "кроссфит"],
    3: ["пляж", "кемпинг", "рыбалка", "санаторий", "спа", "йога", "оздоровительный", "экотуризм", "поход", "природа", "река", "термы", "лес", "заповедник", "озеро", "парк"],
    4: ["парк", "цирк", "зоопарк", "квест", "боулинг", "бильярд", "каток", "скейт", "батут", "аттракцион", "развлечения", "дельфинарий", "колесо обозрения", "набережная"],
    5: ["выставка", "ярмарка", "форум", "конференция", "коммерческий", "рынок", "базар", "торговый центр", "экспо"]
}


app = FastAPI(title="Smart Place Search API", version="1.0")


class GlobalResources:
    tokenizer = None
    model = None
    places = None
    embeddings = None
    perplexity_client = None


class SearchRequest(BaseModel):
    query: str
    max_results: int = 5


class PlaceResult(BaseModel):
    title: str
    description: str
    tags: List[str]
    transport: str
    address: str
    price: int
    time: int
    similarity_score: float
    category: int  # Новое поле: категория от 0 до 5


class SearchResponse(BaseModel):
    source: str
    results: List[PlaceResult]
    best_similarity: float


def categorize_place(title: str, description: str, tags: List[str]) -> int:
    """
    Определяет категорию места на основе названия, описания и тегов.
    Возвращает число от 0 до 5.
    """
    text = (title + " " + description + " " + " ".join(tags)).lower()
    
    category_scores = {cat: 0 for cat in CATEGORIES}
    for cat, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                category_scores[cat] += 1
    
    best_cat = max(category_scores, key=category_scores.get)
    if category_scores[best_cat] == 0:
        return 1  # по умолчанию гастрономия
    return best_cat


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def l2_normalize(embeddings):
    return torch.nn.functional.normalize(embeddings, p=2, dim=1)


def generate_query_embedding(text: str) -> np.ndarray:
    encoded_input = GlobalResources.tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors='pt'
    ).to(DEVICE)
    
    with torch.no_grad():
        model_output = GlobalResources.model(**encoded_input)
        pooled = mean_pooling(model_output, encoded_input['attention_mask'])
        normed = l2_normalize(pooled)
    
    return normed.cpu().numpy()[0]


async def search_in_database(query: str, max_results: int):
    query_emb = generate_query_embedding(query).reshape(1, -1)
    similarities = cosine_similarity(query_emb, GlobalResources.embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][:max_results]
    
    results = []
    for idx in top_indices:
        place = GlobalResources.places[idx]
        category = categorize_place(place['title'], place['description'], place['tags'])
        
        results.append(PlaceResult(
            title=place['title'],
            description=place['description'],
            tags=place['tags'],
            transport=place['transport'],
            address=place['address'],
            price=place['price'],
            time=place['time'],
            similarity_score=float(similarities[idx]),
            category=category
        ))
    
    return results
def _extract_first_json_block(text: str) -> Optional[str]:
    """
    Извлекает первый корректный JSON-блок (массив или объект) из произвольного текста.
    Возвращает строку с JSON или None.
    """
    if not text or not isinstance(text, str):
        return None
    text = text.strip()

    # Пробуем сразу распарсить весь текст
    try:
        json.loads(text)
        return text
    except Exception:
        pass

    # Поиск сбалансированного блока [...] или {...}
    for open_c, close_c in (('[', ']'), ('{', '}')):
        start = text.find(open_c)
        if start == -1:
            continue
        stack = 0
        for i in range(start, len(text)):
            ch = text[i]
            if ch == open_c:
                stack += 1
            elif ch == close_c:
                stack -= 1
                if stack == 0:
                    candidate = text[start:i+1]
                    try:
                        json.loads(candidate)
                        return candidate
                    except Exception:
                        # если невалидный, ищем дальше
                        continue

    # Последняя попытка — найти любой блок, заключённый в [ ... ]
    m = re.search(r"(\[.*\])", text, flags=re.DOTALL)
    if m:
        candidate = m.group(1)
        try:
            json.loads(candidate)
            return candidate
        except Exception:
            pass

    return None

async def search_via_perplexity(query: str, max_results: int, timepad_events: Optional[List[dict]] = None) -> List[PlaceResult]:
    """
    Устойчивая версия вызова Perplexity.
    Возвращает список PlaceResult или пустой список при ошибке.
    """
    try:
        # --- подготовка краткого среза Timepad событий ---
        tp_text = ""
        if timepad_events:
            tp_snip = []
            for ev in timepad_events[:10]:
                # приводим все поля к строкам, чтобы не получить ошибки при сериализации / join
                ev_id = str(ev.get("id", "")) 
                ev_name = str(ev.get("name", "") or "")
                ev_starts = str(ev.get("starts_at", "") or "")
                ev_url = str(ev.get("url", "") or "")
                la = ev.get("location", {}) or {}
                ev_address = str(la.get("address") or la.get("city") or "")
                tp_snip.append({
                    "id": ev_id,
                    "name": ev_name,
                    "starts_at": ev_starts,
                    "url": ev_url,
                    "address": ev_address
                })
            # компактный JSON (без непредсказуемых пробелов)
            tp_text = json.dumps(tp_snip, ensure_ascii=False, separators=(",", ":"))

        # --- system prompt (короткий и жёсткий: только JSON-массив) ---
        system_content = (
            "Ты эксперт по туризму в Ростовской области. Отвечай ТОЛЬКО JSON-массивом объектов (ни слова текста). "
            "Каждый объект: {\"title\":\"...\",\"description\":\"...\",\"tags\":[...],"
            "\"transport\":\"...\",\"address\":\"...\",\"price\":ЧИСЛО,\"time\":ЧИСЛО,"
            "\"event_start\":\"ISO или null\",\"ticket_url\":\"...\"}. "
            f"Максимум {max_results} элементов."
        )

        messages = [
            {"role": "system", "content": str(system_content)}
        ]
        if tp_text:
            # передаём как строку
            messages.append({"role": "system", "content": f"Timepad_events: {tp_text}"})
        messages.append({"role": "user", "content": str(f"Найди интересные места в Ростове по запросу: {query}")})

        # call
        response = GlobalResources.perplexity_client.chat.completions.create(
            model="sonar",
            messages=messages
        )

        raw = response.choices[0].message.content
        if raw is None:
            print("Perplexity returned empty content")
            return []

        text = raw.strip()
        # логим первые 1200 символов для дебага
        print("Perplexity raw preview:", text[:1200])

        # сначала пытаемся парсить прям как JSON
        places_data = None
        try:
            places_data = json.loads(text)
        except Exception:
            # пытаемся извлечь первый корректный JSON-блок
            block = _extract_first_json_block(text)
            if block:
                try:
                    places_data = json.loads(block)
                except Exception as e:
                    print(f"Failed to parse extracted JSON block: {e}\nBlock preview: {block[:500]}")
                    places_data = None
            else:
                print("Не удалось найти JSON в ответе Perplexity.")
                places_data = None

        if places_data is None:
            return []

        # если dict — пытаемся взять внутренний список
        if isinstance(places_data, dict):
            for key in ("results", "places", "items"):
                if key in places_data and isinstance(places_data[key], list):
                    places_data = places_data[key]
                    break
            else:
                # обернём в список, если это единичный объект
                places_data = [places_data]

        if not isinstance(places_data, list):
            print("Perplexity returned JSON, но это не список. Тип:", type(places_data))
            return []

        results: List[PlaceResult] = []
        for i, place in enumerate(places_data[:max_results]):
            if not isinstance(place, dict):
                print(f"Пропускаю элемент #{i}: не объект (type={type(place)})")
                continue

            title = place.get("title") or place.get("name") or ""
            if not title:
                print(f"Пропускаю элемент #{i}: нет title/name. Поля: {list(place.keys())}")
                continue

            description = place.get("description", "") or ""
            tags = place.get("tags", []) or []
            transport = place.get("transport", "") or ""
            address = place.get("address", "") or ""

            # безопасное извлечение чисел
            price_raw = place.get("price", 0)
            try:
                if isinstance(price_raw, str):
                    m = re.search(r'\d+', price_raw)
                    price = int(m.group()) if m else 0
                else:
                    price = int(price_raw) if price_raw is not None else 0
            except Exception:
                price = 0

            time_raw = place.get("time", 0)
            try:
                if isinstance(time_raw, str):
                    m = re.search(r'\d+', time_raw)
                    time_val = int(m.group()) if m else 120
                else:
                    time_val = int(time_raw) if time_raw is not None else 120
            except Exception:
                time_val = 120

            event_start = place.get("event_start")
            ticket_url = place.get("ticket_url") or place.get("url") or None

            results.append(PlaceResult(
                title=str(title),
                description=str(description),
                tags=[str(t) for t in tags],
                transport=str(transport),
                address=str(address),
                price=price,
                time=time_val,
                similarity_score=1.0,
                category=categorize_place(str(title), str(description), [str(t) for t in tags]) if 'category' in PlaceResult.__fields__ else 1
            ))

        return results

    except Exception as e:
        print(f"Perplexity error (unexpected): {e}")
        return []

def create_error_response(best_similarity: float = 0.0) -> SearchResponse:
    """Создаёт стандартный ответ об ошибке"""
    return SearchResponse(
        source="error",
        results=[PlaceResult(
            title="Ошибка",
            description="Извините, мы не можем обработать такой запрос. Пожалуйста, обратитесь в поддержку.",
            tags=["ошибка"],
            transport="",
            address="",
            price=0,
            time=0,
            similarity_score=0.0,
            category=1
        )],
        best_similarity=best_similarity
    )


@app.on_event("startup")
def startup():
    print("Загрузка ресурсов...")
    
    GlobalResources.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    GlobalResources.model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
    GlobalResources.model.eval()
    
    with open(METADATA_FILE, 'r', encoding='utf-8') as f:
        GlobalResources.places = json.load(f)
    
    GlobalResources.embeddings = np.load(EMBEDDINGS_FILE)
    GlobalResources.perplexity_client = Perplexity(api_key=PERPLEXITY_API_KEY)
    
    print(f"Загружено {len(GlobalResources.places)} мест")
    print("Готово!")


@app.get("/")
def root():
    return {
        "status": "running",
        "api_title": "Smart Place Search API",
        "endpoints": {
            "docs": "/docs",
            "search": "/search (POST)"
        },
        "categories": CATEGORIES
    }


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest) -> SearchResponse:
    print(f"Запрос: {request.query}")
    
    db_results = await search_in_database(request.query, request.max_results)
    
    if not db_results:
        print("База пустая")
        return create_error_response()
    
    best_similarity = db_results[0].similarity_score
    print(f"Лучшая схожесть: {best_similarity:.4f}")
    
    if best_similarity < OFF_TOPIC_THRESHOLD:
        print(f"Запрос не по теме (схожесть {best_similarity:.4f} < {OFF_TOPIC_THRESHOLD})")
        return create_error_response(best_similarity)
    
    good_results = [r for r in db_results if r.similarity_score >= SIMILARITY_THRESHOLD]
    print(f"Результатов с score >= {SIMILARITY_THRESHOLD}: {len(good_results)}")
    
    if good_results:
        needed = request.max_results - len(good_results)
        
        if needed > 0:
            print(f"Найдено {len(good_results)} хороших из базы, добираем {needed} через Perplexity...")
            perplexity_results = await search_via_perplexity(request.query, needed)
            
            if needed > 0 and not perplexity_results and len(good_results) == 0:
                print("Perplexity определил запрос как нерелевантный")
                return create_error_response(best_similarity)
            
            final_results = good_results + perplexity_results
            return SearchResponse(
                source="mixed",
                results=final_results[:request.max_results],
                best_similarity=best_similarity
            )
        else:
            print(f"✓ Все результаты из базы")
            return SearchResponse(
                source="database",
                results=good_results[:request.max_results],
                best_similarity=best_similarity
            )
    
    print(f"Все результаты ниже порога, обращаемся к Perplexity...")
    perplexity_results = await search_via_perplexity(request.query, request.max_results)
    
    if perplexity_results:
        print(f"✓ Получены {len(perplexity_results)} результатов от Perplexity")
        return SearchResponse(
            source="perplexity",
            results=perplexity_results,
            best_similarity=best_similarity
        )
    
    print(f"Запрос не по теме (Perplexity не вернул результаты)")
    return create_error_response(best_similarity)
