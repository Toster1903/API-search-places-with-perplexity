import os
import json
import re
import asyncio
from typing import List, Optional

import numpy as np
import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field, AliasChoices
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from perplexity import Perplexity
from dotenv import load_dotenv

load_dotenv()

PERPLEXITY_API_KEY = os.getenv("PPLX_API_KEY")
MODEL_NAME = "cointegrated/rubert-tiny"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CATEGORIES = {
    0: "Культурно-развлекательный отдых",
    1: "Гастрономический и социальный досуг",
    2: "Активный и спортивный отдых",
    3: "Природный и оздоровительный отдых",
    4: "Развлечения и аттракционы",
    5: "Коммерческие мероприятия",
}

app = FastAPI(title="Smart Place Search API", version="2.0")

class GlobalResources:
    tokenizer = None
    model = None
    perplexity_client = None

class PlaceData(BaseModel):
    title: str
    description: Optional[str] = None
    price: Optional[int] = None
    time: Optional[int] = None
    address: Optional[str] = None

class SearchRequest(BaseModel):
    user_query: str = Field(validation_alias=AliasChoices("user_query", "query"))
    places: List[PlaceData]
    max_results: int = 5

class PlaceResultWithScore(BaseModel):
    title: str
    description: str
    price: int
    time: int
    similarity_score: float

class SearchResponse(BaseModel):
    results: List[PlaceResultWithScore]
    best_similarity: float
    worst_similarity: float

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )

def l2_normalize(embeddings):
    return torch.nn.functional.normalize(embeddings, p=2, dim=1)

def generate_embedding(text: str) -> np.ndarray:
    encoded_input = GlobalResources.tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    ).to(DEVICE)
    with torch.no_grad():
        model_output = GlobalResources.model(**encoded_input)
        pooled = mean_pooling(model_output, encoded_input["attention_mask"])
        normed = l2_normalize(pooled)
    return normed.cpu().numpy()[0]

def _to_int(v, default: int) -> int:
    if v is None:
        return default
    try:
        if isinstance(v, str):
            m = re.search(r"\d+", v)
            return int(m.group()) if m else default
        return int(v)
    except Exception:
        return default

def normalize_place_with_original(llm_obj: dict, original_title: str) -> dict:
    return {
        "title": original_title,
        "description": str((llm_obj or {}).get("description") or ""),
        "price": _to_int((llm_obj or {}).get("price"), 0),
        "time": _to_int((llm_obj or {}).get("time"), 120),
        "address": str((llm_obj or {}).get("address") or ""),
        "tags": [str(t) for t in ((llm_obj or {}).get("tags") or [])],
        "transport": str((llm_obj or {}).get("transport") or ""),
    }

def build_perplexity_messages_for_place(place_title: str, user_query: str) -> list:
    system_content = (
        'Ты эксперт по туризму. Для указанного места верни ТОЛЬКО JSON-объект '
        'с ключами: {"title":"...","description":"...","price":ЧИСЛО,"time":ЧИСЛО,} без лишнего текста.'
    )
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": f"Место: {place_title}\nКонтекст запроса пользователя: {user_query}"},
    ]

async def fetch_one_place_detail(place_title: str, user_query: str, semaphore: asyncio.Semaphore) -> dict:
    messages = build_perplexity_messages_for_place(place_title, user_query)
    async with semaphore:
        def _call_sync():
            resp = GlobalResources.perplexity_client.chat.completions.create(
                model="sonar", messages=messages
            )
            return (resp.choices[0].message.content or "").strip()
        text = await asyncio.to_thread(_call_sync)
    data = None
    try:
        data = json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(0))
            except Exception:
                data = None
    if isinstance(data, list) and data:
        data = data[0]
    if not isinstance(data, dict):
        data = {}
    return normalize_place_with_original(data, original_title=place_title)

async def enrich_places_with_perplexity(user_query: str, titles: List[str], max_concurrency: int = 4) -> List[dict]:
    sem = asyncio.Semaphore(max_concurrency)
    tasks = [fetch_one_place_detail(t, user_query, sem) for t in titles]
    results = await asyncio.gather(*tasks, return_exceptions=False)
    uniq = {}
    for item in results:
        if item and item.get("title"):
            uniq[item["title"]] = item
    return list(uniq.values())

async def rank_places_by_relevance(user_query: str, places: List[dict]) -> List[PlaceResultWithScore]:
    if not places:
        return []
    query_embedding = generate_embedding(user_query).reshape(1, -1)
    results_with_scores: List[PlaceResultWithScore] = []
    for place in places:
        safe = {
            "title": place["title"],
            "description": place.get("description", ""),
            "price": place.get("price", 0),
            "time": place.get("time", 120),
            "address": place.get("address", ""),
            "tags": place.get("tags", []),
            "transport": place.get("transport", ""),
        }
        place_text = f"{safe['title']} {safe['description']} {' '.join(safe['tags'])}"
        place_embedding = generate_embedding(place_text).reshape(1, -1)
        similarity = cosine_similarity(query_embedding, place_embedding)[0][0]
        results_with_scores.append(
            PlaceResultWithScore(
                title=safe["title"],
                description=safe["description"],
                price=safe["price"],
                time=safe["time"],
                similarity_score=float(similarity),
            )
        )
    results_with_scores.sort(key=lambda x: x.similarity_score, reverse=True)
    return results_with_scores

@app.on_event("startup")
def startup():
    GlobalResources.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    GlobalResources.model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
    GlobalResources.model.eval()
    GlobalResources.perplexity_client = Perplexity(api_key=PERPLEXITY_API_KEY)

@app.get("/")
def root():
    return {
        "status": "running",
        "api_title": "Smart Place Search API",
        "endpoints": {"docs": "/docs", "search_v2": "/search_v2 (POST)"},
        "categories": CATEGORIES,
    }

@app.post("/search_v2", response_model=SearchResponse)
async def search(request: SearchRequest) -> SearchResponse:
    print(f"\n📍 Запрос: {request.user_query}")
    print(f"📊 Получено мест (названий): {len(request.places)}")
    titles = [p.title for p in request.places if p.title]
    enriched_places = await enrich_places_with_perplexity(
        user_query=request.user_query, titles=titles, max_concurrency=4
    )
    print(f"✓ Обогащено мест: {len(enriched_places)}")
    if not enriched_places:
        return SearchResponse(results=[], best_similarity=0.0, worst_similarity=0.0)
    ranked_results = await rank_places_by_relevance(request.user_query, enriched_places)
    best_score = ranked_results[0].similarity_score if ranked_results else 0.0
    worst_score = ranked_results[-1].similarity_score if ranked_results else 0.0
    return SearchResponse(
        results=ranked_results[: request.max_results],
        best_similarity=best_score,
        worst_similarity=worst_score,
    )
