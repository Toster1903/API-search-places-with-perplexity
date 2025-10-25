import os
import json
import numpy as np
import torch
import re
from typing import List
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
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SIMILARITY_THRESHOLD = 0.6
OFF_TOPIC_THRESHOLD = 0.3  # Порог для определения нерелевантного запроса

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

class SearchResponse(BaseModel):
    source: str
    results: List[PlaceResult]
    best_similarity: float

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
        results.append(PlaceResult(
            title=place['title'],
            description=place['description'],
            tags=place['tags'],
            transport=place['transport'],
            address=place['address'],
            price=place['price'],
            time=place['time'],
            similarity_score=float(similarities[idx])
        ))
    
    return results

async def search_via_perplexity(query: str, max_results: int) -> List[PlaceResult]:
    try:
        response = GlobalResources.perplexity_client.chat.completions.create(
            model="sonar",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Ты эксперт по туризму в Ростовской области. "
                        "ВАЖНО: Отвечай ТОЛЬКО на запросы о туристических местах, достопримечательностях, кафе, ресторанах, парках. "
                        "Если запрос не связан с туризмом - верни пустой массив []. "
                        "Формат JSON массива: [{\"title\": \"...\", \"description\": \"...\", "
                        "\"tags\": [...], \"transport\": \"...\", \"address\": \"...\", "
                        "\"price\": ЧИСЛО (только цифры в рублях), \"time\": ЧИСЛО (только цифры в минутах)}]. "
                        f"Верни максимум {max_results} мест. "
                        "price и time ОБЯЗАТЕЛЬНО должны быть целыми числами без текста."
                    )
                },
                {
                    "role": "user",
                    "content": f"Найди интересные места в Ростове по запросу: {query}"
                }
            ]
        )
        
        text = response.choices[0].message.content.strip()
        places_data = json.loads(text)
        
        if not places_data:
            return []
        
        results = []
        for place in places_data[:max_results]:
            price_raw = place.get('price', 0)
            if isinstance(price_raw, str):
                price_match = re.search(r'\d+', price_raw)
                price = int(price_match.group()) if price_match else 0
            else:
                price = int(price_raw) if price_raw else 0
            
            time_raw = place.get('time', 0)
            if isinstance(time_raw, str):
                time_match = re.search(r'\d+', time_raw)
                time = int(time_match.group()) if time_match else 120
            else:
                time = int(time_raw) if time_raw else 120
            
            results.append(PlaceResult(
                title=place.get('title', ''),
                description=place.get('description', ''),
                tags=place.get('tags', []),
                transport=place.get('transport', ''),
                address=place.get('address', ''),
                price=price,
                time=time,
                similarity_score=1.0
            ))
        
        return results
        
    except Exception as e:
        print(f"Perplexity error: {e}")
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
            similarity_score=0.0
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
        }
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
    
    # Проверка на нерелевантный запрос (не по теме туризма)
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
