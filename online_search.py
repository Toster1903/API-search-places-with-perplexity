import os
import json
from fastapi import FastAPI
from pydantic import BaseModel
from perplexity import Perplexity
from dotenv import load_dotenv


load_dotenv()


PERPLEXITY_API_KEY = os.getenv("PPLX_API_KEY")

app = FastAPI(title="Place Search API", version="1.1")


class PlaceInfo(BaseModel):
    title: str
    description: str
    tags: list[str]
    transport: str
    address: str
    price: int
    time: int


class PlaceRequest(BaseModel):
    query: str


class PlaceResponse(BaseModel):
    place_info: PlaceInfo


perplexity_client = None


@app.on_event("startup")
def startup():
    global perplexity_client
    perplexity_client = Perplexity(api_key=PERPLEXITY_API_KEY)
    print("Perplexity client initialized.")


@app.get("/")
def root():
    return {
        "status": "running",
        "api_title": "Place Search API",
        "endpoint": "/search-place (POST)"
    }


@app.post("/search-place", response_model=PlaceResponse)
async def search_place(request: PlaceRequest):
    try:
        response = perplexity_client.chat.completions.create(
            model="sonar",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Ты — ассистент, структурирующий запросы для поиска мест в Ростовской области. "
                        "Формат JSON-ответа: {"
                        "\"title\": \"название места\", "
                        "\"description\": \"краткое описание\", "
                        "\"tags\": [\"тег1\", \"тег2\"], "
                        "\"transport\": \"типы транспорта\", "
                        "\"address\": \"полный адрес в Ростовской области\", "
                        "\"price\": примерная стоимость посещения в рублях (0 если бесплатно), "
                        "\"time\": примерное время на посещение в минутах}"
                        "Отвечай строго только валидным JSON без дополнительных пояснений."
                    )
                },
                {
                    "role": "user",
                    "content": request.query
                }
            ]
        )
        
        normalized_text = response.choices[0].message.content.strip()
        print(f"Perplexity response: {normalized_text}")
        
        place_data = json.loads(normalized_text)
        
        place_info = PlaceInfo(
            title=place_data.get("title", ""),
            description=place_data.get("description", ""),
            tags=place_data.get("tags", []),
            transport=place_data.get("transport", ""),
            address=place_data.get("address", "Адрес не найден"),
            price=place_data.get("price", 0),
            time=place_data.get("time", 0)
        )
        
        return PlaceResponse(place_info=place_info)
        
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        return PlaceResponse(
            place_info=PlaceInfo(
                title="Ошибка",
                description="Некорректный JSON от API",
                tags=[],
                transport="",
                address="Ошибка парсинга",
                price=0,
                time=0
            )
        )
    except Exception as e:
        print(f"Error: {e}")
        return PlaceResponse(
            place_info=PlaceInfo(
                title="Ошибка",
                description=f"Ошибка: {str(e)}",
                tags=[],
                transport="",
                address="",
                price=0,
                time=0
            )
        )
