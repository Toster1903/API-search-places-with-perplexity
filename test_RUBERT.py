import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

# Пути к файлам
METADATA_FILE = "places_meta.json"
EMBEDDINGS_FILE = "embeddings.npy"
#MODEL_NAME = "sberbank-ai/sbert_large_nlu_ru"
MODEL_NAME = "cointegrated/rubert-tiny" 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Загрузка модели
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME, local_files_only=False).to(DEVICE)
model.eval()

# Загрузка данных
with open(METADATA_FILE, 'r', encoding='utf-8') as f:
    places = json.load(f)

embeddings = np.load(EMBEDDINGS_FILE)

print(f"Загружено {len(places)} мест")
print(f"Размер эмбеддингов: {embeddings.shape}\n")


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def l2_normalize(embeddings):
    return torch.nn.functional.normalize(embeddings, p=2, dim=1)


def generate_query_embedding(text):
    encoded_input = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors='pt'
    ).to(DEVICE)
    
    with torch.no_grad():
        model_output = model(**encoded_input)
        pooled = mean_pooling(model_output, encoded_input['attention_mask'])
        normed = l2_normalize(pooled)
    
    return normed.cpu().numpy()[0]


def search_places(query, top_k=5):
    print(f"Запрос: '{query}'\n")
    
    # Генерация эмбеддинга запроса
    query_emb = generate_query_embedding(query)
    query_emb = query_emb.reshape(1, -1)
    
    # Вычисление схожести
    similarities = cosine_similarity(query_emb, embeddings)[0]
    
    # Сортировка по убыванию
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    print(f"Топ-{top_k} результатов:\n")
    for i, idx in enumerate(top_indices, 1):
        place = places[idx]
        score = similarities[idx]
        
        print(f"{i}. {place['title']}")
        print(f"   Описание: {place['description']}")
        print(f"   Теги: {', '.join(place['tags'])}")
        print(f"   Адрес: {place['address']}")
        print(f"   Цена: {place['price']} руб, Время: {place['time']} мин")
        print(f"   Схожесть: {score:.4f}\n")


def main():
    print("=" * 60)
    print("ТЕСТ ПОИСКА МЕСТ ПО ЗАПРОСУ")
    print("=" * 60)
    print()
    
    while True:
        user_query = input("Введите запрос (или 'exit' для выхода): ").strip()
        
        if user_query.lower() == 'exit':
            print("Выход...")
            break
        
        if not user_query:
            print("Пустой запрос, попробуйте снова\n")
            continue
        
        print()
        search_places(user_query)
        print("-" * 60)
        print()


if __name__ == "__main__":
    main()
