import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# Пути к файлам
INPUT_JSON = "rostov_all_places_full.json"
OUTPUT_META = "places_meta.json"
OUTPUT_EMBEDDINGS = "embeddings.npy"

# Модель ruBERT
MODEL_NAME = "cointegrated/rubert-tiny"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Загрузка модели и токенизатора
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def l2_normalize(embeddings):
    return torch.nn.functional.normalize(embeddings, p=2, dim=1)

def generate_embedding(text):
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

def create_text_for_embedding(place_info):
    text_parts = [
        place_info.get('title', ''),
        place_info.get('description', ''),
        " ".join(place_info.get('tags', [])),
        place_info.get('address', '')
    ]
    return " ".join([p for p in text_parts if p])

def main():
    with open(INPUT_JSON, encoding='utf-8') as f:
        data = json.load(f)

    places = [item['place_info'] for item in data]

    embeddings = []
    for place in tqdm(places, desc="Генерация эмбеддингов"):
        text = create_text_for_embedding(place)
        emb = generate_embedding(text)
        embeddings.append(emb)

    embeddings = np.array(embeddings)

    with open(OUTPUT_META, 'w', encoding='utf-8') as f:
        json.dump(places, f, ensure_ascii=False, indent=2)

    np.save(OUTPUT_EMBEDDINGS, embeddings)

    print(f"Сохранено {len(places)} мест в '{OUTPUT_META}'")
    print(f"Сохранены эмбеддинги размером {embeddings.shape} в '{OUTPUT_EMBEDDINGS}'")

if __name__ == "__main__":
    main()
