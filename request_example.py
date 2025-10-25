import requests

url = "http://127.0.0.1:5000/search"
data = {
    "query": "хочу поесть грузинской кухни",
    "max_results": 5
}

response = requests.post(url, json=data)
print(response.json())
