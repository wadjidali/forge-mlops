import requests

# URL de l'API exécutée dans Docker
url = "http://127.0.0.1:8000/predict"

payloads = [
    {"text": "This product is fantastic, I love it!"},
    {"text": "This gadget is terrible and completely useless."},
    {"text": "It is okay, nothing special."},
]

print("🚀 Envoi des requêtes de test à l'API Docker...\n")

for payload in payloads:
  try:
    response = requests.post(url, json=payload)
    print("Statut HTTP :", response.status_code)
    print("Réponse API :", response.json())
    print("-" * 50)
  except Exception as e:
    print(f"❌ Impossible de contacter l'API : {e}")