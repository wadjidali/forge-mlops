from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_home_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "online"

def test_predict_endpoint():
    payload = {"text": "This product is absolutely amazing!"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    json_data = response.json()
    assert "prediction" in json_data
    assert "sentiment" in json_data