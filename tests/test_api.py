from fastapi.testclient import TestClient
from src.app import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200

def test_recommend():
    r = client.get("/recommend?user_id=1&n=3")
    assert r.status_code == 200

def test_batch():
    r = client.post("/predict_batch", json={
        "predictions": [{"user_id":1,"movie_id":1}]
    })
    assert r.status_code == 200