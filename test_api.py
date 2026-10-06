from fastapi.testclient import TestClient
from api import app

client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_valid_input():
    response = client.post(
        "/predict",
        json={
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }
    )

    assert response.status_code == 200

    assert response.json()["predicted_class"] in [
        "setosa",
        "versicolor",
        "virginica"
    ]


def test_predict_missing_field_returns_422():
    response = client.post(
        "/predict",
        json={
            "sepal_length": 5.1
        }
    )

    assert response.status_code == 422