"""
NeuroLens API Tests
"""

import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_root():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "NeuroLens AI"


@pytest.mark.anyio
async def test_health():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"


@pytest.mark.anyio
async def test_analyze():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/analyze",
            json={
                "text": "I honestly believe this is the truth, trust me on this.",
                "explain": True,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "deception" in data
        assert "emotions" in data
        assert "manipulation" in data
        assert "confidence_score" in data


@pytest.mark.anyio
async def test_batch_analyze():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/batch-analyze",
            json={
                "texts": [
                    "I am very happy today!",
                    "You should be ashamed of yourself.",
                ],
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 2


@pytest.mark.anyio
async def test_metrics():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "model" in data
        assert "cache" in data
