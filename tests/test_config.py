import pytest

from quartapp import create_app


def test_config():
    assert not create_app().testing
    assert create_app({"TESTING": True}).testing


@pytest.mark.asyncio
async def test_hello(client):
    response = await client.get("/hello")
    data = await response.get_json()

    assert response.status_code == 200
    assert "Hello, World!" in data["answer"]
