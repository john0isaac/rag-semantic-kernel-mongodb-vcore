import pytest


@pytest.mark.asyncio
async def test_landing_page(client):
    response = await client.get("/")
    body = await response.get_data()

    assert response.status_code == 200
    assert b"RAG using Semantic Kernel with Azure OpenAI and Azure Cosmos DB for MongoDB vCore" in body


@pytest.mark.asyncio
async def test_chat_handler_400(client):
    response = await client.post("/chat", json={"bad_request": "bad_request", "option": "bad_option"})
    body = await response.get_json()

    assert response.status_code == 400
    assert "Invalid option" in body["answer"]


@pytest.mark.asyncio
async def test_chat_handler_vector(client):
    response = await client.post("/chat", json={"message": "Hello", "option": "vector"})
    body = await response.get_json()

    assert response.status_code == 200
    assert "Not found!" in body["answer"]


@pytest.mark.asyncio
async def test_chat_handler_rag(client):
    response = await client.post("/chat", json={"message": "Hello", "option": "rag"})
    body = await response.get_json()

    assert response.status_code == 200
    assert "The requested data is not Found." in body["answer"]
