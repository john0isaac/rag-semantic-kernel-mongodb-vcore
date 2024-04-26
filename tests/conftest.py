import pytest
import pytest_asyncio

from quartapp import create_app


@pytest_asyncio.fixture
async def app():
    app = create_app()
    app.config.update(
        {
            "TESTING": True,
        }
    )
    async with app.test_app() as test_app:
        yield test_app


@pytest.fixture
def client(app):
    return app.test_client()
