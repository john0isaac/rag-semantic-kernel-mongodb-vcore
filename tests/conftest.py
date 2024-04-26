import pytest
import pytest_asyncio

from quartapp import create_app


@pytest.fixture
def app():
    app = create_app()
    app.config.update(
        {
            "TESTING": True,
        }
    )

    return app.test_app()


@pytest_asyncio.fixture
async def client(app):
    yield app.test_client()
