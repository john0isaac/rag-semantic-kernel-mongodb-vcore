import logging
from typing import Any

from quart import Quart, current_app, jsonify, render_template, request

from quartapp.rag import (
    grounded_response,
    initialize_sk_chat_embedding,
    initialize_sk_memory_store,
    perform_rag_search,
    perform_vector_search,
)

# logging.basicConfig(level=logging.DEBUG)


class CustomQuart(Quart):  # types: ignore
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.sk_kernel = None
        self.sk_memory = None
        self.sk_function = None


def create_app() -> CustomQuart:
    app = CustomQuart(__name__, template_folder="../frontend", static_folder="../frontend/static")

    @app.before_serving
    async def initialize_sk() -> None:
        app.sk_kernel = initialize_sk_chat_embedding()
        app.sk_memory, _ = await initialize_sk_memory_store(app.sk_kernel)
        app.sk_function = await grounded_response(app.sk_kernel)

        current_app.logger.info("Serving the app...")

    @app.route("/", methods=["GET"])
    async def landing_page() -> Any:
        return await render_template(
            "index.html",
            title="RAG using Semantic Kernel with Azure OpenAI and Azure Cosmos DB for MongoDB vCore",
        )

    @app.route("/chat", methods=["POST"])
    async def chat_handler() -> Any:
        body = await request.get_json()
        query_term = body.get("message", "Blank")
        rag_or_vector = body.get("option", "rag")

        try:
            if rag_or_vector == "rag":
                rag_response: str = await perform_rag_search(app.sk_kernel, app.sk_memory, app.sk_function, query_term)
                return jsonify({"answer": str(rag_response)})
            elif rag_or_vector == "vector":
                vector_response = await perform_vector_search(app.sk_memory, query_term)
                vector_response = vector_response[0].text if vector_response else "Not found!"
                return jsonify({"answer": str(vector_response)})
        except ValueError as e:
            logging.exception("Exception in %s: %s", "/chat", e)
            return (
                jsonify({"error": "Invalid option. Please choose either 'rag' or 'only-vector'."}),
                400,
            )

    return app


# Or specify port manually:
"""
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
"""
