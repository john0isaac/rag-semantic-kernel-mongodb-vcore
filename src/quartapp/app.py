import logging
from typing import Any

from quart import Quart, current_app, jsonify, render_template, request

from quartapp.rag import (
    FunctionResult,
    Kernel,
    KernelFunction,
    MemoryQueryResult,
    SemanticTextMemory,
    grounded_response,
    initialize_sk_chat_embedding,
    initialize_sk_memory_store,
    perform_rag_search,
    perform_vector_search,
)


class SKernel:
    def __init__(self, sk_kernel: Kernel, sk_memory: SemanticTextMemory, sk_function: KernelFunction) -> None:
        self.sk_kernel = sk_kernel
        self.sk_memory = sk_memory
        self.sk_function = sk_function


class CustomQuart(Quart, SKernel):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)


def create_app(test_config=None) -> CustomQuart:
    app = CustomQuart(__name__, template_folder="../frontend", static_folder="../frontend/static")

    if test_config:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    @app.before_serving
    async def initialize_sk() -> None:
        app.sk_kernel = initialize_sk_chat_embedding()
        app.sk_memory, _ = await initialize_sk_memory_store(app.sk_kernel)
        app.sk_function = await grounded_response(app.sk_kernel)

        current_app.logger.info("Serving the app...")

    @app.route("/hello", methods=["GET"])
    async def hello() -> Any:
        return jsonify({"answer": "Hello, World!"})

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
                rag_response: FunctionResult | None = await perform_rag_search(
                    app.sk_kernel, app.sk_memory, app.sk_function, query_term
                )
                return jsonify({"answer": str(rag_response)})
            elif rag_or_vector == "vector":
                vector_response: list[MemoryQueryResult] = await perform_vector_search(app.sk_memory, query_term)
                return jsonify({"answer": str(vector_response[0].text)})
            else:
                return (
                    jsonify({"answer": "Invalid option. Please choose either 'rag' or 'only-vector'."}),
                    400,
                )

        except ValueError as e:
            logging.error(f"Error: {e}")
            return jsonify({"answer": f"Error: {e}"}), 400

    return app


# Or specify port manually:
"""
if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
"""
