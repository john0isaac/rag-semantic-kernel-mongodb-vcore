import logging
from quart import Quart, render_template, request, current_app, jsonify
from rag import (
    initialize_sk_chat_embedding,
    initialize_sk_memory_store,
    perform_rag_search,
    perform_vector_search,
    grounded_response,
)

# logging.basicConfig(level=logging.DEBUG)

app = Quart(__name__)


@app.before_serving
async def initialize_sk():
    app.sk_kernel = initialize_sk_chat_embedding()
    app.sk_memory, app.sk_store = await initialize_sk_memory_store(app.sk_kernel)
    app.sk_function = await grounded_response(app.sk_kernel)

    current_app.logger.info("Serving the app...")


@app.route("/", methods=["GET"])
async def landing_page():
    return await render_template(
        "index.html",
        title="RAG using Semantic Kernel with Azure OpenAI and Azure Cosmos DB for MongoDB vCore",
    )


@app.route("/chat", methods=["POST"])
async def chat_handler():
    body = await request.get_json()
    query_term = body.get("message", "Blank")
    rag_or_vector = body.get("option", "rag")

    try:
        if rag_or_vector == "rag":
            response = await perform_rag_search(
                app.sk_kernel, app.sk_memory, app.sk_function, query_term
            )
        elif rag_or_vector == "vector":
            response = await perform_vector_search(app.sk_memory, query_term)
            response = response[0].text
        return jsonify({"answer": str(response)})
    except ValueError as e:
        logging.exception("Exception in %s: %s", "/chat", e)
        return (
            jsonify(
                {
                    "error": "Invalid option. Please choose either 'rag' or 'only-vector'."
                }
            ),
            400,
        )


# Or specify port manually:
"""
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
"""
