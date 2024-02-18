from quart import (
    Quart,
    render_template,
    request,
    current_app
)
from rag import (
    initialize_sk_chat_embedding,
    initialize_sk_memory_store,
    perform_rag_search,
    perform_vector_search,
    upsert_data_to_memory_store
)
import urllib.parse

app = Quart(__name__)

@app.before_serving
async def intialize_sk():
    app.sk_kernel = initialize_sk_chat_embedding()
    app.sk_store = await initialize_sk_memory_store(app.sk_kernel)

    # Load the new data into the memory store
    # Must be used the first time you run the app to populate the database
    # For faster startup, you can set the first_run variable to False to skip it
    first_run = False
    if first_run:
        try:
            await upsert_data_to_memory_store(app.sk_kernel.memory, app.sk_store, "text-sample.json")
        except TimeoutError:
            await upsert_data_to_memory_store(app.sk_kernel.memory, app.sk_store, "text-sample.json")

    current_app.logger.info("Serving the app...")

@app.route('/', methods=['GET'])
async def landing_page():
    return await render_template(
        'index.html',
        title = 'RAG using Semantic Kernel with Azure OpenAI and Azure Cosmos DB for MongoDB vCore'
    )

@app.route('/', methods=['POST'])
async def get_response():
    body = await request.get_data(as_text=True)
    parsed_dict = urllib.parse.parse_qs(body)
    query_term = parsed_dict.get('text', ['Blank'])[0]
    rag_or_vector = parsed_dict.get('option', ['option'])[0]

    try:
        if rag_or_vector == "rag":
            response = await perform_rag_search(app.sk_kernel, query_term)
        elif rag_or_vector == "vector":
            response = await perform_vector_search(app.sk_kernel.memory, query_term)
            response = response[0].text
    except ValueError as e:
        raise ValueError("Invalid option. Please choose either 'rag' or 'only-vector'.") from e

    return await render_template('result.html',
        title = 'RAG using Semantic Kernel with Azure OpenAI and Azure Cosmos DB for MongoDB vCore',
        question = query_term,
        response = response,
        rag_option = rag_or_vector
    )

# Default port:
if __name__ == '__main__':
    app.run()

# Or specify port manually:
"""
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
"""