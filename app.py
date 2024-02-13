from quart import Quart, render_template, request
from rag import prompt_with_rag_or_vector
import urllib.parse

app = Quart(__name__)

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

    response = await prompt_with_rag_or_vector(
        query_term,
        rag_or_vector,
        update_data=False
    )
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