from rag import prompt_with_rag_or_vector
from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/', methods=['GET'])
def landing_page():
    return render_template('index.html', title = 'RAG using Semantic Kernel with Azure OpenAI and Azure Cosmos DB for MongoDB vCore' )

@app.route('/', methods=['POST'])
async def get_response():
    query_term = request.form.get('text', None)
    rag_or_vector = request.form.get('option', None)
    response = await prompt_with_rag_or_vector(query_term, rag_or_vector, update_data=False)
    return render_template('result.html',
        title = 'RAG using Semantic Kernel with Azure OpenAI and Azure Cosmos DB for MongoDB vCore',
        question = query_term,
        response = response,
        rag_option = rag_or_vector
    )

if __name__ == '__main__':
    app.run(debug=True)