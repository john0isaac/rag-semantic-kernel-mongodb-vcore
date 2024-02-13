# RAG using Semantic Kernel with Azure OpenAI and Azure Cosmos DB for MongoDB vCore

A sample for implementing retrieval augmented generation using Azure Open AI to generate embeddings, Azure Cosmos DB for MongoDB vCore to perform vector search, and semantic kernel.

📝 Start here 👉 [rag-azure-openai-cosmosdb-notebook.ipynb](./rag-azure-openai-cosmosdb-notebook.ipynb)

Test it inside codespaces 👇

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/john0isaac/rag-semantic-kernel-mongodb-vcore?devcontainer_path=.devcontainer/devcontainer.json)

## Running the web app

To run the flask application, follow these steps:

1. **Download the project starter code locally**

    ```bash
    git clone https://github.com/john0isaac/rag-semantic-kernel-mongodb-vcore.git
    cd rag-semantic-kernel-mongodb-vcore
    ```

1. **Install, initialize and activate a virtualenv using:**

    ```bash
    pip install virtualenv
    python -m virtualenv venv
    source venv/bin/activate
    ```

    >**Note** - In Windows, the `venv` does not have a `bin` directory. Therefore, you'd use the analogous command shown below:

    ```bash
    source venv\Scripts\activate
    ```

1. **Install the dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

1. **Run the [notebook](./rag-azure-openai-cosmosdb-notebook.ipynb) to generate the .env file and test out everything first**

1. **Execute the following command in your terminal to start the flask app**

    ```bash
    export FLASK_APP=app.py
    export FLASK_ENV=development
    export FLASK_DEBUG=true
    flask run --reload
    ```

    **For Windows, use [`setx`](https://learn.microsoft.com/windows-server/administration/windows-commands/setx) command shown below:**

   ```powershell
    setx FLASK_APP app.py
    setx FLASK_ENV development
    setx FLASK_DEBUG true
    flask run --reload
    ```

1. **Verify on the Browser**

Navigate to project homepage [http://127.0.0.1:5000/](http://127.0.0.1:5000/) or [http://localhost:5000](http://localhost:5000)
