# RAG using Semantic Kernel with Azure OpenAI and Azure Cosmos DB for MongoDB vCore

A sample for implementing retrieval augmented generation using Azure Open AI to generate embeddings, Azure Cosmos DB for MongoDB vCore to perform vector search, and semantic kernel.

## How to use?

1. Create the following resources on Microsoft Azure:

    - Azure Cosmos DB for MongoDB vCore cluster. See the [Quick Start guide here](https://learn.microsoft.com/azure/cosmos-db/mongodb/vcore/quickstart-portal).
    - Azure OpenAI resource with:
        - Embedding model deployment. (ex. `text-embedding-ada-002`) See the [guide here](https://learn.microsoft.com/azure/ai-services/openai/how-to/create-resource?pivots=web-portal).
        - Chat model deployment. (ex. `gpt-35-turbo`)

1. 📝 Start here 👉 [rag-azure-openai-cosmosdb-notebook.ipynb](./rag-azure-openai-cosmosdb-notebook.ipynb)


https://github.com/john0isaac/rag-semantic-kernel-mongodb-vcore/assets/64026625/676a0e10-876f-45e6-942d-0494ac327c75


Test it inside codespaces 👇

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/john0isaac/rag-semantic-kernel-mongodb-vcore?devcontainer_path=.devcontainer/devcontainer.json)

## Running the web app

To run the Quart application, follow these steps:

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
    pip install -r requirements-dev.txt
    ```

1. **Run the [notebook](./rag-azure-openai-cosmosdb-notebook.ipynb) to generate the .env file and test out everything first**

1. **Install the app as an editable package:**

    ```bash
    pip install -e src
    ```

1. **Execute the following command in your terminal to start the quart app**

    ```bash
    export QUART_APP=src.quartapp
    export QUART_ENV=development
    export QUART_DEBUG=true
    quart run --reload
    ```

    **For Windows, use [`setx`](https://learn.microsoft.com/windows-server/administration/windows-commands/setx) command shown below:**

   ```powershell
    setx QUART_APP src.quartapp
    setx QUART_ENV development
    setx QUART_DEBUG true
    quart run --reload
    ```

1. **Verify on the Browser**

Navigate to project homepage [http://127.0.0.1:5000/](http://127.0.0.1:5000/) or [http://localhost:5000](http://localhost:5000)


https://github.com/john0isaac/rag-semantic-kernel-mongodb-vcore/assets/64026625/8a7556d6-2b54-40b5-825b-06d6efd4d1ca

## Deployment

![architecture-thumbnail](https://github.com/john0isaac/rag-semantic-kernel-mongodb-vcore/assets/64026625/7ce8ae6e-0424-47ce-b0a3-9be072eda27f)

This repository is set up for deployment on Azure App Service (w/Azure Cosmos DB for MongoDB vCore) using the configuration files in the `infra` folder.

To deploy your own instance, follow these steps:

1. Sign up for a [free Azure account](https://azure.microsoft.com/free/)

1. Install the [Azure Dev CLI](https://learn.microsoft.com/azure/developer/azure-developer-cli/install-azd).

1. Login to your Azure account:

    ```shell
    azd auth login
    ```

1. Initialize a new `azd` environment:

    ```shell
    azd init
    ```

    It will prompt you to provide a name (like "quart-app") that will later be used in the name of the deployed resources.

1. Provision and deploy all the resources:

    ```shell
    azd up
    ```

    It will prompt you to login, pick a subscription, and provide a location (like "eastus"). Then it will provision the resources in your account and deploy the latest code. If you get an error with deployment, changing the location (like to "centralus") can help, as there may be availability constraints for some of the resources.

When azd has finished deploying, you'll see an endpoint URI in the command output. Visit that URI to browse the app! 🎉

> [!NOTE]
> If you make any changes to the app code, you can just run this command to redeploy it:
>
> ```shell
> azd deploy
> ```
>

### Add the Data

1. Open the [Azure portal](https://portal.azure.com) and sign in.

1. Navigate to your App Service page.

    ![Azure App service screenshot with the word SSH highlighted in a red box.](https://github.com/john0isaac/rag-semantic-kernel-mongodb-vcore/assets/64026625/759db6be-604e-433c-878e-b6c3de671fd1)

1. Select **SSH** from the left menu then, select **Go**.

1. In the SSH terminal, run `python ./scripts/add_data.py`.

## Add your Own Data

The Python scrips that adds the data is configured to accept any JSON file with your data but you need to specify the following parameters when you run it:

- Data file path: Path to the JSON file that contains your data. `--file="./data/text-sample.json"` or `-f "./data/text-sample.json"`
- ID field: This is the name of the field that cosmos uses to identify your database records. `--id-field=id` or `-id id`
- Text field: This is the name of the field that will be used to generate the vector embeddings from and stored in the database. `--text-field=content` or `-txt content`
- Description field: This is the name of the description field that cosmos will store along with the embeddings. `--description-field=title` or `-desc title`

    ```bash
    python ./scripts/add_data.py --file="./data/text-sample.json" --id-field=id --text-field=contnet --description-field=title
    ```

### Example for Step by step Manual Deployment

1. Add your JSON data to the [data folder](./src/data/).

1. The workflow will trigger automatically and push your data to the Azure App service.

1. Open the [Azure portal](https://portal.azure.com) and sign in.

1. Navigate to your App Service page.

    ![Azure App service screenshot with the word SSH highlighted in a red box.](https://github.com/john0isaac/rag-semantic-kernel-mongodb-vcore/assets/64026625/759db6be-604e-433c-878e-b6c3de671fd1)

1. Select **SSH** from the left menu then, select **Go**.

1. In the SSH terminal, run the following command with the changed values to suit your data:

    ```bash
    python ./scripts/add_data.py --file="./data/text-sample.json" --id-field=id --text-field=contnet --description-field=title
    ```

### Example for azd Deployment

1. Add your JSON data to the [data folder](./src/data/).

1. Run `azd deploy` to upload the data to Azure App Service.

1. Open the [Azure portal](https://portal.azure.com) and sign in.

1. Navigate to your App Service page.

    ![Azure App service screenshot with the word SSH highlighted in a red box.](https://github.com/john0isaac/rag-semantic-kernel-mongodb-vcore/assets/64026625/759db6be-604e-433c-878e-b6c3de671fd1)

1. Select **SSH** from the left menu then, select **Go**.

1. In the SSH terminal, run the following command with the changed values to suit your data:

    ```bash
    python ./scripts/add_data.py --file="./data/text-sample.json" --id-field=id --text-field=contnet --description-field=title
    ```
