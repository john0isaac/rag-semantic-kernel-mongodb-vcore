import logging
import os
from urllib.parse import quote_plus

from pymongo.errors import ServerSelectionTimeoutError
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import (  # type: ignore [import-untyped]
    AzureChatCompletion,
    AzureTextEmbedding,
    OpenAITextPromptExecutionSettings,
)
from semantic_kernel.connectors.memory.azure_cosmosdb import (  # type: ignore [import-untyped]
    AzureCosmosDBMemoryStore,
)
from semantic_kernel.connectors.memory.azure_cosmosdb.utils import (
    CosmosDBSimilarityType,
    CosmosDBVectorSearchType,
)
from semantic_kernel.core_plugins.text_memory_plugin import TextMemoryPlugin  # type: ignore [import-untyped]
from semantic_kernel.exceptions import FunctionExecutionException, KernelInvokeException, ServiceResponseException
from semantic_kernel.functions import KernelArguments, KernelFunction, KernelFunctionMetadata
from semantic_kernel.kernel import FunctionResult  # type: ignore [import-untyped]
from semantic_kernel.memory.memory_store_base import MemoryStoreBase  # type: ignore [import-untyped]
from semantic_kernel.memory.semantic_text_memory import (  # type: ignore [import-untyped]
    MemoryQueryResult,
    SemanticTextMemory,
)
from semantic_kernel.prompt_template import PromptTemplateConfig
from semantic_kernel.prompt_template.input_variable import InputVariable  # type: ignore [import-untyped]

logging.basicConfig(
    handlers=[logging.StreamHandler()],
    format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
    level=logging.INFO,
)


# collection name will be used multiple times in the code so we store it in a variable
database_name = os.getenv("AZURE_COSMOS_DATABASE_NAME", "semanticKernel")
collection_name = os.getenv("AZURE_COSMOS_COLLECTION_NAME", "textMemory")

# Vector search index parameters
index_name = os.getenv("AZURE_COSMOS_INDEX_NAME", "VectorSearchIndex")
vector_dimensions = 1536  # text-embedding-ada-002 uses a 1536-dimensional embedding vector
num_lists = 100
similarity = CosmosDBSimilarityType.COS
kind = CosmosDBVectorSearchType.VECTOR_HNSW
m = 16
ef_construction = 64
ef_search = 40


def get_mongo_connection_string() -> str:
    mongo_connection_string = os.getenv("AZURE_COSMOS_CONNECTION_STRING", "<YOUR-COSMOS-DB-CONNECTION-STRING>")
    mongo_username = quote_plus(os.getenv("AZURE_COSMOS_USERNAME", "admin"))
    mongo_password = quote_plus(os.getenv("AZURE_COSMOS_PASSWORD", "password"))
    return mongo_connection_string.replace("<user>", mongo_username).replace("<password>", mongo_password)


async def prompt_with_rag_or_vector(query_term: str, option: str) -> str:
    kernel: Kernel = initialize_sk_chat_embedding()
    memory, _ = await initialize_sk_memory_store(kernel)

    if option == "rag":
        chat_function: KernelFunction = await grounded_response(kernel)
        rag_result: FunctionResult = await perform_rag_search(kernel, memory, chat_function, query_term)
        rag_result_str: str = str(rag_result)
        return rag_result_str
    if option == "vector":
        result: list[MemoryQueryResult] = await perform_vector_search(memory, query_term)
        vector_result: str = str(result[0].text)
        return vector_result
    raise ValueError("Invalid option. Please choose either 'rag' or 'only-vector'.")


def initialize_sk_chat_embedding() -> Kernel:
    kernel = Kernel()
    # adding azure openai chat service
    chat_model_deployment_name = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", "chat-deployment")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "https://test-endpoint.openai.com/")
    api_key = os.getenv("AZURE_OPENAI_API_KEY", "VerySecretApiKey")

    kernel.add_service(
        AzureChatCompletion(
            service_id="chat_completion",
            deployment_name=chat_model_deployment_name,
            endpoint=endpoint,
            api_key=api_key,
        )
    )
    logging.info("Added Azure OpenAI Chat Service...")

    # adding azure openai text embedding service
    embedding_model_deployment_name = os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME", "embedding-deployment")

    kernel.add_service(
        AzureTextEmbedding(
            service_id="text_embedding",
            deployment_name=embedding_model_deployment_name,
            endpoint=endpoint,
            api_key=api_key,
        )
    )
    logging.info("Added Azure OpenAI Embedding Generation Service...")

    return kernel


async def initialize_sk_memory_store(
    kernel: Kernel,
) -> tuple[SemanticTextMemory, MemoryStoreBase]:
    # create azure cosmos db for mongo db vcore api store and collection with vector ivf
    # currently, semantic kernel only supports the ivf vector kind

    try:
        logging.info("Creating or updating Azure Cosmos DB Memory Store...")
        store = await AzureCosmosDBMemoryStore.create(
            cosmos_connstr=get_mongo_connection_string(),
            cosmos_api="mongo-vcore",
            database_name=database_name,
            collection_name=collection_name,
            index_name=index_name,
            vector_dimensions=vector_dimensions,
            num_lists=num_lists,
            similarity=similarity,
            kind=kind,
            m=m,
            ef_construction=ef_construction,
            ef_search=ef_search,
        )
        logging.info("Finished updating Azure Cosmos DB Memory Store...")

        memory = SemanticTextMemory(storage=store, embeddings_generator=kernel.get_service("text_embedding"))
        kernel.add_plugin(TextMemoryPlugin(memory), "TextMemoryPluginACDB")

        logging.info("Successfully Registered Azure Cosmos DB Memory Store...")
    except ServerSelectionTimeoutError:
        logging.error("Failed to Create or Update Azure Cosmos DB Memory Store...")
        logging.info("Creating or Updating Volatile Memory Store...")

        from semantic_kernel.memory.volatile_memory_store import VolatileMemoryStore

        store = VolatileMemoryStore()
        memory = SemanticTextMemory(storage=store, embeddings_generator=kernel.get_service("text_embedding"))
        kernel.add_plugin(TextMemoryPlugin(memory), "TextMemoryPluginACDB")

        logging.info("Successfully Registered Volatile Memory Store...")

    return memory, store


async def grounded_response(kernel: Kernel) -> KernelFunction:
    prompt = """
    You are a chatbot that can have a conversations about any topic related to the provided context.
    Give explicit answers from the provided context or say 'I don't know' if it does not have an answer.
    provided context: {{$db_record}}

    User: {{$query_term}}
    Chatbot:"""

    chat_model_deployment_name = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", "chat-deployment")

    execution_settings = OpenAITextPromptExecutionSettings(
        service_id="chat_completion",
        ai_model_id=chat_model_deployment_name,
        max_tokens=500,
        temperature=0.0,
        top_p=0.5,
    )
    chat_prompt_template_config = PromptTemplateConfig(
        template=prompt,
        name="grounded_response",
        template_format="semantic-kernel",
        input_variables=[
            InputVariable(name="db_record", description="The database record", is_required=True),
            InputVariable(name="query_term", description="The user input", is_required=True),
        ],
        execution_settings=execution_settings,
    )

    chat_function: KernelFunction = kernel.add_function(
        function_name="ChatGPTFunc",
        plugin_name="chatGPTPlugin",
        prompt_template_config=chat_prompt_template_config,
    )

    return chat_function


async def perform_rag_search(
    kernel: Kernel,
    memory: SemanticTextMemory,
    chat_function: KernelFunction,
    query_term: str,
) -> FunctionResult | None:
    vector_search_result: list[MemoryQueryResult] = await perform_vector_search(memory, query_term)
    db_record: str = str(vector_search_result[0].additional_metadata)

    try:
        return await kernel.invoke(
            chat_function,
            KernelArguments(query_term=query_term, db_record=db_record),
        )

    except (ServiceResponseException, FunctionExecutionException, KernelInvokeException):
        return FunctionResult(
            function=KernelFunctionMetadata(name=chat_function.name, is_prompt=chat_function.is_prompt),
            value="The requested data is not Found.",
        )


async def perform_vector_search(memory: SemanticTextMemory, query_term: str) -> list[MemoryQueryResult]:
    try:
        vector_search_result = await memory.search(collection_name, query_term)
    except ServiceResponseException:
        vector_search_result = [
            MemoryQueryResult(
                is_reference=False,
                external_source_name=None,
                id="0",
                description="The requested data is not Found.",
                text="Not found!",
                additional_metadata="The requested data is not Found.",
                embedding=None,
                relevance=0.0,
            )
        ]
    return vector_search_result
