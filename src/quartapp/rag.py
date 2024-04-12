import os
from typing import Any

import semantic_kernel as sk  # type: ignore [import-untyped]
from semantic_kernel import Kernel, KernelFunction, PromptTemplateConfig
from semantic_kernel.connectors.ai.open_ai import (  # type: ignore [import-untyped]
    AzureChatCompletion,
    AzureTextEmbedding,
    OpenAITextPromptExecutionSettings,
)
from semantic_kernel.connectors.memory.azure_cosmosdb import (  # type: ignore [import-untyped]
    AzureCosmosDBMemoryStore,
)
from semantic_kernel.core_plugins.text_memory_plugin import TextMemoryPlugin  # type: ignore [import-untyped]
from semantic_kernel.kernel import FunctionResult  # type: ignore [import-untyped]
from semantic_kernel.memory.memory_store_base import MemoryStoreBase  # type: ignore [import-untyped]
from semantic_kernel.memory.semantic_text_memory import (  # type: ignore [import-untyped]
    MemoryQueryResult,
    SemanticTextMemory,
)
from semantic_kernel.prompt_template.input_variable import InputVariable  # type: ignore [import-untyped]

# collection name will be used multiple times in the code so we store it in a variable
collection_name = os.environ.get("AZCOSMOS_CONTAINER_NAME")

# Vector search index parameters
index_name = "VectorSearchIndex"
vector_dimensions = 1536  # text-embedding-ada-002 uses a 1536-dimensional embedding vector
num_lists = 1
similarity = "COS"  # cosine distance


async def prompt_with_rag_or_vector(query_term: str, option: str) -> str:
    kernel: Kernel = initialize_sk_chat_embedding()
    memory, _ = await initialize_sk_memory_store(kernel)

    if option == "rag":
        chat_function: KernelFunction = await grounded_response(kernel)
        rag_result: FunctionResult = await perform_rag_search(kernel, memory, chat_function, query_term)
        rag_result_str: str = rag_result.__str__()
        return rag_result_str
    if option == "vector":
        result: list[MemoryQueryResult] = await perform_vector_search(memory, query_term)
        vector_result: str = result[0].text if result else "Not found!"
        return vector_result
    raise ValueError("Invalid option. Please choose either 'rag' or 'only-vector'.")


def initialize_sk_chat_embedding() -> Kernel:
    kernel = sk.Kernel()
    # adding azure openai chat service
    chat_model_deployment_name = os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")

    kernel.add_service(
        AzureChatCompletion(
            service_id="chat_completion",
            deployment_name=chat_model_deployment_name,
            endpoint=endpoint,
            api_key=api_key,
        )
    )
    print("Added Azure OpenAI Chat Service...")

    # adding azure openai text embedding service
    embedding_model_deployment_name = os.environ.get("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME")

    kernel.add_service(
        AzureTextEmbedding(
            service_id="text_embedding",
            deployment_name=embedding_model_deployment_name,
            endpoint=endpoint,
            api_key=api_key,
        )
    )
    print("Added Azure OpenAI Embedding Generation Service...")

    return kernel


async def initialize_sk_memory_store(
    kernel: Kernel,
) -> tuple[SemanticTextMemory, MemoryStoreBase]:
    print("Creating or updating Azure Cosmos DB Memory Store...")
    # create azure cosmos db for mongo db vcore api store and collection with vector ivf
    # currently, semantic kernel only supports the ivf vector kind
    store = await AzureCosmosDBMemoryStore.create(
        cosmos_connstr=os.environ.get("AZCOSMOS_CONNSTR"),
        cosmos_api=os.environ.get("AZCOSMOS_API"),
        database_name=os.environ.get("AZCOSMOS_DATABASE_NAME"),
        collection_name=collection_name,
        index_name=index_name,
        vector_dimensions=vector_dimensions,
        num_lists=num_lists,
        similarity=similarity,
    )
    print("Finished updating Azure Cosmos DB Memory Store...")
    memory = SemanticTextMemory(storage=store, embeddings_generator=kernel.get_service("text_embedding"))
    kernel.import_plugin_from_object(TextMemoryPlugin(memory), "TextMemoryPluginACDB")
    print("Registered Azure Cosmos DB Memory Store...")
    return memory, store


async def grounded_response(kernel: Kernel) -> KernelFunction:
    prompt = """
    You are a chatbot that can have a conversations about any topic related to the provided context.
    Give explicit answers from the provided context or say 'I don't know' if it does not have an answer.
    provided context: {{$db_record}}

    User: {{$query_term}}
    Chatbot:"""

    chat_model_deployment_name = os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")

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

    chat_function: KernelFunction = kernel.create_function_from_prompt(
        prompt=prompt,
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
) -> FunctionResult:
    result: list[MemoryQueryResult] = await perform_vector_search(memory, query_term)
    db_record: str = result[0].additional_metadata if result else "The requested data is not Found."
    return await kernel.invoke(
        chat_function,
        sk.KernelArguments(query_term=query_term, db_record=db_record),
    )


async def perform_vector_search(memory: SemanticTextMemory, query_term: str) -> list[MemoryQueryResult] | Any:
    return await memory.search(collection_name, query_term)
