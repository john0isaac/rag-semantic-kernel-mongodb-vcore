import json
import os
import semantic_kernel as sk
from semantic_kernel import Kernel, KernelFunction
from semantic_kernel.connectors.ai.open_ai import (
    AzureChatCompletion,
    AzureTextEmbedding,
)
from semantic_kernel.connectors.memory.azure_cosmosdb import (
    AzureCosmosDBMemoryStore,
)
from semantic_kernel.memory.semantic_text_memory import SemanticTextMemory
from semantic_kernel.memory.memory_store_base import MemoryStoreBase
from semantic_kernel.core_plugins.text_memory_plugin import TextMemoryPlugin

from semantic_kernel.prompt_template.input_variable import InputVariable
import semantic_kernel.connectors.ai.open_ai as sk_oai


# collection name will be used multiple times in the code so we store it in a variable
collection_name = os.environ.get("AZCOSMOS_CONTAINER_NAME")

# Vector search index parameters
index_name = "VectorSearchIndex"
vector_dimensions = (
    1536  # text-embedding-ada-002 uses a 1536-dimensional embedding vector
)
num_lists = 1
similarity = "COS"  # cosine distance


async def prompt_with_rag_or_vector(
    query_term: str, option: str, update_data: bool
) -> str:
    """
    This asynchronous function initializes a kernel and a memory store, optionally updates the memory store with data from a file,
    and performs a search based on the provided query term using either the RAG (Retrieval-Augmented Generation) method or vector search.

    Args:
        query_term (str): The term to be searched for.
        option (str): The search method to be used. Must be either 'rag' or 'vector'.
        update_data (bool): If True, the memory store is updated with data from the file "text-sample.json".

    Returns:
        str: The function returns the text response.

    Raises:
        ValueError: If an invalid option is provided.
    """
    kernel = initialize_sk_chat_embedding()
    memory, store = await initialize_sk_memory_store(kernel)

    if update_data:
        await upsert_data_to_memory_store(memory, store, "text-sample.json")

    if option == "rag":
        chat_function = await grounded_response(kernel)
        result = await perform_rag_search(kernel, memory, chat_function, query_term)
        return result
    if option == "vector":
        result = await perform_vector_search(memory, query_term)
        return result[0].text
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
    embedding_model_deployment_name = os.environ.get(
        "AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME"
    )

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
    memory = SemanticTextMemory(
        storage=store, embeddings_generator=kernel.get_service("text_embedding")
    )
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

    execution_settings = sk_oai.OpenAITextPromptExecutionSettings(
        service_id="chat_completion",
        ai_model_id=chat_model_deployment_name,
        max_tokens=500,
        temperature=0.0,
        top_p=0.5,
    )
    chat_prompt_template_config = sk.PromptTemplateConfig(
        template=prompt,
        name="grounded_response",
        template_format="semantic-kernel",
        input_variables=[
            InputVariable(
                name="db_record", description="The database record", is_required=True
            ),
            InputVariable(
                name="query_term", description="The user input", is_required=True
            ),
        ],
        execution_settings=execution_settings,
    )

    chat_function = kernel.create_function_from_prompt(
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
) -> str:
    result = await perform_vector_search(memory, query_term)
    return await kernel.invoke(
        chat_function,
        sk.KernelArguments(
            query_term=query_term, db_record=result[0].additional_metadata
        ),
    )


async def perform_vector_search(memory: SemanticTextMemory, query_term: str) -> list:
    return await memory.search(collection_name, query_term)


async def upsert_data_to_memory_store(
    memory: SemanticTextMemory, store: MemoryStoreBase, data_file_path: str
) -> None:
    """
    This asynchronous function takes two memory stores and a data file path as arguments.
    It is designed to upsert (update or insert) data into the memory stores from the data file.

    Args:
        kernel_memory_store (callable): A callable object that represents the kernel memory store where data will be upserted.
        memory_store (callable): A callable object that represents the memory store where data will be upserted.
        data_file_path (str): The path to the data file that contains the data to be upserted.

    Returns:
        None. The function performs an operation that modifies the memory stores in-place.
    """
    with open(file=data_file_path, mode="r", encoding="utf-8") as f:
        data = json.load(f)
        n = 0
        for item in data:
            n += 1
            # check if the item already exists in the memory store
            # if the id doesn't exist, it throws an exception
            try:
                already_created = bool(
                    await store.get(collection_name, item["id"], with_embedding=True)
                )
            except Exception:
                already_created = False
            # if the record doesn't exist, we generate embeddings and save it to the database
            if not already_created:
                await memory.save_information(
                    collection=collection_name,
                    id=item["id"],
                    # the embedding is generated from the text field
                    text=item["content"],
                    description=item["title"],
                )
                print(
                    "Generating embeddings and saving new item:",
                    n,
                    "/",
                    len(data),
                    end="\r",
                )
            else:
                print("Skipping item already exits:", n, "/", len(data), end="\r")
