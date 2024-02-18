import json
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import (
    AzureChatCompletion,
    AzureTextEmbedding,
)
from semantic_kernel.connectors.memory.azure_cosmosdb import (
    AzureCosmosDBMemoryStore,
)

# load the environment variables file
from dotenv import dotenv_values

config = dotenv_values(".env")

# collection name will be used multiple times in the code so we store it in a variable
collection_name = config.get("AZCOSMOS_CONTAINER_NAME")

# Vector search index parameters
index_name = "VectorSearchIndex"
vector_dimensions = 1536 # text-embedding-ada-002 uses a 1536-dimensional embedding vector
num_lists = 1
similarity = "COS" # cosine distance 

async def prompt_with_rag_or_vector(query_term: str, option: str, update_data: bool) -> str:
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
    store = await initialize_sk_memory_store(kernel)

    if update_data:
        await upsert_data_to_memory_store(kernel.memory, store, "text-sample.json")

    if option == "rag":
        result = await perform_rag_search(kernel, query_term)
        return result
    elif option == "vector":
        result = await perform_vector_search(kernel.memory, query_term)
        return result[0].text
    else:
        raise ValueError("Invalid option. Please choose either 'rag' or 'only-vector'.")

def initialize_sk_chat_embedding() -> callable:
    # get api key and endpoint from .env file
    _, api_key, endpoint = sk.azure_openai_settings_from_dot_env()
    kernel = sk.Kernel()
    # adding azure openai chat service
    chat_model_deployment_name = config.get("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")

    azure_chat_service = AzureChatCompletion(
        deployment_name=chat_model_deployment_name,
        endpoint=endpoint,
        api_key=api_key
    )
    kernel.add_chat_service(chat_model_deployment_name, azure_chat_service)
    print("Added Azure OpenAI Chat Service...")

    # adding azure openai text embedding service
    embedding_model_deployment_name = config.get("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME")

    azure_text_embedding_service = AzureTextEmbedding(
        deployment_name=embedding_model_deployment_name,
        endpoint=endpoint,
        api_key=api_key
    )

    kernel.add_text_embedding_generation_service(embedding_model_deployment_name, azure_text_embedding_service)
    print("Added Azure OpenAI Embedding Generation Service...")

    return kernel

async def initialize_sk_memory_store(kernel: callable) -> callable:
    print("Creating or updating Azure Cosmos DB Memory Store...")
    # create azure cosmos db for mongo db vcore api store and collection with vector ivf
    # currently, semantic kernel only supports the ivf vector kind
    store  = await AzureCosmosDBMemoryStore.create(
        database_name=config.get("AZCOSMOS_DATABASE_NAME"),
        collection_name=collection_name,
        index_name=index_name,
        vector_dimensions=vector_dimensions,
        num_lists=num_lists,
        similarity=similarity
    )
    print("Finished updating Azure Cosmos DB Memory Store...")
    kernel.register_memory_store(memory_store=store)
    print("Registered Azure Cosmos DB Memory Store...")
    return store

async def perform_rag_search(kernel: callable, query_term: str):
    result = await perform_vector_search(kernel.memory, query_term)

    prompt = """
    You are a chatbot that can have a conversations about any topic related to the provided context.
    Start by saying how relevant the question is using the provided context relevancy score.
    Give explicit answers from the provided context or say 'I don't know' if it does not have an answer.
    provided context: {{$db_record}}

    User: {{$query_term}}
    Chatbot:"""

    chat_function = kernel.create_semantic_function(prompt, max_tokens=500, temperature=0.0, top_p=0.5)
    context = kernel.create_new_context()

    context['query_term'] = query_term
    context['db_record'] = result[0].additional_metadata

    return await chat_function.invoke(context=context)
 
async def perform_vector_search(kernel_memory_store: callable, query_term: str) -> list:
    return await kernel_memory_store.search(collection_name, query_term)

async def upsert_data_to_memory_store(kernel_memory_store: callable, memory_store: callable, data_file_path: str) -> None:
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
            n+=1
            try:
                already_created = bool(await memory_store.get(collection_name, item["id"], with_embedding=True))
            except Exception:
                already_created = False
            if not already_created:
                await kernel_memory_store.save_information(
                    collection=collection_name,
                    id=item["id"],
                    text=item["content"],
                    description=item["title"]
                )
                print("Generating embeddings and saving new item:", n, "/" ,len(data), end='\r')
            else:
                print("Skipping item already exits:", n, "/" ,len(data), end='\r')
