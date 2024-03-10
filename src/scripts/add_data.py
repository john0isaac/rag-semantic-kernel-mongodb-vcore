#!/usr/bin/env python3
import json
import os
import semantic_kernel as sk
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import (
    AzureChatCompletion,
    AzureTextEmbedding,
)
from semantic_kernel.connectors.memory.azure_cosmosdb import (
    AzureCosmosDBMemoryStore,
)
from semantic_kernel.core_plugins.text_memory_plugin import TextMemoryPlugin
from semantic_kernel.memory.semantic_text_memory import SemanticTextMemory
from semantic_kernel.memory.memory_store_base import MemoryStoreBase


# collection name will be used multiple times in the code so we store it in a variable
collection_name = os.environ.get("AZCOSMOS_CONTAINER_NAME")

# Vector search index parameters
index_name = "VectorSearchIndex"
vector_dimensions = (
    1536  # text-embedding-ada-002 uses a 1536-dimensional embedding vector
)
num_lists = 1
similarity = "COS"  # cosine distance


async def add_data() -> None:
    sk_kernel = initialize_sk_chat_embedding()
    sk_memory, sk_store = await initialize_sk_memory_store(sk_kernel)
    try:
        await upsert_data_to_memory_store(
            sk_memory, sk_store, "../data/text-sample.json"
        )
    except TimeoutError:
        await upsert_data_to_memory_store(
            sk_memory, sk_store, "../data/text-sample.json"
        )
    print("Added the data successfully...")


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
    # collection name will be used multiple times in the code so we store it in a variable
    collection_name = os.environ.get("AZCOSMOS_CONTAINER_NAME")

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


if __name__ == "__main__":
    import asyncio

    asyncio.run(add_data())
