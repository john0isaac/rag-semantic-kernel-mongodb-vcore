#!/usr/bin/env python3
import json
import os
from semantic_kernel.memory.semantic_text_memory import SemanticTextMemory
from semantic_kernel.memory.memory_store_base import MemoryStoreBase

from quartapp.rag import initialize_sk_chat_embedding, initialize_sk_memory_store

# path to JSON file containing the data
data_path = os.environ.get("JSON_DATA_PATH", "./data/text-sample.json")

# collection name will be used multiple times in the code so we store it in a variable
collection_name = os.environ.get("AZCOSMOS_CONTAINER_NAME")


async def add_data() -> None:
    sk_kernel = initialize_sk_chat_embedding()
    sk_memory, sk_store = await initialize_sk_memory_store(sk_kernel)
    try:
        await upsert_data_to_memory_store(sk_memory, sk_store, data_path)
    except TimeoutError:
        await upsert_data_to_memory_store(sk_memory, sk_store, data_path)
    print("Added the data successfully...")


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
