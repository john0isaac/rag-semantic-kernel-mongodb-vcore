#!/usr/bin/env python3
from rag import (
    initialize_sk_chat_embedding,
    initialize_sk_memory_store,
    upsert_data_to_memory_store,
)


async def add_data() -> None:
    sk_kernel = initialize_sk_chat_embedding()
    sk_memory, sk_store = await initialize_sk_memory_store(sk_kernel)
    try:
        await upsert_data_to_memory_store(sk_memory, sk_store, "text-sample.json")
    except TimeoutError:
        await upsert_data_to_memory_store(sk_memory, sk_store, "text-sample.json")
    print("Added the data successfully...")


if __name__ == "__main__":
    import asyncio
    asyncio.run(add_data())
