#!/usr/bin/env python3
import json
import logging
import os
from argparse import ArgumentParser, Namespace

from semantic_kernel.memory.memory_store_base import MemoryStoreBase  # type: ignore [import-untyped]
from semantic_kernel.memory.semantic_text_memory import SemanticTextMemory  # type: ignore [import-untyped]

from quartapp.rag import initialize_sk_chat_embedding, initialize_sk_memory_store

logging.basicConfig(
    handlers=[logging.StreamHandler()],
    format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
    level=logging.INFO,
)


async def add_data(input_args: Namespace) -> None:
    sk_kernel = initialize_sk_chat_embedding()
    sk_memory, sk_store = await initialize_sk_memory_store(sk_kernel)
    try:
        await upsert_data_to_memory_store(
            sk_memory,
            sk_store,
            input_args.file,
            input_args.id_field,
            input_args.text_field,
            input_args.description_field,
        )
    except TimeoutError:
        await upsert_data_to_memory_store(
            sk_memory,
            sk_store,
            input_args.file,
            input_args.id_field,
            input_args.text_field,
            input_args.description_field,
        )
    logging.info("Successfully Added the data...")


async def upsert_data_to_memory_store(
    memory: SemanticTextMemory,
    store: MemoryStoreBase,
    data_file_path: str,
    id_field_name: str,
    text_field_name: str,
    description_field_name: str,
) -> None:
    # collection name will be used multiple times in the code so we store it in a variable
    collection_name: str = os.environ.get("AZCOSMOS_CONTAINER_NAME") or "sk_collection"

    with open(file=data_file_path, encoding="utf-8") as f:
        data = json.load(f)
        n = 0
        for item in data:
            n += 1
            # check if the item already exists in the memory store
            # if the id doesn't exist, it throws an exception
            try:
                already_created = bool(await store.get(collection_name, item[id_field_name], with_embedding=True))
            except Exception:
                already_created = False
            # if the record doesn't exist, we generate embeddings and save it to the database
            if not already_created:
                await memory.save_information(
                    collection=collection_name,
                    id=item[id_field_name],
                    # the embedding is generated from the text field
                    text=item[text_field_name],
                    description=item[description_field_name],
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


def get_input_args() -> Namespace:
    # Parse using ArgumentParser
    parser = ArgumentParser()

    parser.add_argument(
        "-f",
        "--file",
        type=str,
        default="./data/text-sample.json",
        help="path to the JSON file containing the data",
    )

    parser.add_argument(
        "-id",
        "--id-field",
        type=str,
        help="The id field to identify the data",
        default="id",
    )

    parser.add_argument(
        "-txt",
        "--text-field",
        type=str,
        help="The text field to generate the embedding from",
        default="content",
    )

    parser.add_argument(
        "-desc",
        "--description-field",
        type=str,
        help="The description field for the data",
        default="title",
    )

    return parser.parse_args()


if __name__ == "__main__":
    import asyncio

    input_args = get_input_args()
    asyncio.run(add_data(input_args))
