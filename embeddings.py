import concurrent
import os
from typing import List

import pandas as pd
import tiktoken
from dotenv import load_dotenv
from openai.lib.azure import AzureOpenAI
from tenacity import retry, wait_random_exponential, stop_after_attempt
from tqdm import tqdm

load_dotenv()


# Batch Embedding
# Simple function to take in a list of text objects and return them as a list of embeddings
@retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(10))
def get_embeddings(input_to_embed: List):
    model = "recipes-embedding"
    response = client.embeddings.create(
        model=model,
        input=input_to_embed
    ).data
    return [data.embedding for data in response]


# Splits an iterable into batches of size n.
def batchify(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx: min(ndx + n, l)]


# Function for batching and parallel processing the embeddings
def embed_corpus(
        corpus: List[str],
        batch_size=64,
        num_workers=8,
        max_context_len=8191,
):
    # Encode the corpus, truncating to max_context_len
    encoding = tiktoken.get_encoding("cl100k_base")
    encoded_corpus = [
        encoded_article[:max_context_len] for encoded_article in encoding.encode_batch(corpus)
    ]

    # Calculate corpus statistics: the number of inputs, the total number of tokens, and the estimated cost to embed
    num_tokens = sum(len(article) for article in encoded_corpus)
    print(
        f"num_articles={len(encoded_corpus)}, num_tokens={num_tokens}"
    )

    # Embed the corpus
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:

        futures = [
            executor.submit(get_embeddings, text_batch)
            for text_batch in batchify(encoded_corpus, batch_size)
        ]

        with tqdm(total=len(encoded_corpus)) as pbar:
            for _ in concurrent.futures.as_completed(futures):
                pbar.update(batch_size)

        embeddings = []
        for future in futures:
            data = future.result()
            embeddings.extend(data)

        return embeddings


# Function to generate embeddings for a given column in a DataFrame
def generate_embeddings(df, column_name, embeddings_column_name='embeddings'):
    # Initialize an empty list to store embeddings
    recipe_names = df[column_name].astype(str).tolist()
    embeddings = embed_corpus(corpus=recipe_names)

    # Add the embeddings as a new column to the DataFrame
    df[embeddings_column_name] = embeddings
    print("Embeddings created successfully.")


def establish_azure_openai_connection() -> AzureOpenAI:
    print("Establishing Azure OpenAI connection...")
    azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_openai_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
    api_version = "2024-02-01"
    return AzureOpenAI(api_version=api_version,
                       api_key=azure_openai_api_key,
                       azure_endpoint=azure_openai_endpoint)


def controller() -> None:
    print("Opening dataset...")
    recipes_filepath = 'data/recipes.csv'
    recipes_df = pd.read_csv(recipes_filepath, on_bad_lines='skip')

    print(recipes_df.head())
    print("Opened dataset successfully. Dataset has {} recipes.".format(len(recipes_df)))

    print("Generating embeddings to file ...")
    generate_embeddings(df=recipes_df, column_name='Title', embeddings_column_name='titleEmbeddings')
    print("Writing embeddings to file ...")
    recipes_df.to_csv('data/recipes_with_embeddings.csv', index=False)
    print("Embeddings successfully stored in recipes_with_embeddings.csv")
    print(recipes_df.head())
    print("Opened dataset successfully. Dataset has {} items recipes along with their embeddings."
          .format(len(recipes_df)))


if __name__ == '__main__':
    client = establish_azure_openai_connection()
    controller()
