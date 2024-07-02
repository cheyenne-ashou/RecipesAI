# This is a sample Python script.
import argparse
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import ast
import json
import os
from typing import List

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from openai.lib.azure import AzureOpenAI
from dotenv import load_dotenv
from recipes import controller

load_dotenv()


def main() -> None:
    # Set up command line argument parsing
    parser = get_parser()
    args = parser.parse_args()
    recipes_df = get_df()
    controller(recipes_df, args.diet, args.ingredients)


def get_parser():
    parser = argparse.ArgumentParser(description="AI Recipe Generator")
    parser.add_argument("--diet", type=str, required=False,
                        help='The type of diet to find a recipe for. For example: "Lose fat, gain muscle mass"')
    parser.add_argument("--ingredients", type=str, required=False, nargs='+',
                        help='A list of ingredients. For example: "eggs" "ground beef" "bell peppers"')
    parser.add_argument("--mode",
                        type=str,
                        required=False,
                        help="Defines which part of the image to edit (only for image editing).",
                        choices=["background", "foreground"])
    return parser


def get_df():
    print('\nReading embeddings data file...')
    recipes_df = pd.read_csv('data/recipes_with_embeddings.csv', on_bad_lines='skip')
    # # Convert the 'embeddings' column from string representations of lists to actual lists of floats
    recipes_df['titleEmbeddings'] = recipes_df['titleEmbeddings'].apply(lambda x: ast.literal_eval(x))
    print('\nConverted embeddings from string representation')
    return recipes_df


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
