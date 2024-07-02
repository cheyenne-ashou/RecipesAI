# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import json
import os
import re
from typing import List

from dotenv import load_dotenv
from openai.lib.azure import AzureOpenAI
from pandas import DataFrame
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()


# Remove non-alphabetical characters from a string.
def clean_text(text):
    return re.sub(r'[^a-zA-Z\s]', '', text)


def process_diet_and_ingredients(diet, ingredients):
    user_preferences = {}
    # Clean the diet string
    if diet:
        user_preferences['dietary_goals'] = clean_text(diet)
    if ingredients:
        # Clean each ingredient and convert to comma-separated string
        cleaned_ingredients = [clean_text(ingredient) for ingredient in ingredients]
        ingredients_str = ', '.join(cleaned_ingredients)
        user_preferences['ingredients'] = ingredients_str

    return user_preferences


def controller(recipes_df: DataFrame, diet: str, ingredients: [str]) -> None:
    # Analyze the image and return the results
    user_preferences = process_diet_and_ingredients(diet=diet, ingredients=ingredients)

    generated_recipes = analyze_diet(user_preferences)
    print('\nChat completions generated')

    print(f'\nChat completions diet recommendation: {generated_recipes}')

    # Extract the relevant features from the analysis
    recipes = generated_recipes['recipe_names']

    # Filter data such that we only look through the recipes of the same gender (or unisex) and different category
    # filtered_recipes = recipes_df.loc[recipes_df['ingredients'].contains("cheese")]
    # filtered_recipes = filtered_recipes[filtered_recipes['category'] != recipe_category]
    # print(str(len(filtered_recipes)) + " Remaining recipes")

    # Find the most similar recipes based on the input recipe descriptions
    print('\nFinding matching recipes...')
    matching_recipes = find_matching_recipes_with_rag(recipes_df, recipes)
    # Display the matching recipes (this will display 2 recipes for each description in the image analysis)

    get_recipes(matching_recipes)


def get_recipes(matching_recipes):
    for i, recipe in enumerate(matching_recipes):
        image_name = recipe['Image_Name']
        # Path to the image file
        image_path = f'data/images/{image_name}.jpg'
        recipe_title = recipe['Title']
        print("\nRecommended recipe: ", recipe_title)


def analyze_diet(user_preferences: dict) -> dict:
    print("Generating chat completions...")
    try:
        ingredients = user_preferences['ingredients']
        dietary_goals = user_preferences['dietary_goals']
        instructions = ''
        if ingredients:
            ingredient_prompt = (f'\n```'
                                 f'\nfind recipes that traditionally make use of some of the following ingredients: {ingredients}'
                                 f'\n```')
            instructions += ingredient_prompt
        if dietary_goals:
            dietary_prompt = (f'\n```'
                              f'\nfind recipes that will help a user or patient meet the following dietary goal: {dietary_goals}'
                              '\n```')
            instructions += dietary_prompt
        example = 'Example Input: "Lose weight and gain muscle.".\n' \
                  'Example Output: {"recipe_names": ["Chicken Fajita Wrap", "Korean Beef Bowl", "Smoky Chicken Quinoa ' \
                  'Soup", "Indian Butter Chickpeas"]}'
        'Example Input: "rice, chicken, quinoa, bell peppers".\n' \
        'Example Output: {"recipe_names": ["Chicken Fajita Wrap", "Korean Spicy Chicken Bowl", "Smoky Chicken Quinoa and Pepper Soup' \
        'Soup", "Indian Butter Chickpeas"]}'
        prompt = 'Given textual information of a list of ingredients or dietary goals, ' \
                 'generate JSON output with the following field: "recipe_names".' \
                 '\nUse your understanding of nutrition, along with cuisines from around the world, ' \
                 'to find recipes based on the following instructions delimited by triple backticks. ' \
                 'If there are multiple instructions, follow all of them to generate a single list of recipe_names:' \
                 f'{instructions}' \
                 '\nDo not include the actual recipe instructions or ingredients. Do not include the ```json ``` tag in the output.' \
                 f'\n{example}'

        print("\nConnected to Azure OpenAI")
        response = client.chat.completions.create(
            model="text-generation",
            messages=[
                {"role": "system", "content": 'Assistant is a professional nutritionist and amateur chef.'},
                {"role": "user", "content": prompt}
            ]
        )
        features = response.choices[0].message.content
    except Exception as e:
        raise SystemExit(f"Failed to make the request. Error: {e}")
    return json.loads(features)


def establish_azure_openai_connection() -> AzureOpenAI:
    print("Establishing Azure OpenAI connection...")
    azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_openai_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
    api_version = "2024-02-01"
    return AzureOpenAI(api_version=api_version,
                       api_key=azure_openai_api_key,
                       azure_endpoint=azure_openai_endpoint)


# Function to calculate cosine similarity between two embeddings
def calculate_similarity(embedding1, embedding2):
    return cosine_similarity(embedding1, [embedding2])[0][0]


def find_similar_recipes(input_embedding, embeddings, threshold=0.5, top_k=1):
    """Find the most similar recipes based on cosine similarity."""
    # Calculate cosine similarity between the input embedding and all other embeddings
    similarities = [(index, calculate_similarity(input_embedding, vec)) for index, vec in enumerate(embeddings)]

    # Filter out any similarities below the threshold
    filtered_similarities = [(index, sim) for index, sim in similarities if sim >= threshold]

    # Sort the filtered similarities by similarity score
    sorted_indices = sorted(filtered_similarities, key=lambda x: x[1], reverse=True)[:top_k]

    # Return the top-k most similar recipes
    return sorted_indices


def find_matching_recipes_with_rag(recipes_df, recipes):
    """Take the input recipe descriptions and find the most similar recipes based on cosine similarity for each
    description."""
    # Select the embeddings from the DataFrame.
    embeddings = recipes_df["titleEmbeddings"].tolist()

    similar_recipes = []
    for recipe in recipes:
        # Generate the embedding for the input recipe
        input_embedding = get_embeddings([recipe])

        # Find the most similar recipes based on cosine similarity
        similar_indices = find_similar_recipes(input_embedding, embeddings, threshold=0.6)
        similar_recipes += [recipes_df.iloc[i[0]] for i in similar_indices]

    return similar_recipes


def get_embeddings(input_to_embed: List):
    model = "recipes-embedding"
    response = client.embeddings.create(
        model=model,
        input=input_to_embed
    ).data
    return [data.embedding for data in response]


client = establish_azure_openai_connection()
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    controller()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
