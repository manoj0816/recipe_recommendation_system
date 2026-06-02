import pandas as pd
import numpy as np
import ipywidgets as widgets
from IPython.display import display
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_csv("corrected_recipe_recommendation_dataset.csv")

df['available_ingredients'] = df['available_ingredients'].astype(str).str.lower()
df['recipe_name'] = df['recipe_name'].astype(str).str.lower()
df['recipe_ingredients'] = df['recipe_ingredients'].astype(str).str.lower()

def process_text(text):
    return text.lower().strip()

# Vectorize ingredients and food names
vectorizer = TfidfVectorizer(ngram_range=(1,2), stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['available_ingredients'] + " " + df['recipe_name'])

def recommend_recipes(user_input, top_n=5):
    user_input = process_text(user_input)
    if not user_input:
        return "Please enter ingredients or a food name."

    # Check for direct recipe name match
    match = df[df['recipe_name'].str.contains(user_input, case=False, na=False)]
    if not match.empty:
        return match[['recipe_name', 'recipe_ingredients', 'instructions']]

    # Compute similarity scores
    user_vector = vectorizer.transform([user_input])
    similarity_scores = cosine_similarity(user_vector, tfidf_matrix).flatten()

    if similarity_scores.max() == 0:
        return "No suitable recipe found. Try different ingredients."

    top_indices = similarity_scores.argsort()[::-1][:top_n]
    best_recipes = df.iloc[top_indices][['recipe_name', 'recipe_ingredients', 'instructions']]
    best_recipes = best_recipes.drop_duplicates(subset=['recipe_name'])

    return best_recipes if not best_recipes.empty else "No matching recipes found."

# Create input widgets
ingredient_input = widgets.Text(
    placeholder="Enter ingredients or food name",
    description="Input:",
    layout=widgets.Layout(width='70%')
)

button = widgets.Button(description="Get Recipe")
output = widgets.Output()

def on_button_click(b):
    with output:
        output.clear_output()
        user_query = ingredient_input.value.strip()
        result = recommend_recipes(user_query)
        display(result)

button.on_click(on_button_click)

display(ingredient_input, button, output)