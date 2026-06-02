from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# Initialize Flask App
app = Flask(__name__)
CORS(app)

# =========================
# Load Dataset
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR,
    "corrected_recipe_recommendation_dataset1.csv"
)

try:
    df = pd.read_csv(DATA_PATH)

    # Convert columns to lowercase strings
    df['available_ingredients'] = (
        df['available_ingredients']
        .astype(str)
        .str.lower()
    )

    df['recipe_name'] = (
        df['recipe_name']
        .astype(str)
        .str.lower()
    )

    df['recipe_ingredients'] = (
        df['recipe_ingredients']
        .astype(str)
        .str.lower()
    )

    df['instructions'] = (
        df['instructions']
        .astype(str)
    )

    print("Dataset loaded successfully!")

except Exception as e:
    print(f"Error loading dataset: {str(e)}")
    raise


# =========================
# TF-IDF Vectorizer
# =========================

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    stop_words='english'
)

# Create TF-IDF Matrix
tfidf_matrix = vectorizer.fit_transform(
    df['available_ingredients'] + " " + df['recipe_name']
)


# =========================
# Recipe Recommendation Function
# =========================

def recommend_recipes(user_input, top_n=5):

    user_input = user_input.lower().strip()

    # Empty input check
    if not user_input:
        return []

    # Transform user input into vector
    user_vector = vectorizer.transform([user_input])

    # Calculate cosine similarity
    similarity_scores = cosine_similarity(
        user_vector,
        tfidf_matrix
    ).flatten()

    # No similarity found
    if similarity_scores.max() == 0:
        return []

    # Get top matching recipe indices
    top_indices = similarity_scores.argsort()[::-1][:top_n]

    # Select recipe details
    best_recipes = df.iloc[top_indices][
        [
            'recipe_name',
            'recipe_ingredients',
            'instructions'
        ]
    ]

    # Remove duplicate recipes
    best_recipes = best_recipes.drop_duplicates(
        subset=['recipe_name']
    )

    return best_recipes.to_dict(orient="records")


# =========================
# Home Route
# =========================

@app.route('/')
def home():
    return """
    <h1>Recipe Recommendation System</h1>
    <p>Backend is running successfully!</p>
    <p>Use POST API endpoint:</p>
    <code>/get-recipe</code>
    """


# =========================
# Health Check Route
# =========================

@app.route('/health')
def health():
    return jsonify({
        "status": "success",
        "message": "Server is running properly"
    })


# =========================
# Recipe Recommendation API
# =========================

@app.route('/get-recipe', methods=['POST'])
def get_recipe():

    try:

        # Get JSON data
        data = request.get_json()

        # Check JSON input
        if not data:
            return jsonify({
                "status": "error",
                "message": "No JSON data received"
            }), 400

        # Get ingredients
        ingredients = data.get("ingredients", "")

        # Empty ingredients check
        if not ingredients:
            return jsonify({
                "status": "error",
                "message": "No ingredients provided"
            }), 400

        # Get recommendations
        recipes = recommend_recipes(ingredients)

        # Return recipes
        if recipes:

            return jsonify({
                "status": "success",
                "total_recipes": len(recipes),
                "recipes": recipes
            })

        # No recipes found
        else:

            return jsonify({
                "status": "not_found",
                "message": "No suitable recipes found"
            })

    except Exception as e:

        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


# =========================
# Run Flask App
# =========================

if __name__ == '__main__':

    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )