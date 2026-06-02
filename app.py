from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# =========================
# Initialize Flask App
# =========================

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =========================
# Load Dataset
# =========================

DATA_PATH = os.path.join(
    BASE_DIR,
    "corrected_recipe_recommendation_dataset1.csv"
)

try:
    df = pd.read_csv(DATA_PATH)

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

tfidf_matrix = vectorizer.fit_transform(
    df['available_ingredients'] + " " + df['recipe_name']
)

# =========================
# Recipe Recommendation Function
# =========================

def recommend_recipes(user_input, top_n=5):

    user_input = user_input.lower().strip()

    if not user_input:
        return []

    user_vector = vectorizer.transform([user_input])

    similarity_scores = cosine_similarity(
        user_vector,
        tfidf_matrix
    ).flatten()

    if similarity_scores.max() == 0:
        return []

    top_indices = similarity_scores.argsort()[::-1][:top_n]

    best_recipes = df.iloc[top_indices][
        [
            'recipe_name',
            'recipe_ingredients',
            'instructions'
        ]
    ]

    best_recipes = best_recipes.drop_duplicates(
        subset=['recipe_name']
    )

    return best_recipes.to_dict(orient="records")

# =========================
# HTML FILE ROUTES
# =========================

@app.route('/')
def welcome():
    return send_from_directory(BASE_DIR, 'welcome.html')

@app.route('/welcome.html')
def welcome_page():
    return send_from_directory(BASE_DIR, 'welcome.html')

@app.route('/home.html')
def home_page():
    return send_from_directory(BASE_DIR, 'home.html')

@app.route('/about.html')
def about():
    return send_from_directory(BASE_DIR, 'about.html')

@app.route('/chatbot.html')
def chatbot():
    return send_from_directory(BASE_DIR, 'chatbot.html')

@app.route('/results.html')
def results():
    return send_from_directory(BASE_DIR, 'results.html')

@app.route('/country-selection.html')
def country_selection():
    return send_from_directory(BASE_DIR, 'country-selection.html')

@app.route('/breakfast.html')
def breakfast():
    return send_from_directory(BASE_DIR, 'breakfast.html')

@app.route('/lunch.html')
def lunch():
    return send_from_directory(BASE_DIR, 'lunch.html')

@app.route('/dinner.html')
def dinner():
    return send_from_directory(BASE_DIR, 'dinner.html')

@app.route('/drinks.html')
def drinks():
    return send_from_directory(BASE_DIR, 'drinks.html')

@app.route('/diet.html')
def diet():
    return send_from_directory(BASE_DIR, 'diet.html')

@app.route('/vegeterian.html')
def vegeterian():
    return send_from_directory(BASE_DIR, 'vegeterian.html')

@app.route('/non-vegeterian.html')
def non_vegeterian():
    return send_from_directory(BASE_DIR, 'non-vegeterian.html')

@app.route('/sweets.html')
def sweets():
    return send_from_directory(BASE_DIR, 'sweets.html')

@app.route('/deserts.html')
def deserts():
    return send_from_directory(BASE_DIR, 'deserts.html')

# =========================
# VIDEO ROUTE
# =========================

@app.route('/cooking3.mp4')
def cooking_video():
    return send_from_directory(BASE_DIR, 'cooking3.mp4')

# =========================
# HEALTH CHECK
# =========================

@app.route('/health')
def health():
    return jsonify({
        "status": "success",
        "message": "Server is running properly"
    })

# =========================
# RECIPE API
# =========================

@app.route('/get-recipe', methods=['POST'])
def get_recipe():

    try:

        data = request.get_json()

        if not data:
            return jsonify({
                "status": "error",
                "message": "No JSON data received"
            }), 400

        ingredients = data.get("ingredients", "").strip()

        if not ingredients:
            return jsonify({
                "status": "error",
                "message": "No ingredients provided"
            }), 400

        recipes = recommend_recipes(ingredients)

        if recipes:

            return jsonify({
                "status": "success",
                "total_recipes": len(recipes),
                "recipes": recipes
            })

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
# LOCAL RUN
# =========================

if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )

