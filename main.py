from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

app = FastAPI()

# Enable CORS to allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (Change this when deploying)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the dataset
df = pd.read_csv("corrected_recipe_recommendation_dataset.csv")

@app.get("/get_recipe/")
def get_recipe(ingredients: str = Query(..., description="Comma-separated list of ingredients")):
    user_ingredients = set(ingredients.lower().split(", "))

    matching_recipes = []
    for index, row in df.iterrows():
        recipe_ingredients = set(row["ingredients"].lower().split(", "))
        if user_ingredients.issubset(recipe_ingredients):  # Check if user's ingredients are a subset
            matching_recipes.append({"recipe": row["recipe_name"], "procedure": row["procedure"]})

    return {"recipes": matching_recipes if matching_recipes else "No matching recipes found"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
