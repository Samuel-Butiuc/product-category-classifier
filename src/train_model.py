import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

# 1. Încarcă datele
df = pd.read_csv("../data/products.csv")
df.columns = df.columns.str.strip()

# 2. Curățare minimă
df = df.dropna(subset=["Product Title", "Category Label"])
df["Product Title"] = df["Product Title"].str.lower()

# 3. Feature engineering 
df["title_length"] = df["Product Title"].apply(len)
df["word_count"] = df["Product Title"].apply(lambda x: len(x.split()))
df["has_number"] = df["Product Title"].str.contains(r"\d").astype(int)

# 4. Separare features / target
X = df[["Product Title", "title_length", "word_count", "has_number"]]
y = df["Category Label"]

# 5. Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Preprocesare + model
preprocessor = ColumnTransformer(
    transformers=[
        ("text", TfidfVectorizer(stop_words="english", max_features=5000), "Product Title"),
        ("num", "passthrough", ["title_length", "word_count", "has_number"])
    ]
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LinearSVC())
    ]
)

# 7. Antrenare
model.fit(X_train, y_train)

# 8. Salvare model
with open("../models/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model antrenat și salvat cu succes!")
