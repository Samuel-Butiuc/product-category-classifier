import pickle
import pandas as pd

with open("../models/model.pkl", "rb") as f:
    model = pickle.load(f)

while True:
    title = input("Introdu titlul produsului (sau 'exit'): ")
    if title == "exit":
        break

    df = pd.DataFrame([{
        "Product Title": title.lower(),
        "title_length": len(title),
        "word_count": len(title.split()),
        "has_number": int(any(char.isdigit() for char in title))
    }])

    prediction = model.predict(df)
    print("Categoria prezisă:", prediction[0])
