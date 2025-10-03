from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load(r"C:\Users\johnn\random_forest_model.pkl")
@app.route("/")
def home():
    return "ML Prediction API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]
    return jsonify({"predicted_sales": prediction})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)




