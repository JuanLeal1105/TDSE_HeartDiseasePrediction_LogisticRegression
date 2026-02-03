from flask import Flask, request, jsonify
import json
import numpy as np

app = Flask(__name__)

with open("heart_disease_model.json", "r") as f:
    model = json.load(f)

w = np.array(model["weights"])
b = model["bias"]
mean = np.array(model["mean"])
std = np.array(model["std"])

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    X = np.array(data["features"]).reshape(1, -1)
    X_norm = (X - mean) / std

    z = X_norm @ w + b
    prob = float(sigmoid(z)[0])

    return jsonify({
        "probability": prob,
        "risk": "High" if prob >= 0.5 else "Low"
    })

if __name__ == "__main__":
    print("Local model server running at http://127.0.0.1:5000")
    app.run(port=5000, debug=True)

