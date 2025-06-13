from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import logging
import joblib
import os

# ========================
# Logging Configuration
# ========================
logging.basicConfig(
  level=logging.INFO,
  format="%(asctime)s [%(levelname)s] %(message)s",
)

# ========================
# App Initialization
# ========================
app = Flask(__name__)
CORS(app)
logging.info("Flask app initialized.")


# ========================
# Load Model and Metadata
# ========================

MODEL_PATH = os.path.join("model", "best_model.pkl")
METADATA_PATH = os.path.join("model", "model_metadata.pkl")

try:
  model = joblib.load(MODEL_PATH)
  metadata = joblib.load(METADATA_PATH)
  features = metadata["features"]
# conversion_formula = metadata.get("conversion_formula", "np.exp(pred) - 1")
  logging.info(f"Model and metadata loaded. Using features: {features}")
except Exception as e:
  logging.error(f"Error loading model or metadata: {e}")
  raise


@app.route("/")
def index():
  return "Airbnb Price Prediction API is running."

@app.route("/predict", methods=["POST"])
def predict():
  try:
    data = request.get_json()
    logging.info(f"Received input: {data}")

    # Validate input
    missing_features = [f for f in features if f not in data]
    if missing_features:
      return jsonify({
        "error": f"Missing features in request: {missing_features}"
      }), 400

    # Construct input in correct order
    input_df = pd.DataFrame([data], columns=features)
    logging.debug(f"Input DataFrame for prediction:\n{input_df}")

    # Make prediction
    predicted_log_price = model.predict(input_df)[0]
    predicted_price = float(np.exp(predicted_log_price) - 1)
    logging.info(f"Prediction (log): {predicted_log_price}, Converted: ${predicted_price:.2f}")

    return jsonify({
      "predicted_price": f"${round(predicted_price, 2)}"
    })

  except Exception as e:
    return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
  app.run(debug=True)
