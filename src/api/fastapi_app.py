from flask import Flask, request, jsonify
from flask_cors import CORS
from prediction.predictor_api import predict_from_json

app = Flask(__name__)
CORS(app)  # Allow all origins, methods, headers

@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_data = request.get_json()
        print(input_data)

        # Pass data directly to your prediction function
        data = predict_from_json(input_data)
        print("Data =", data)

        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# 🔹 Main method to run the Flask server
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9000, debug=True)
