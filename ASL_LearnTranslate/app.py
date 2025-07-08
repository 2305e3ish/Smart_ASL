from flask import Flask, request, jsonify
from flask_cors import CORS
import onnxruntime as ort
import numpy as np

app = Flask(__name__)
CORS(app)

session = ort.InferenceSession("ml/model.onnx")

@app.route('/predict', methods=['POST'])
def predict():
    data = np.array(request.json["landmarks"]).astype(np.float32)
    outputs = session.run(None, {"input": data})
    pred = int(np.argmax(outputs[0]))
    return jsonify({"letter": chr(pred + ord("A"))})
