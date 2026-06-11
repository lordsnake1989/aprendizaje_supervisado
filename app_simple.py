from flask import Flask, request, jsonify
import pickle
import numpy as np

modelo = None

with open("modelo.pkl", "rb") as file:
    modelo = pickle.load(file)

app = Flask(__name__)

@app.route("/predecir", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    
    input_data = np.array(data["input"]).reshape(1, -1)
    
    prediccion = modelo.predict(input_data)
    
    return jsonify({"prediccion": int(prediccion[0])})

if __name__ == "__main__":
    app.run(debug=True)
    
    