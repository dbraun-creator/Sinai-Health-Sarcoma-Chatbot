from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allows frontend to connect from different domain

@app.route('/')
def home():
    return jsonify({"message": "API is running!", "status": "success"})

@app.route('/api/test', methods=['GET'])
def test():
    return jsonify({"message": "Test endpoint works!", "data": [1, 2, 3, 4, 5]})

@app.route('/api/echo', methods=['POST'])
def echo():
    data = request.get_json()
    return jsonify({"received": data, "message": "Echo successful"})

if __name__ == '__main__':
    app.run(debug=True)
