from flask import Flask, request
from flask_cors import CORS, cross_origin
import traceback
import sys
import joblib
import numpy as np
import os

app = Flask(__name__)
CORS(app)

model = None

@app.route('/')
def helloworld():
    return 'Hello World'

@app.route('/area', methods=['GET'])
@cross_origin()
def area():
    w = float(request.values['w'])
    h = float(request.values['h'])
    return str(w*h)

@app.route('/bmi', methods=['GET'])
@cross_origin()
def bmi():
    w = float(request.values['w'])
    h = float(request.values['h'])
    return str(w / ((h/100)**2))

@app.route('/iris', methods=['POST'])
@cross_origin()
def predict_species():
    req = request.values['param']
    inputs = np.array(req.split(','), dtype=np.float32).reshape(1,-1)
    predict_target = model.predict(inputs)
    if predict_target == 0:
        return 'Setosa'
    elif predict_target == 1:
        return 'Versicolour'
    else:
        return 'Virginica'

if __name__ == '__main__':
    try:
        # Load model
        model = joblib.load('iris.model')

        port = int(os.environ.get('PORT', 5000))
        app.run(host='0.0.0.0', port=port, debug=True)
    except Exception as ex:
        traceback.print_exc(file=sys.stdout)