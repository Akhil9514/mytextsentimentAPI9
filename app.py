from flask import Flask,request,jsonify
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from preprocess import *



encoder = pickle.load(open('encoder.pkl', 'rb'))
cv = pickle.load(open('CountVectorizer.pkl', 'rb'))

model = tf.keras.models.load_model('my_model.h5')

app = Flask(__name__)


@app.route('/')
def home():
    return "THE BASIC API IS WORKING FINE"

@app.route('/predict', methods=['POST'] )
def predict():

    text = request.form.get('text')
    input = preprocess(text)

    array = cv.transform([input]).toarray()

    pred = model.predict(array)
    a = np.argmax(pred, axis=1)
    prediction = encoder.inverse_transform(a)[0]

    # print(prediction)
    # print(text)
    # print(type(text))


    result = prediction

    print(result)
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)



