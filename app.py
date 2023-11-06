from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)
filename = 'file_StudentPerformance.pkl'
model = pickle.load(open(filename, 'rb'))    # load the model
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])  # The user input is processed here
def predict():
    Math_Score = request.form['math_score']
    Reading_Score = request.form['reading_score']
    Writing_Score = request.form['writing_score']
    pred = model.predict(np.array([[float(Math_Score), float(Reading_Score), float(Writing_Score)]]))    #print(pred)
    return render_template('index.html', predict=str(pred))
if __name__ == '__main__':
    app.run(host="127.0.0.9",port=8080, debug=True)


