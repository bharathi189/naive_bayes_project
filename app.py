from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
with open("navie_bayes.pkl", "rb") as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    sl = float(request.form['sepal_length'])
    sw = float(request.form['sepal_width'])
    pl = float(request.form['petal_length'])
    pw = float(request.form['petal_width'])

    features = np.array([[sl, sw, pl, pw]])

    prediction = model.predict(features)[0]

    species = {
        0: "Iris Setosa",
        1: "Iris Versicolor",
        2: "Iris Virginica"
    }

    result = species[prediction]

    return render_template(
        'index.html',
        prediction_text=f'Predicted Flower Species: {result}'
    )

if __name__ == '__main__':
    app.run(debug=True)