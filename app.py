# Importing essential libraries and modules

from flask import Flask, render_template, request
import numpy as np
import joblib
from warnings import filterwarnings
filterwarnings('ignore')

# Load the model
loaded_model = joblib.load('model/rf.pkl')
model = joblib.load('model/regression_model.pkl')

def preprocess_input(commodity_index, month_index, year):
    return np.array([[commodity_index, month_index, year]])

def predict_price(model, input_data):
    print(input_data)
    return model.predict(input_data)[0]




app = Flask(__name__)

# render home page
@ app.route('/')
def home():
    title = 'Crop harvest'
    return render_template('index.html', title=title)

# render crop recommendation form page
@ app.route('/crop_Analysis')
def crop_Analysis():
    
    title = 'Crop Recommendation'
    # , n=n, p=p, k=k, temp=temp)
    return render_template('crop.html', title=title)

# render 

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        Temperature = request.form['Temperature']
        humidity = request.form['humidity']
        ph = request.form['ph']
        water = request.form['water']
        season = request.form['season']
        soill = request.form['soill']
        month_index = request.form['mon']
        year = request.form['yearr']

        # Construct the input array
        new_input = np.array([[Temperature, humidity, ph, water, season, soill]])

        # Use the loaded model to predict probabilities of each class
        predicted_probabilities = loaded_model.predict_proba(new_input)

        # Get the indices of the top two predicted classes
        top_two_indices = predicted_probabilities.argsort()[0][-2:]

        # Map the indices back to crop names using the label mapping
        label_mapping = ['blackgram','chickpea','coconut','groundnut','maize','mungbean','pigeonpeas','ragi','rice']
        crops = [label_mapping[index] for index in top_two_indices]

        print(f"Top two recommended crops: \n 1) {crops[0]} \n 2) {crops[1]}")

        # CROP 1 PRICE
        cp1=crops[0]
        input_data1 = preprocess_input(top_two_indices[0], month_index, year)
        predicted_price1 = predict_price(model, input_data1)
        print(f"Predicted price for commodity index {top_two_indices[0]} in month index {month_index} {year}: {predicted_price1}")

        # CROP 2 PRICE
        cp2=crops[1]
        input_data2 = preprocess_input(top_two_indices[1], month_index, year)
        predicted_price2 = predict_price(model, input_data2)
        print(f"Predicted price for commodity index {top_two_indices[1]} in month index {month_index} {year}: {predicted_price2}")

        if predicted_price1 > predicted_price2:
            res=cp1
        else:
            res=cp2
        return render_template('crop-result.html',cp1=cp1,cp2=cp2,predicted_price1=predicted_price1,predicted_price2=predicted_price2,res=res)
    return render_template('crop.html')

if __name__ == '__main__':
    app.run(debug=True)
