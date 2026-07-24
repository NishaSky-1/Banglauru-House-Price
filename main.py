from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

# Data aur Model Load karo
data = pd.read_csv('Cleaned_data.csv')
pipe = pickle.load(open("RidgeModel.pkl", 'rb'))

@app.route('/')
def index():
    locations = sorted(data['location'].unique())
    return render_template('index.html', locations=locations)

@app.route('/predict', methods=['POST'])
def predict():
    area = float(request.form.get('area'))
    bhk = int(request.form.get('bhk'))
    bath = int(request.form.get('bath'))
    location = request.form.get('location')

    total_sqft = area
    price_per_feet = 5000
    
    input_data = pd.DataFrame([[area, bhk, bath, location, total_sqft, price_per_feet]], 
                            columns=['area', 'bhk', 'bath', 'location', 'total_sqft', 'price_per_feet'])
    
    prediction = pipe.predict(input_data)[0] 
    prediction = round(prediction, 2)

    print("Prediction:", prediction)
    
    locations = sorted(data['location'].unique())
    return render_template('index.html', prediction=prediction, locations=locations)

if __name__ == "__main__":
    app.run(debug=True)