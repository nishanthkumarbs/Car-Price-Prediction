import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from flask import Flask, request, render_template
import h5py
import pickle
import numpy as np

app = Flask(__name__)

# Load the model and scaler once when the application starts
with h5py.File('models/train_test_split.h5', 'r') as h5f:
    model_data = h5f['model'][()]
    rf = pickle.loads(model_data)
    X_train_data = h5f['X_train'][:]
    scaler = StandardScaler()
    scaler.fit(X_train_data)

@app.route('/', methods=['GET', 'POST'])
def predict():
    result = None
    if request.method == 'POST':
        try:
            # Retrieve values from form
            Mileage = int(request.form['Mileage'])
            Age_yrs = int(request.form['Age(yrs)'])

            # Create a dataframe for features with the correct column names
            features = pd.DataFrame([[Mileage, Age_yrs]], columns=['Mileage', 'Age(yrs)'])

            # Scale features
            features_scaled = scaler.transform(features)

            # Make prediction
            prediction = rf.predict(features_scaled)

            # Interpret result
            result = f"Predicted Sell Price: ${prediction[0]:,.2f}"
        except ValueError:
            result = "Please enter valid values in all fields"

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)

