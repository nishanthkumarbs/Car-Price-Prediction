import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import h5py
import pickle
import numpy as np

# Load the dataset
df = pd.read_csv('data\carprices.csv')
print(df)

# Split data into training and test sets
X = df[['Mileage', 'Age(yrs)']]
y = df['Sell Price($)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a Random Forest Regressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predict and display the mean squared error
y_pred = rf.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Save the model and data splits into an HDF5 file
hdf5_path = 'models/train_test_split.h5'
with h5py.File(hdf5_path, 'w') as h5f:
    h5f.create_dataset('X_train', data=X_train)
    h5f.create_dataset('X_test', data=X_test)
    h5f.create_dataset('y_train', data=y_train)
    h5f.create_dataset('y_test', data=y_test)
    model_data = pickle.dumps(rf)
    h5f.create_dataset('model', data=np.void(model_data))

print(f"Data and model saved to {hdf5_path}")

