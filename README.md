# Car Price Prediction

This project is a web application that predicts the selling price of a car based on its mileage and age using a machine learning model. The application is built using Flask for the web interface and scikit-learn for the machine learning model.

#### Supervised by 
Prof. Agughasi Victor Ikechukwu, 
(Assistant Professor) 
Department of CSE, MIT Mysore

## Project Structure

- `model.py`: Script to train the RandomForestRegressor model and save it along with the scaler in an HDF5 file.
- `app.py`: Flask application script to serve the web interface and make predictions.
- `templates/index.html`: HTML template for the web interface.
- `model/train_test_split.h5`: HDF5 file containing the trained model and scaler (generated by `model.py`).


```
Car Price Prediction/
│
├── models/
│   └── train_test_split.h5
│
├── templates/
│   └── index.html
│
├── app.py
└── model.py
```
## Project Files

**model.py**
- This script trains a RandomForestRegressor model on the car prices dataset and saves the trained model and scaler in an HDF5 file.

**app.py**
- This is the Flask application script. It loads the trained model and scaler from the HDF5 file and serves the web interface for making predictions.

**templates/index.html**
- This HTML file contains the web interface for the car price prediction form. It includes styling and animations for a better user experience.

**train_test_split.h5**
- This HDF5 file contains the trained RandomForestRegressor model and the scaler for feature scaling. It is generated by running the train_model.py script.


## Setup Instructions

## Prerequisites

- Python 3.10.10
- Required Python packages:
  - pandas
  - scikit-learn
  - h5py
  - flask
  - pickle
  - numpy

## Setup and Usage

### Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/nishanthkumarbs/Car-Price-Prediction.git
    cd Car-Price-Prediction
    ```

2. **Create a virtual environment**:
    ```bash
    # On Windows use
    python -m venv venv
    venv\Scripts\activate
    ```

3. **Install the required packages**:
    ```bash
    pip install pandas scikit-learn h5py flask nump
    ```

4. **Ensure the model file exists**:
    Make sure `train_test_split.h5` is present in the `models` directory.

## Usage

1. **Run the Flask application**:
    ```bash
    python app.py
    ```

2. **Access the application**:
    Open your web browser and go to `http://127.0.0.1:5000/`.

3. **Enter the required details**:
    Fill in the form with the necessary details and click on the "Predict" button to get the prediction.



## Screenshots
![image](https://github.com/user-attachments/assets/c0aa9296-19c9-41de-97fc-c1bb5b052514)

## Conclusion

This project demonstrates how to build a car price prediction web application using Flask and scikit-learn. The model predicts car prices in rupees based on mileage and age. By following the setup instructions, you can train the model and deploy the web application locally. This project serves as a practical example of integrating machine learning with web development.


## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.