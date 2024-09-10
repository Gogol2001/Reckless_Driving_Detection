import csv
from flask import Flask, request, jsonify, render_template, redirect, url_for
import joblib
import pandas as pd
import warnings

app = Flask(__name__)

# Suppress the warning about feature names mismatch
warnings.filterwarnings("ignore", message="X has feature names, but RandomForestClassifier was fitted without feature names")

# Load the saved model
model = joblib.load('models/random_forest_model.joblib')

# Define a route for the root URL
@app.route('/')
def index():
    return render_template('index.html')

# Define a function to check user credentials from CSV
def check_credentials(email, password):
    with open('users.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if row[0] == email and row[1] == password:
                return True
    return False

# Define a route for the login page
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        if check_credentials(email, password):
            return redirect(url_for('index'))  # Redirect to the homepage on successful login
        else:
            return render_template('login.html', message='Invalid email or password')
    return render_template('login.html')

# Define a route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Log the received data
        print("Received data:", request.json)
        
        # Get input data from the request
        data = request.json
        acc_x = data.get('acc_x')
        acc_y = data.get('acc_y')
        acc_z = data.get('acc_z')
        gyro_x = data.get('gyro_x')
        gyro_y = data.get('gyro_y')
        gyro_z = data.get('gyro_z')
        
        # Log the input values
        print("Received input:", acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z)
        
        # Check if any input field is empty
        if not all([acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]):
            return jsonify({'error': 'All fields are required'}), 400
        
        # Convert input data to float
        acc_x = float(acc_x)
        acc_y = float(acc_y)
        acc_z = float(acc_z)
        gyro_x = float(gyro_x)
        gyro_y = float(gyro_y)
        gyro_z = float(gyro_z)
        
        # Prepare input data as a DataFrame
        input_data = {'acc_x': [acc_x], 'acc_y': [acc_y], 'acc_z': [acc_z],
                      'gyro_x': [gyro_x], 'gyro_y': [gyro_y], 'gyro_z': [gyro_z]}
        input_df = pd.DataFrame(input_data)
        
        # Make predictions
        predictions = model.predict(input_df)
        
        # Convert predictions to string ('Reckless' or 'Not Reckless')
        result = "Reckless" if predictions[0] == 1 else "Not Reckless"
        
        # Return the prediction result
        return jsonify({'prediction': result})
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': 'Invalid input data'}), 500

# Define a route for the result page
@app.route('/result')
def result():
    prediction = request.args.get('prediction')
    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
