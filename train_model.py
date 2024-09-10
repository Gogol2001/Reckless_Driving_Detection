import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

# Load your dataset (assuming it's in a CSV file)
data = pd.read_csv('data/FinalDatasetCsv.csv')

# Separate features and target variable
X = data.drop(columns=['label'])
y = data['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize and train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Random Forest Accuracy: {accuracy:.2f}')

# Specify the directory where you want to save the model
model_directory = 'models'
os.makedirs(model_directory, exist_ok=True)

# Define the file path for saving the model
model_path = os.path.join(model_directory, 'random_forest_model.joblib')

# Save the trained Random Forest model
joblib.dump(rf_model, model_path)

# Function to load the saved model and make predictions
def predict_with_random_forest(input_data):
    # Load the saved model
    loaded_model = joblib.load('random_forest_model.joblib')

    # Standardize the input data using the same scaler used during training
    scaled_input = scaler.transform(input_data)

    # Make predictions
    predictions = loaded_model.predict(scaled_input)

    return predictions

# Example input data for prediction
new_data = pd.DataFrame({
    'Acc_Magnitude': [-0.843144092767758],
    'Gyro_Magnitude': [-0.64903539618777],
    'Gyro_Y_Std': [4.00866243844088E-16],
    'Acc_X_Kurtosis': [0.0],
    'Acc_X_Entropy': [0.0],
    'Gyro_Y_Entropy': [-1.46048267992983E-15],
    'Acc_X_Range': [0.0],
    'Gyro_Y_Range': [0.0],
    'Gyro_Y_Autocorrelation_Lag1': [-4.53961122344393E-17],
    'Acc_X_Mean_Absolute_Deviation': [0.0]
})

# Use the predict_with_random_forest function to make predictions
predicted_labels = predict_with_random_forest(new_data)

# Display the predictions
print("Predicted Labels:", predicted_labels)

# Get feature importances
feature_importances = rf_model.feature_importances_

# Create a DataFrame to display feature importances
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
})

# Sort the DataFrame by importance in descending order
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Display the feature importances
print("\nFeature Importance:")
print(feature_importance_df)
