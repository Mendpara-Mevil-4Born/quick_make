import os
import json
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Define dataset and model save paths
DATASET_PATH = "model/dataset.json"
MODEL_SAVE_PATH = "model/"

# Ensure the model directory exists
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

def load_dataset():
    """Loads dataset from JSON file and converts it into a DataFrame."""
    try:
        with open(DATASET_PATH, "r") as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        return df
    except (FileNotFoundError, json.JSONDecodeError):
        print("Error: JSON dataset not found or invalid format.")
        return None

def preprocess_data(df):
    """Encodes categorical features and prepares training data."""
    df["Module Name"] = df["Module Name"].str.lower()
    
    # Clean technology format - remove brackets if present
    df["Technology"] = df["Technology"].astype(str).apply(lambda x: x.strip())
    df["Technology"] = df["Technology"].apply(lambda x: x[1:-1].strip() if x.startswith('[') and x.endswith(']') else x)
    df["Technology"] = df["Technology"].str.lower()

    label_enc_module = LabelEncoder()
    label_enc_lang = LabelEncoder()

    df["Module Name"] = label_enc_module.fit_transform(df["Module Name"])
    df["Technology"] = label_enc_lang.fit_transform(df["Technology"])

    X = df[["Module Name", "Technology"]]
    y_time = df["Time"]
    y_cost = df["Cost"]

    return X, y_time, y_cost, label_enc_module, label_enc_lang

def split_data(X, y_time, y_cost):
    """Splits data into training and testing sets."""
    X_train, X_test, y_train_time, y_test_time, y_train_cost, y_test_cost = train_test_split(
        X, y_time, y_cost, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train_time, y_test_time, y_train_cost, y_test_cost

def train_model(X_train, y_train):
    """Trains a RandomForestRegressor model and returns it."""
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, metric_name):
    """Evaluates model performance using Mean Absolute Error."""
    y_pred = model.predict(X_test)
    error = mean_absolute_error(y_test, y_pred)
    print(f"[SUCCESS] {metric_name} Prediction Mean Absolute Error: {error:.2f}")
    return error

def save_model(model, filepath):
    """Saves the trained model to disk."""
    with open(filepath, "wb") as f:
        pickle.dump(model, f)

def load_user_dataset(user_id):
    """Loads user-specific dataset from JSON file and converts it into a DataFrame."""
    user_file_path = os.path.join('user_data', str(user_id), f'{user_id}.json')
    try:
        with open(user_file_path, "r") as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        return df
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"Error: User dataset for user_id {user_id} not found or invalid format.")
        return None

def main(user_id=None):
    """Main function to execute the training pipeline."""
    if user_id:
        df = load_user_dataset(user_id)
    else:
        df = load_dataset()

    if df is None:
        return

    X, y_time, y_cost, label_enc_module, label_enc_lang = preprocess_data(df)
    X_train, X_test, y_train_time, y_test_time, y_train_cost, y_test_cost = split_data(X, y_time, y_cost)

    # Train models
    time_model = train_model(X_train, y_train_time)
    cost_model = train_model(X_train, y_train_cost)

    # Evaluate models
    evaluate_model(time_model, X_test, y_test_time, "Time")
    evaluate_model(cost_model, X_test, y_test_cost, "Cost")

    # Ensure the user-specific model directory exists
    if user_id:
        user_model_path = os.path.join('user_data', str(user_id))
        os.makedirs(user_model_path, exist_ok=True)
        logging.info(f"Saving models to {user_model_path}")
        save_model(time_model, os.path.join(user_model_path, "time_prediction_model.pkl"))
        save_model(cost_model, os.path.join(user_model_path, "cost_prediction_model.pkl"))
        save_model(label_enc_module, os.path.join(user_model_path, "module_name_encoder.pkl"))
        save_model(label_enc_lang, os.path.join(user_model_path, "language_encoder.pkl"))
    else:
        logging.info("Saving models to default model directory")
        save_model(time_model, "time_prediction_model.pkl")
        save_model(cost_model, "cost_prediction_model.pkl")
        save_model(label_enc_module, "module_name_encoder.pkl")
        save_model(label_enc_lang, "language_encoder.pkl")

    print("[SUCCESS] Models and encoders saved successfully.")

if __name__ == "__main__":
    # Example usage: main(user_id="5")
    main()
