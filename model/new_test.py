import pickle
import json
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Define model paths
model_path = "model/"
dataset_path = "model/dataset.json"  # Using JSON dataset

# Load Trained Models & Encoders
time_model = pickle.load(open(model_path + "time_prediction_model.pkl", "rb"))
cost_model = pickle.load(open(model_path + "cost_prediction_model.pkl", "rb"))
label_enc_module = pickle.load(open(model_path + "module_name_encoder.pkl", "rb"))
label_enc_lang = pickle.load(open(model_path + "language_encoder.pkl", "rb"))

# Load Dataset for Reference
try:
    with open(dataset_path, "r") as f:
        data = json.load(f)
    df = [entry["Module Name"].lower() for entry in data]  # Get module names in lowercase
    technologies = [entry["Technology"].lower() for entry in data]  # Get technologies in lowercase
except (FileNotFoundError, json.JSONDecodeError):
    print("Error: JSON dataset not found or invalid format.")
    exit()

# Function to Predict Time & Cost
def predict_time_cost(module_name, technology):
    module_name = module_name.lower()
    technology = technology.lower()

    # Encode Inputs
    if module_name in df:
        module_encoded = label_enc_module.transform([module_name])[0]
    else:
        print(f"⚠ Warning: '{module_name}' not found in training data. Using default encoding.")
        module_encoded = len(label_enc_module.classes_)  # Assign new encoding

    if technology in technologies:
        lang_encoded = label_enc_lang.transform([technology])[0]
    else:
        print(f"⚠ Warning: '{technology}' not found in training data. Using default encoding.")
        lang_encoded = len(label_enc_lang.classes_)  # Assign new encoding

    # Make Predictions
    predicted_time = time_model.predict([[module_encoded, lang_encoded]])[0]
    predicted_cost = cost_model.predict([[module_encoded, lang_encoded]])[0]

    # Determine Complexity Level
    complexity = "Small" if predicted_time <= 10 else "Medium" if predicted_time <= 30 else "Large"

    return {
        "Module": module_name,
        "Technology": technology,
        "Predicted Time (hrs)": round(predicted_time, 2),
        "Predicted Cost (INR)": round(predicted_cost, 2),
        "Complexity": complexity
    }

# Example Test Inputs
test_module = "register_user"
test_technology = "php"

# Run Prediction
prediction_result = predict_time_cost(test_module, test_technology)

# Print Results
print("\n🔹 **Prediction Results** 🔹")
print(json.dumps(prediction_result, indent=4))
