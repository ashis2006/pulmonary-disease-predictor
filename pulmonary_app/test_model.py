import pickle
import numpy as np

# Load model and encoder
with open('pulmonary_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('label_encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)
rev_encoder = {v: k for k, v in encoder.items()}

# Get feature names from the model
features = list(model.feature_names_in_)

def test_prediction(symptoms_list, description):
    print(f"\n--- Testing: {description} ---")
    vals = [1 if f in symptoms_list else 0 for f in features]
    X = np.array(vals).reshape(1, -1)
    
    probs = model.predict_proba(X)[0]
    top3_indices = np.argsort(probs)[::-1][:3]
    
    for idx in top3_indices:
        prob = probs[idx] * 100
        disease = rev_encoder[idx]
        print(f"{disease}: {prob:.2f}%")

test_prediction(["Fever"], "Only fever")
test_prediction(["Blood in sputum (Hemoptysis)", "Fever", "Night sweats", "Weight loss"], "TB / Lung Cancer symptoms")
test_prediction(["Shortness of breath", "Wheezing", "Chest tightness"], "Asthma / COPD symptoms")
