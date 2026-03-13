import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

# Cough Classes
COUGH_TYPES = ["Healthy Cough", "Dry Cough", "Wet Cough", "Asthma Cough", "Pneumonia Cough"]
CONDITIONS = {
    "Healthy Cough": ["No specific underlying condition", "Normal clearance"],
    "Dry Cough": ["Allergies", "Viral Infection", "Asthma"],
    "Wet Cough": ["Bronchitis", "Pneumonia", "COPD"],
    "Asthma Cough": ["Asthma flare-up", "Reactive airway"],
    "Pneumonia Cough": ["Pneumonia", "Severe Respiratory Infection"]
}

def train_cough_model():
    print("Generating synthetic audio feature dataset for Cough Model...")
    np.random.seed(42)
    X = []
    y = []
    
    # Generate 500 samples per class with slight variance in the 13 MFCC features
    samples_per_class = 500
    for idx, ctype in enumerate(COUGH_TYPES):
        # Base vector
        base = np.random.randn(13) * (idx + 1)
        
        for _ in range(samples_per_class):
            noise = np.random.randn(13) * 0.5
            feats = base + noise
            X.append(feats)
            y.append(idx)
            
    X = np.array(X)
    y = np.array(y)
    
    print(f"Training RandomForest Classifier on {len(X)} samples...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    
    score = clf.score(X, y)
    print(f"Training accuracy: {score:.4f}")
    
    # Save Model
    joblib.dump(clf, "cough_model.pkl")
    print("Saved 'cough_model.pkl'.")

if __name__ == "__main__":
    train_cough_model()
