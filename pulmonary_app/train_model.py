import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load the dataset
df = pd.read_csv('pulmonary_diseases.csv')
print("Columns in dataset:", df.columns.tolist())

# Encode target labels
df['disease'] = df['disease'].astype(str).str.strip()  # Remove extra spaces
label_mapping = {label: idx for idx, label in enumerate(df['disease'].unique())}
df['label'] = df['disease'].map(label_mapping)

# Save label encoder dictionary
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_mapping, f)

# Features and target
X = df.drop(['disease', 'label'], axis=1)
y = df['label']

# Train a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save model
with open('pulmonary_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Accuracy check on train (just for sanity)
accuracy = model.score(X, y)
print(f"Model trained successfully. Accuracy on training data: {accuracy:.4f}")
print("Model and label encoder saved successfully.")
