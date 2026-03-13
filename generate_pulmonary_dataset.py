import pandas as pd
import random

# Define diseases and symptom likelihoods (1 = very common, 0 = rare)
diseases = {
    "Pneumonia": {
        "Cough": 1.0, "Dry cough": 0.2, "Chronic cough": 0.1, "Chest pain": 0.9, "Shortness of breath": 0.9, 
        "Fatigue": 0.8, "Fever": 1.0, "Weight loss": 0.3, "Wheezing": 0.4, "Night sweats": 0.4, 
        "Sputum production": 1.0, "Blood in sputum (Hemoptysis)": 0.2, "Rapid breathing": 0.8, 
        "Chest tightness": 0.7, "Difficulty breathing during activity": 0.9, "Bluish lips": 0.4, 
        "Hoarseness": 0.1, "Sore throat": 0.2, "Nasal congestion": 0.2, "Dizziness": 0.3, 
        "Swelling in legs": 0.1, "Loss of appetite": 0.7, "Throat irritation": 0.3, "Lung crackles": 1.0,
        "Shallow breathing": 0.7, "Muscle aches": 0.6, "Chills": 0.8, "Confusion (in elderly)": 0.5,
        "Nausea": 0.4, "Headache": 0.5, "Cyanosis": 0.3, "Sweating": 0.6
    },
    "Tuberculosis": {
        "Cough": 0.9, "Dry cough": 0.6, "Chronic cough": 1.0, "Chest pain": 0.7, "Shortness of breath": 0.5, 
        "Fatigue": 1.0, "Fever": 0.9, "Weight loss": 1.0, "Wheezing": 0.2, "Night sweats": 1.0, 
        "Sputum production": 0.8, "Blood in sputum (Hemoptysis)": 0.7, "Rapid breathing": 0.4, 
        "Chest tightness": 0.6, "Difficulty breathing during activity": 0.6, "Bluish lips": 0.1, 
        "Hoarseness": 0.3, "Sore throat": 0.2, "Nasal congestion": 0.1, "Dizziness": 0.4, 
        "Swelling in legs": 0.1, "Loss of appetite": 1.0, "Throat irritation": 0.4, "Lung crackles": 0.5,
        "Shallow breathing": 0.3, "Muscle aches": 0.5, "Chills": 0.8, "Confusion (in elderly)": 0.1,
        "Nausea": 0.3, "Headache": 0.4, "Cyanosis": 0.1, "Sweating": 0.9
    },
    "COVID-19 pneumonia": {
        "Cough": 0.9, "Dry cough": 1.0, "Chronic cough": 0.3, "Chest pain": 0.8, "Shortness of breath": 0.9, 
        "Fatigue": 1.0, "Fever": 1.0, "Weight loss": 0.4, "Wheezing": 0.3, "Night sweats": 0.5, 
        "Sputum production": 0.2, "Blood in sputum (Hemoptysis)": 0.1, "Rapid breathing": 0.8, 
        "Chest tightness": 0.9, "Difficulty breathing during activity": 0.9, "Bluish lips": 0.5, 
        "Hoarseness": 0.4, "Sore throat": 0.8, "Nasal congestion": 0.7, "Dizziness": 0.6, 
        "Swelling in legs": 0.1, "Loss of appetite": 0.8, "Throat irritation": 0.7, "Lung crackles": 0.8,
        "Shallow breathing": 0.7, "Muscle aches": 0.9, "Chills": 0.8, "Confusion (in elderly)": 0.4,
        "Nausea": 0.5, "Headache": 0.8, "Cyanosis": 0.4, "Sweating": 0.6, "Loss of taste/smell": 0.9
    },
    "Bronchitis": {
        "Cough": 1.0, "Dry cough": 0.4, "Chronic cough": 0.8, "Chest pain": 0.6, "Shortness of breath": 0.6, 
        "Fatigue": 0.8, "Fever": 0.5, "Weight loss": 0.1, "Wheezing": 0.8, "Night sweats": 0.3, 
        "Sputum production": 1.0, "Blood in sputum (Hemoptysis)": 0.1, "Rapid breathing": 0.4, 
        "Chest tightness": 0.8, "Difficulty breathing during activity": 0.7, "Bluish lips": 0.1, 
        "Hoarseness": 0.5, "Sore throat": 0.7, "Nasal congestion": 0.8, "Dizziness": 0.2, 
        "Swelling in legs": 0.1, "Loss of appetite": 0.3, "Throat irritation": 0.8, "Lung crackles": 0.4,
        "Shallow breathing": 0.3, "Muscle aches": 0.4, "Chills": 0.3, "Confusion (in elderly)": 0.1,
        "Nausea": 0.1, "Headache": 0.5, "Cyanosis": 0.1, "Sweating": 0.2
    },
    "Asthma": {
        "Cough": 0.8, "Dry cough": 0.7, "Chronic cough": 0.6, "Chest pain": 0.6, "Shortness of breath": 1.0, 
        "Fatigue": 0.7, "Fever": 0.1, "Weight loss": 0.1, "Wheezing": 1.0, "Night sweats": 0.2, 
        "Sputum production": 0.4, "Blood in sputum (Hemoptysis)": 0.0, "Rapid breathing": 0.8, 
        "Chest tightness": 0.9, "Difficulty breathing during activity": 0.9, "Bluish lips": 0.3, 
        "Hoarseness": 0.2, "Sore throat": 0.2, "Nasal congestion": 0.5, "Dizziness": 0.4, 
        "Swelling in legs": 0.1, "Loss of appetite": 0.2, "Throat irritation": 0.4, "Lung crackles": 0.2,
        "Shallow breathing": 0.6, "Muscle aches": 0.2, "Chills": 0.1, "Confusion (in elderly)": 0.1,
        "Nausea": 0.1, "Headache": 0.3, "Cyanosis": 0.2, "Sweating": 0.4
    },
    "COPD": {
        "Cough": 0.9, "Dry cough": 0.3, "Chronic cough": 1.0, "Chest pain": 0.6, "Shortness of breath": 1.0, 
        "Fatigue": 0.9, "Fever": 0.3, "Weight loss": 0.6, "Wheezing": 0.9, "Night sweats": 0.2, 
        "Sputum production": 0.9, "Blood in sputum (Hemoptysis)": 0.2, "Rapid breathing": 0.7, 
        "Chest tightness": 0.8, "Difficulty breathing during activity": 1.0, "Bluish lips": 0.6, 
        "Hoarseness": 0.3, "Sore throat": 0.3, "Nasal congestion": 0.4, "Dizziness": 0.5, 
        "Swelling in legs": 0.5, "Loss of appetite": 0.5, "Throat irritation": 0.5, "Lung crackles": 0.5,
        "Shallow breathing": 0.8, "Muscle aches": 0.3, "Chills": 0.2, "Confusion (in elderly)": 0.3,
        "Nausea": 0.2, "Headache": 0.6, "Cyanosis": 0.7, "Sweating": 0.4
    },
    "Pulmonary Fibrosis": {
        "Cough": 0.9, "Dry cough": 1.0, "Chronic cough": 0.9, "Chest pain": 0.5, "Shortness of breath": 1.0, 
        "Fatigue": 1.0, "Fever": 0.1, "Weight loss": 0.6, "Wheezing": 0.2, "Night sweats": 0.1, 
        "Sputum production": 0.1, "Blood in sputum (Hemoptysis)": 0.1, "Rapid breathing": 0.8, 
        "Chest tightness": 0.6, "Difficulty breathing during activity": 1.0, "Bluish lips": 0.4, 
        "Hoarseness": 0.1, "Sore throat": 0.1, "Nasal congestion": 0.1, "Dizziness": 0.4, 
        "Swelling in legs": 0.4, "Loss of appetite": 0.6, "Throat irritation": 0.2, "Lung crackles": 0.9,
        "Shallow breathing": 0.9, "Muscle aches": 0.5, "Chills": 0.1, "Confusion (in elderly)": 0.2,
        "Nausea": 0.2, "Headache": 0.3, "Cyanosis": 0.5, "Sweating": 0.2, "Clubbing of fingers": 0.8
    },
    "Lung Cancer": {
        "Cough": 0.9, "Dry cough": 0.6, "Chronic cough": 0.9, "Chest pain": 0.8, "Shortness of breath": 0.8, 
        "Fatigue": 0.9, "Fever": 0.3, "Weight loss": 0.9, "Wheezing": 0.6, "Night sweats": 0.4, 
        "Sputum production": 0.6, "Blood in sputum (Hemoptysis)": 0.9, "Rapid breathing": 0.5, 
        "Chest tightness": 0.7, "Difficulty breathing during activity": 0.8, "Bluish lips": 0.2, 
        "Hoarseness": 0.8, "Sore throat": 0.4, "Nasal congestion": 0.1, "Dizziness": 0.4, 
        "Swelling in legs": 0.3, "Loss of appetite": 0.9, "Throat irritation": 0.5, "Lung crackles": 0.4,
        "Shallow breathing": 0.6, "Muscle aches": 0.5, "Chills": 0.2, "Confusion (in elderly)": 0.3,
        "Nausea": 0.4, "Headache": 0.5, "Cyanosis": 0.2, "Sweating": 0.3, "Bone pain": 0.6, "Clubbing of fingers": 0.4
    },
    "Emphysema": {
        "Cough": 0.7, "Dry cough": 0.6, "Chronic cough": 0.8, "Chest pain": 0.4, "Shortness of breath": 1.0, 
        "Fatigue": 0.9, "Fever": 0.1, "Weight loss": 0.7, "Wheezing": 0.8, "Night sweats": 0.1, 
        "Sputum production": 0.4, "Blood in sputum (Hemoptysis)": 0.1, "Rapid breathing": 0.9, 
        "Chest tightness": 0.7, "Difficulty breathing during activity": 1.0, "Bluish lips": 0.6, 
        "Hoarseness": 0.2, "Sore throat": 0.2, "Nasal congestion": 0.2, "Dizziness": 0.5, 
        "Swelling in legs": 0.4, "Loss of appetite": 0.6, "Throat irritation": 0.3, "Lung crackles": 0.3,
        "Shallow breathing": 0.9, "Muscle aches": 0.3, "Chills": 0.1, "Confusion (in elderly)": 0.4,
        "Nausea": 0.2, "Headache": 0.6, "Cyanosis": 0.7, "Sweating": 0.3, "Barrel chest": 0.9
    },
    "Pulmonary Hypertension": {
        "Cough": 0.4, "Dry cough": 0.5, "Chronic cough": 0.3, "Chest pain": 0.8, "Shortness of breath": 1.0, 
        "Fatigue": 1.0, "Fever": 0.1, "Weight loss": 0.3, "Wheezing": 0.2, "Night sweats": 0.1, 
        "Sputum production": 0.1, "Blood in sputum (Hemoptysis)": 0.3, "Rapid breathing": 0.7, 
        "Chest tightness": 0.7, "Difficulty breathing during activity": 1.0, "Bluish lips": 0.7, 
        "Hoarseness": 0.3, "Sore throat": 0.1, "Nasal congestion": 0.1, "Dizziness": 0.9, 
        "Swelling in legs": 0.9, "Loss of appetite": 0.6, "Throat irritation": 0.1, "Lung crackles": 0.2,
        "Shallow breathing": 0.6, "Muscle aches": 0.3, "Chills": 0.1, "Confusion (in elderly)": 0.3,
        "Nausea": 0.4, "Headache": 0.4, "Cyanosis": 0.8, "Sweating": 0.3, "Fainting (Syncope)": 0.8
    },
    "Pulmonary Edema": {
        "Cough": 0.9, "Dry cough": 0.2, "Chronic cough": 0.3, "Chest pain": 0.6, "Shortness of breath": 1.0, 
        "Fatigue": 0.9, "Fever": 0.2, "Weight loss": 0.1, "Wheezing": 0.8, "Night sweats": 0.6, 
        "Sputum production": 0.9, "Blood in sputum (Hemoptysis)": 0.6, "Rapid breathing": 0.9, 
        "Chest tightness": 0.8, "Difficulty breathing during activity": 1.0, "Bluish lips": 0.6, 
        "Hoarseness": 0.3, "Sore throat": 0.1, "Nasal congestion": 0.2, "Dizziness": 0.7, 
        "Swelling in legs": 0.8, "Loss of appetite": 0.5, "Throat irritation": 0.3, "Lung crackles": 1.0,
        "Shallow breathing": 0.8, "Muscle aches": 0.2, "Chills": 0.2, "Confusion (in elderly)": 0.6,
        "Nausea": 0.5, "Headache": 0.4, "Cyanosis": 0.7, "Sweating": 0.9, "Pink frothy sputum": 0.9
    },
    "Pleural Effusion": {
        "Cough": 0.8, "Dry cough": 0.9, "Chronic cough": 0.4, "Chest pain": 0.9, "Shortness of breath": 0.9, 
        "Fatigue": 0.8, "Fever": 0.5, "Weight loss": 0.4, "Wheezing": 0.2, "Night sweats": 0.4, 
        "Sputum production": 0.2, "Blood in sputum (Hemoptysis)": 0.1, "Rapid breathing": 0.7, 
        "Chest tightness": 0.8, "Difficulty breathing during activity": 0.9, "Bluish lips": 0.3, 
        "Hoarseness": 0.1, "Sore throat": 0.1, "Nasal congestion": 0.1, "Dizziness": 0.3, 
        "Swelling in legs": 0.2, "Loss of appetite": 0.5, "Throat irritation": 0.2, "Lung crackles": 0.4,
        "Shallow breathing": 0.8, "Muscle aches": 0.4, "Chills": 0.4, "Confusion (in elderly)": 0.2,
        "Nausea": 0.3, "Headache": 0.4, "Cyanosis": 0.3, "Sweating": 0.4, "Hiccups": 0.4
    },
    "Pneumothorax": {
        "Cough": 0.3, "Dry cough": 0.5, "Chronic cough": 0.1, "Chest pain": 1.0, "Shortness of breath": 1.0, 
        "Fatigue": 0.8, "Fever": 0.1, "Weight loss": 0.1, "Wheezing": 0.1, "Night sweats": 0.1, 
        "Sputum production": 0.1, "Blood in sputum (Hemoptysis)": 0.1, "Rapid breathing": 0.9, 
        "Chest tightness": 0.9, "Difficulty breathing during activity": 1.0, "Bluish lips": 0.6, 
        "Hoarseness": 0.1, "Sore throat": 0.1, "Nasal congestion": 0.1, "Dizziness": 0.6, 
        "Swelling in legs": 0.1, "Loss of appetite": 0.3, "Throat irritation": 0.1, "Lung crackles": 0.1,
        "Shallow breathing": 0.9, "Muscle aches": 0.3, "Chills": 0.1, "Confusion (in elderly)": 0.4,
        "Nausea": 0.3, "Headache": 0.4, "Cyanosis": 0.7, "Sweating": 0.7, "Sharp chest pain": 1.0
    },
    "Sarcoidosis": {
        "Cough": 0.8, "Dry cough": 1.0, "Chronic cough": 0.7, "Chest pain": 0.6, "Shortness of breath": 0.8, 
        "Fatigue": 0.9, "Fever": 0.4, "Weight loss": 0.6, "Wheezing": 0.5, "Night sweats": 0.5, 
        "Sputum production": 0.2, "Blood in sputum (Hemoptysis)": 0.1, "Rapid breathing": 0.6, 
        "Chest tightness": 0.6, "Difficulty breathing during activity": 0.8, "Bluish lips": 0.1, 
        "Hoarseness": 0.3, "Sore throat": 0.2, "Nasal congestion": 0.2, "Dizziness": 0.3, 
        "Swelling in legs": 0.2, "Loss of appetite": 0.5, "Throat irritation": 0.3, "Lung crackles": 0.4,
        "Shallow breathing": 0.5, "Muscle aches": 0.7, "Chills": 0.2, "Confusion (in elderly)": 0.1,
        "Nausea": 0.2, "Headache": 0.5, "Cyanosis": 0.1, "Sweating": 0.4, "Skin rash": 0.8, "Eye pain/redness": 0.7
    },
    "Interstitial Lung Disease": {
        "Cough": 0.9, "Dry cough": 1.0, "Chronic cough": 0.9, "Chest pain": 0.5, "Shortness of breath": 1.0, 
        "Fatigue": 0.9, "Fever": 0.2, "Weight loss": 0.7, "Wheezing": 0.3, "Night sweats": 0.2, 
        "Sputum production": 0.1, "Blood in sputum (Hemoptysis)": 0.1, "Rapid breathing": 0.8, 
        "Chest tightness": 0.6, "Difficulty breathing during activity": 1.0, "Bluish lips": 0.4, 
        "Hoarseness": 0.2, "Sore throat": 0.1, "Nasal congestion": 0.1, "Dizziness": 0.4, 
        "Swelling in legs": 0.3, "Loss of appetite": 0.6, "Throat irritation": 0.3, "Lung crackles": 0.9,
        "Shallow breathing": 0.8, "Muscle aches": 0.5, "Chills": 0.1, "Confusion (in elderly)": 0.2,
        "Nausea": 0.2, "Headache": 0.3, "Cyanosis": 0.5, "Sweating": 0.3, "Clubbing of fingers": 0.7, "Joint pain": 0.6
    }
}

# The unique symptoms list based on what was added above
unique_symptoms = list(next(iter(diseases.values())).keys()) # But this only gets pneumonia ones. 
unique_symptoms_set = set()
for disease, _symptoms in diseases.items():
    for sym in _symptoms.keys():
        unique_symptoms_set.add(sym)

symptoms_list = list(unique_symptoms_set)
print(f"Total Unique Symptoms: {len(symptoms_list)}")

data = []
records_per_disease = 70  # 15 diseases * 70 = 1050 records

for disease, probs in diseases.items():
    for _ in range(records_per_disease):
        entry = {symptom: int(random.random() < probs.get(symptom, 0.05)) for symptom in symptoms_list}
        
        # Add new patient info inputs (Age, Gender, Smoking History, Previous Lung Disease, Environmental exposure)
        
        # Age
        if disease in ["COPD", "Emphysema", "Lung Cancer", "Pulmonary Fibrosis"]:
            entry["Age"] = random.randint(50, 85)
        elif disease in ["Asthma"]:
            entry["Age"] = random.randint(5, 40)
        else:
            entry["Age"] = random.randint(18, 75)
            
        # Gender (1=Male, 0=Female)
        entry["Gender"] = random.choice([0, 1])
        
        # Smoking History (1=Yes, 0=No)
        if disease in ["COPD", "Emphysema", "Lung Cancer"]:
            entry["Smoking_History"] = 1 if random.random() < 0.85 else 0
        else:
            entry["Smoking_History"] = 1 if random.random() < 0.25 else 0
            
        # Previous Lung Disease (1=Yes, 0=No)
        entry["Previous_Lung_Disease"] = 1 if random.random() < 0.15 else 0
        
        # Environmental exposure (1=Yes, 0=No)
        if disease in ["Pulmonary Fibrosis", "COPD", "Interstitial Lung Disease", "Asthma"]:
            entry["Environmental_Exposure"] = 1 if random.random() < 0.6 else 0
        else:
            entry["Environmental_Exposure"] = 1 if random.random() < 0.2 else 0
            
        entry["disease"] = disease
        data.append(entry)

# Create DataFrame and save to CSV
df = pd.DataFrame(data)

# Reorder columns to put "disease" at the end like original setup expects
cols = [col for col in df.columns if col != 'disease'] + ['disease']
df = df[cols]

df.to_csv("pulmonary_diseases.csv", index=False)
print(f"✅ Dataset saved as 'pulmonary_diseases.csv' with {len(df)} records and {len(symptoms_list)} symptoms.")
