from flask import Flask, render_template, request, redirect, url_for, session, flash, send_file, jsonify
import sqlite3
import os
import joblib
import pickle
import numpy as np
import uuid
import audio_processing
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from audio_processing import extract_features

app = Flask(__name__)
app.secret_key = "your_secret_key"  # Change this for production security
DB_PATH = "db.sqlite3"

# ---------------------------
# Load model and encoder
# ---------------------------
MODEL_FILE = "pulmonary_model.pkl"
ENCODER_FILE = "label_encoder.pkl"

def load_any(path):
    """Try joblib then pickle to load an object; return None if both fail."""
    if not os.path.exists(path):
        return None
    try:
        return joblib.load(path)
    except Exception:
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None

model = load_any(MODEL_FILE)
encoder_obj = load_any(ENCODER_FILE)

# Cough model
COUGH_MODEL_FILE = "cough_model.pkl"
cough_model = load_any(COUGH_MODEL_FILE)
COUGH_TYPES = ["Healthy Cough", "Dry Cough", "Wet Cough", "Asthma Cough", "Pneumonia Cough"]
COUGH_CONDITIONS = {
    "Healthy Cough": ["No specific underlying condition", "Normal clearance"],
    "Dry Cough": ["Allergies", "Viral Infection", "Asthma"],
    "Wet Cough": ["Bronchitis", "Pneumonia", "COPD"],
    "Asthma Cough": ["Asthma flare-up", "Reactive airway"],
    "Pneumonia Cough": ["Pneumonia", "Severe Respiratory Infection"]
}

def decode_label(encoded):
    """Decode encoded label safely."""
    if encoder_obj is None:
        return str(encoded)
    if hasattr(encoder_obj, "inverse_transform"):
        try:
            return encoder_obj.inverse_transform([encoded])[0]
        except Exception:
            return str(encoded)
    if isinstance(encoder_obj, dict):
        rev = {v: k for k, v in encoder_obj.items()}
        return rev.get(encoded, str(encoded))
    return str(encoded)

# ---------------------------
# Model features
# ---------------------------
default_model_features = [
    "Cough", "Shortness of breath", "Chest pain", "Fatigue", "Fever",
    "Weight loss", "Wheezing", "Night sweats", "Sputum production", "Blood in sputum"
]

try:
    model_feature_names = list(model.feature_names_in_) if hasattr(model, "feature_names_in_") else default_model_features
except Exception:
    model_feature_names = default_model_features

def norm(s):
    return s.strip().lower().replace(" ", "_").replace("-", "_")

form_feature_names = [norm(f) for f in model_feature_names]
orig_to_form = dict(zip(model_feature_names, form_feature_names))
symptoms_for_template = form_feature_names

# ---------------------------
# Database setup
# ---------------------------
def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            email TEXT,
            password TEXT NOT NULL
        )
    ''')
    cur.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            symptoms TEXT,
            prediction TEXT,
            timestamp TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    ''')
    try:
        cur.execute("ALTER TABLE users ADD COLUMN age INTEGER")
        cur.execute("ALTER TABLE users ADD COLUMN gender TEXT")
        cur.execute("ALTER TABLE users ADD COLUMN smoking_history TEXT")
    except Exception:
        pass
    conn.commit()
    conn.close()

init_db()

# ---------------------------
# Disease info dictionary
# ---------------------------
DISEASE_INFO = {
    "Pneumonia": {
        "description": "Pneumonia is an infection that inflames the air sacs in one or both lungs.",
        "treatment": "Antibiotics (if bacterial), adequate rest, fluids, and medical care."
    },
    "Tuberculosis": {
        "description": "A contagious bacterial infection usually affecting the lungs.",
        "treatment": "Long-term multi-drug antibiotic regimen under medical supervision."
    },
    "COVID-19 pneumonia": {
        "description": "Severe lung infection caused by the SARS-CoV-2 virus.",
        "treatment": "Antiviral medications, oxygen therapy, steroids, and rest."
    },
    "Bronchitis": {
        "description": "Inflammation of the lining of your bronchial tubes, which carry air to and from your lungs.",
        "treatment": "Rest, fluids, cough suppressants, and sometimes bronchodilators."
    },
    "Asthma": {
        "description": "Asthma causes airways to narrow and swell and may produce extra mucus.",
        "treatment": "Inhaled corticosteroids, bronchodilators, and avoiding triggers."
    },
    "COPD": {
        "description": "Chronic obstructive pulmonary disease causes airflow blockage and breathing problems.",
        "treatment": "Quit smoking, inhalers, pulmonary rehab, oxygen therapy."
    },
    "Pulmonary Fibrosis": {
        "description": "Scarring of lung tissue leading to progressive shortness of breath.",
        "treatment": "Medications to slow progression, oxygen therapy, pulmonary rehab."
    },
    "Lung Cancer": {
        "description": "Cancer that begins in the lungs and most often occurs in people who smoke.",
        "treatment": "Surgery, chemotherapy, radiation therapy, targeted drug therapy."
    },
    "Emphysema": {
        "description": "A lung condition that causes shortness of breath, often part of COPD.",
        "treatment": "Bronchodilators, inhaled steroids, supplemental oxygen, smoking cessation."
    },
    "Pulmonary Hypertension": {
        "description": "High blood pressure that affects the arteries in your lungs and the right side of your heart.",
        "treatment": "Blood vessel dilators, anticoagulants, diuretics, oxygen."
    },
    "Pulmonary Edema": {
        "description": "A condition caused by excess fluid in the lungs, making it difficult to breathe.",
        "treatment": "Supplemental oxygen and medications (diuretics) to remove fluid."
    },
    "Pleural Effusion": {
        "description": "A buildup of fluid between the tissues that line the lungs and the chest.",
        "treatment": "Draining fluid, treating the underlying cause (e.g., antibiotics for pneumonia)."
    },
    "Pneumothorax": {
        "description": "A collapsed lung occurring when air leaks into the space between your lung and chest wall.",
        "treatment": "Observation, needle or chest tube insertion, or surgery."
    },
    "Sarcoidosis": {
        "description": "The growth of inflammatory cells (granulomas) in different parts of your body, most commonly the lungs.",
        "treatment": "Corticosteroids, immunosuppressive drugs, or careful monitoring if mild."
    },
    "Interstitial Lung Disease": {
        "description": "A large group of disorders that cause progressive scarring of lung tissue.",
        "treatment": "Corticosteroids, oxygen therapy, pulmonary rehabilitation."
    }
}

# ---------------------------
# Routes
# ---------------------------

@app.route("/")
def root():
    # If already logged in, go home, else login or maybe just home
    return redirect(url_for("home"))

# ---------------------------
# Registration Route
# ---------------------------
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "").strip()

        if not username or not password:
            flash("Username and password are required.", "error")
            return redirect(url_for("register"))

        hashed = generate_password_hash(password)
        conn = get_db_connection()
        cur = conn.cursor()
        try:
            cur.execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
                        (username, email, hashed))
            conn.commit()
            flash("Registration successful — please login.", "success")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("Username already exists — choose another.", "error")
        finally:
            conn.close()
    return render_template("register.html")

# ---------------------------
# Login Route
# ---------------------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE username = ?", (username,))
        user = cur.fetchone()
        conn.close()
        if user and check_password_hash(user["password"], password):
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            flash(f"Welcome, {user['username']}!", "success")
            return redirect(url_for("home"))
        else:
            flash("Invalid username or password.", "error")
    return render_template("login.html")

# ---------------------------
# Disease Info Route
# ---------------------------
@app.route("/disease/<disease_name>")
def disease_info(disease_name):
    # Retrieve info from dictionary
    info = DISEASE_INFO.get(disease_name)
    if not info:
        flash("Disease information not found.", "warning")
        return redirect(url_for("predict"))
        
    return render_template("disease_info.html", disease=disease_name, info=info, username=session.get("username"))


# ---------------------------
# Logout Route
# ---------------------------
@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("home"))

# ---------------------------
# Core Pages Route
# ---------------------------
@app.route("/home")
def home():
    return render_template("home.html", username=session.get("username"))

@app.route("/about")
def about():
    return render_template("about.html", username=session.get("username"))

@app.route("/profile", methods=["GET", "POST"])
def profile():
    if not session.get("user_id"):
        flash("Please login first.", "error")
        return redirect(url_for("login"))
    
    conn = get_db_connection()
    cur = conn.cursor()
    
    if request.method == "POST":
        new_username = request.form.get("username", "").strip()
        new_email = request.form.get("email", "").strip()
        new_age = request.form.get("age", "").strip()
        new_gender = request.form.get("gender", "").strip()
        new_smoking = request.form.get("smoking", "").strip()
        
        # safely cast age
        age_val = None
        if new_age.isdigit():
            age_val = int(new_age)
        
        if not new_username:
            flash("Username cannot be empty.", "error")
        else:
            try:
                cur.execute("UPDATE users SET username = ?, email = ?, age = ?, gender = ?, smoking_history = ? WHERE id = ?", (new_username, new_email, age_val, new_gender, new_smoking, session["user_id"]))
                conn.commit()
                session["username"] = new_username
                flash("Profile updated successfully!", "success")
            except sqlite3.IntegrityError:
                flash("Username already exists. Choose another one.", "error")
                
    cur.execute("SELECT email, age, gender, smoking_history FROM users WHERE id = ?", (session["user_id"],))
    user = cur.fetchone()
    conn.close()
    
    email = user["email"] if user and user["email"] else ""
    age = user["age"] if user and user["age"] else ""
    gender = user["gender"] if user and user["gender"] else ""
    smoking = user["smoking_history"] if user and user["smoking_history"] else ""
    
    return render_template("profile.html", username=session.get("username"), email=email, age=age, gender=gender, smoking=smoking)

# ---------------------------
# Cough Analysis Route
# ---------------------------
@app.route("/analyze_cough", methods=["POST"])
def analyze_cough():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
        
    audio_file = request.files["audio"]
    if audio_file.filename == "":
        return jsonify({"error": "Empty file"}), 400
        
    try:
        temp_path = f"temp_audio_{uuid.uuid4().hex}.wav"
        audio_file.save(temp_path)
        
        features = extract_features(temp_path)
        
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        if features is None:
            return jsonify({"error": "Could not extract audio features. Make sure the recording isn't silent."}), 400
            
        global cough_model
        if cough_model is None:
            cough_model = load_any(COUGH_MODEL_FILE)
            
        if cough_model is None:
            return jsonify({"error": "Cough AI model is not trained/available."}), 500
            
        X = features.reshape(1, -1)
        probas = cough_model.predict_proba(X)[0]
        pred_idx = np.argmax(probas)
        
        predicted_type = COUGH_TYPES[pred_idx]
        confidence = probas[pred_idx] * 100
        possible_conditions = COUGH_CONDITIONS.get(predicted_type, [])
        
        return jsonify({
            "cough_type": predicted_type,
            "possible_conditions": possible_conditions,
            "confidence": round(confidence, 1)
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------------------------
# Predict Route
# ---------------------------
@app.route("/predict", methods=["GET", "POST"])
def predict():
    if not session.get("user_id"):
        flash("Please login first.", "error")
        return redirect(url_for("login"))

    prediction = None
    description = None
    treatment = None
    confidence = None
    risk_level = None
    selected_symptoms = []
    specialist = None

    if request.method == "POST":
        form = request.form
        gender = form.get("gender")
        age = form.get("age")
        height = form.get("height")
        weight = form.get("weight")

        # Compute BMI
        try:
            height_m = float(height) / 100
            bmi = float(weight) / (height_m ** 2)
        except:
            bmi = 0

        # Features extraction
        vals = []
        selected_symptoms = []
        respiratory_count = 0

        # Medical Safety Rule checklist
        respiratory_symptoms = [
            'cough', 'dry_cough', 'chronic_cough', 'shortness_of_breath', 'chest_pain',
            'wheezing', 'sputum_production', 'blood_in_sputum_(hemoptysis)', 'rapid_breathing',
            'chest_tightness', 'difficulty_breathing_during_activity', 'pink_frothy_sputum',
            'shallow_breathing', 'barrel_chest', 'lung_crackles'
        ]

        # Parse categorical/numerical inputs
        parsed_age = float(form.get("age", 40))
        parsed_gender = 1 if form.get("gender") == "Male" else 0
        smoker_status = form.get("smoking_status", "Never")
        parsed_smoking = 1 if smoker_status in ["Former", "Current"] else 0
        parsed_previous = 1 if form.get("previous_conditions") else 0
        parsed_env = 1 if form.get("environmental_exposure") == "Yes" else 0

        for orig in model_feature_names:
            if orig == "Age":
                vals.append(parsed_age)
            elif orig == "Gender":
                vals.append(parsed_gender)
            elif orig == "Smoking_History":
                vals.append(parsed_smoking)
            elif orig == "Previous_Lung_Disease":
                vals.append(parsed_previous)
            elif orig == "Environmental_Exposure":
                vals.append(parsed_env)
            else:
                # It's a symptom
                fname = orig_to_form[orig]
                is_selected = 1 if form.get(fname) else 0
                vals.append(is_selected)
                
                if is_selected:
                    selected_symptoms.append(orig)
                    if fname in respiratory_symptoms:
                        respiratory_count += 1
                        
        X = np.array(vals).reshape(1, -1)

        if len(selected_symptoms) > 0 and respiratory_count == 0:
            flash("Symptoms are insufficient for pulmonary disease prediction. Please consult a medical professional.", "warning")
            prediction = "Insufficient Symptoms"
            description = "You have selected symptoms that are exclusively non-respiratory (e.g., only fever, only fatigue). A pulmonary disease cannot be confidently predicted without at least one respiratory symptom."
            treatment = "Please consult a healthcare provider for a proper medical evaluation."
            confidence = 0
            risk_level = "Unknown"
            specialist = "General Physician"
            top3 = []
        elif model is None:
            flash("Model not loaded. Please run training script first.", "error")
            top3 = []
        else:
            try:
                probabilities = model.predict_proba(X)[0]
                
                # Get Top 3 predictions
                top3_indices = np.argsort(probabilities)[::-1][:3]
                top3 = []
                for idx in top3_indices:
                    prob = round(probabilities[idx] * 100, 2)
                    label = decode_label(idx)
                    top3.append({'disease': label, 'probability': prob})

                pred_label = top3[0]['disease']
                confidence = top3[0]['probability']
                prediction = pred_label
                if confidence >= 80:
                    risk_level = "High 🔴"
                elif confidence >= 60:
                    risk_level = "Medium 🟡"
                else:
                    risk_level = "Low 🟢"

                # Example condition: elderly female = higher risk
                if gender == "Female" and float(age) > 60:
                    prediction += " (High Risk)"

                info = DISEASE_INFO.get(pred_label, {})
                description = info.get("description", "Description not available.")
                treatment = info.get("treatment", "Treatment not available.")
                # Specialist recommendation
                SPECIALIST = {
                    "Asthma": "Pulmonologist",
                    "Pneumonia": "Pulmonologist",
                    "COVID-19 pneumonia": "Pulmonologist / Infectious Disease",
                    "Bronchitis": "Pulmonologist",
                    "COPD": "Pulmonologist",
                    "Pulmonary Fibrosis": "Pulmonologist",
                    "Lung Cancer": "Oncologist / Pulmonologist",
                    "Emphysema": "Pulmonologist",
                    "Pulmonary Hypertension": "Cardiologist / Pulmonologist",
                    "Pulmonary Edema": "Cardiologist / Emergency Medicine",
                    "Pleural Effusion": "Pulmonologist",
                    "Pneumothorax": "Emergency Medicine / Thoracic Surgeon",
                    "Sarcoidosis": "Pulmonologist / Rheumatologist",
                    "Interstitial Lung Disease": "Pulmonologist",
                    "Tuberculosis": "Infectious Disease Specialist"
                }

                specialist = SPECIALIST.get(pred_label, "General Physician")
                # Save history in DB
                conn = get_db_connection()
                cur = conn.cursor()
                symptom_str = ",".join([f"{k}:{v}" for k, v in zip(form_feature_names, vals)])
                
                if top3:
                    pred_db_str = ", ".join([f"{t['disease']} ({t['probability']}%)" for t in top3])
                else:
                    pred_db_str = prediction

                cur.execute(
                    "INSERT INTO history (user_id, symptoms, prediction, timestamp) VALUES (?, ?, ?, ?)",
                    (session["user_id"], symptom_str, pred_db_str, datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                )
                conn.commit()
                conn.close()

                # Save extended details to CSV
                with open("history.csv", "a") as f:
                    f.write(f"{session['username']},{gender},{age},{height},{weight},{bmi},{pred_db_str},{datetime.now()}\n")

            except Exception as e:
                flash("Prediction error: " + str(e), "error")

    return render_template(
        "predict.html",
        symptoms=symptoms_for_template,
        prediction=prediction,
        description=description,
        treatment=treatment,
        confidence=confidence,
        risk_level=risk_level,
        selected_symptoms=selected_symptoms,
        specialist=specialist,
        top3=top3 if 'top3' in locals() else [],
        username=session.get("username"),
    )

# ---------------------------
# History Route
# ---------------------------
@app.route("/history")
def history():
    if not session.get("user_id"):
        flash("Please login first.", "error")
        return redirect(url_for("login"))

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT symptoms, prediction, timestamp FROM history WHERE user_id = ? ORDER BY timestamp DESC",
        (session["user_id"],),
    )
    rows = cur.fetchall()
    conn.close()
    history = [{"prediction": r["prediction"], "timestamp": r["timestamp"], "symptoms": r["symptoms"]} for r in rows]
    return render_template("history.html", history=history, username=session.get("username"))
@app.route("/dashboard")
def dashboard():

    if not session.get("user_id"):
        flash("Please login first.", "error")
        return redirect(url_for("login"))

    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute(
        "SELECT symptoms, prediction, timestamp FROM history WHERE user_id=?",
        (session["user_id"],)
    )

    rows = cur.fetchall()
    conn.close()

    # -------------------------
    # Disease count
    # -------------------------
    disease_counts = {}

    for r in rows:
        disease = r["prediction"]
        disease_counts[disease] = disease_counts.get(disease, 0) + 1

    disease_labels = list(disease_counts.keys())
    disease_values = list(disease_counts.values())

    # -------------------------
    # Symptom count
    # -------------------------
    symptom_counts = {}

    for r in rows:

        symptoms = r["symptoms"].split(",")

        for s in symptoms:
            if ":" not in s:
                continue
            name, value = s.split(":", 1)
            if value == "1":
                name = name.replace("_", " ").title()
                symptom_counts[name] = symptom_counts.get(name, 0) + 1

    symptom_labels = list(symptom_counts.keys())
    symptom_values = list(symptom_counts.values())

    # -------------------------
    # Timeline chart
    # -------------------------
    timeline_counts = {}

    for r in rows:

        date = r["timestamp"].split(" ")[0]

        timeline_counts[date] = timeline_counts.get(date, 0) + 1

    timeline_labels = list(timeline_counts.keys())
    timeline_values = list(timeline_counts.values())

    return render_template(
        "dashboard.html",
        disease_labels=disease_labels,
        disease_values=disease_values,
        symptom_labels=symptom_labels,
        symptom_values=symptom_values,
        timeline_labels=timeline_labels,
        timeline_values=timeline_values
    )
@app.route("/download_report")
def download_report():

    if not session.get("user_id"):
        return redirect(url_for("login"))

    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute(
        "SELECT symptoms, prediction, timestamp FROM history WHERE user_id=? ORDER BY timestamp DESC LIMIT 1",
        (session["user_id"],)
    )

    row = cur.fetchone()
    conn.close()

    if not row:
        return "No prediction found"

    styles = getSampleStyleSheet()

    filename = "health_report.pdf"
    doc = SimpleDocTemplate(filename)

    elements = []

    elements.append(Paragraph("Respida AI Health Report", styles["Title"]))
    elements.append(Spacer(1, 20))

    elements.append(Paragraph(f"<b>Prediction(s) / Probabilities:</b> {row['prediction']}", styles["Normal"]))
    elements.append(Paragraph(f"<b>Date:</b> {row['timestamp']}", styles["Normal"]))
    
    # Parse symptoms and patient info
    symp_list = []
    patient_info = []
    
    try:
        items = row['symptoms'].split(",")
        for item in items:
            if ":" in item:
                k, v = item.split(":", 1)
                k = k.strip()
                v = v.strip()
                if k in ['age', 'gender', 'height', 'weight', 'smoking_history', 'previous_lung_disease', 'environmental_exposure']:
                    display_k = k.replace('_', ' ').title()
                    # format gender
                    if k == 'gender':
                        v_disp = 'Male' if v == '1' else ('Female' if v == '0' else v)
                    elif k in ['smoking_history', 'previous_lung_disease', 'environmental_exposure']:
                        v_disp = 'Yes' if v == '1' else ('No' if v == '0' else v)
                    else:
                        v_disp = v
                    patient_info.append(f"{display_k}: {v_disp}")
                else:
                    if v == "1" or v == "1.0":
                        symp_list.append(k.replace('_', ' ').title())
    except Exception as e:
        print("Error parsing symptoms:", e)
        
    symps = ", ".join(symp_list) if symp_list else "None reported"
    info_str = ", ".join(patient_info) if patient_info else "Not provided"

    elements.append(Spacer(1, 10))
    elements.append(Paragraph(f"<b>Patient Info & Risk Factors:</b> {info_str}", styles["Normal"]))
    elements.append(Paragraph(f"<b>Symptoms Selected:</b> {symps}", styles["Normal"]))
    elements.append(Spacer(1, 15))
    
    # Infer main prediction from string for generic info
    main_pred = row['prediction'].split(" (")[0]
    info = DISEASE_INFO.get(main_pred, {})
    if info:
        elements.append(Paragraph(f"<b>Description:</b> {info.get('description', '')}", styles["Normal"]))
        elements.append(Spacer(1, 5))
        elements.append(Paragraph(f"<b>Treatment Suggestions:</b> {info.get('treatment', '')}", styles["Normal"]))
        elements.append(Spacer(1, 5))
        elements.append(Paragraph("<b>Prevention Advice:</b> Avoid smoking and secondhand smoke, minimize exposure to pollutants, and maintain good hand hygiene.", styles["Normal"]))
        
    elements.append(Spacer(1, 10))
    elements.append(Paragraph("<b>Nearby Facilities:</b> Search Google Maps for 'Pulmonologist or Hospitals near me' to find medical care.", styles["Normal"]))

    elements.append(Spacer(1, 20))
    # Disclaimer
    elements.append(Paragraph("<font color='red'><b>Medical Disclaimer:</b> This report is generated by AI for educational and informational purposes only and is NOT a medical diagnosis. Please consult a qualified healthcare professional before taking any medical action.</font>", styles["Normal"]))

    doc.build(elements)

    return send_file("health_report.pdf", as_attachment=True)

# ---------------------------
import os

# ---------------------------
# Run App
# ---------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
