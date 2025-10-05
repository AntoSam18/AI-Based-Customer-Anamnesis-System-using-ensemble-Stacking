from flask import Flask, render_template, request
import pandas as pd
import joblib, pickle, os

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model and feature order
model = joblib.load(os.path.join(BASE_DIR, "stacking_model_final.pkl"))
with open(os.path.join(BASE_DIR, "feature_list.pkl"), "rb") as f:
    feature_order = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        tenure = float(request.form['tenure'])
        monthly_charges = float(request.form['MonthlyCharges'])
        total_charges = float(request.form['TotalCharges'])
        senior_citizen = int(request.form['SeniorCitizen'])
        contract = request.form['contract_type']
        dependents = request.form['dependents']
        device_protection = request.form['device_protection']

        # Initialize input dict with 0s
        input_dict = {col: 0 for col in feature_order}
        input_dict['tenure'] = tenure
        input_dict['MonthlyCharges'] = monthly_charges
        input_dict['TotalCharges'] = total_charges
        input_dict['SeniorCitizen'] = senior_citizen

        # One-hot encode categorical features
        if f'Contract_{contract}' in input_dict:
            input_dict[f'Contract_{contract}'] = 1
        if f'Dependents_{dependents}' in input_dict:
            input_dict[f'Dependents_{dependents}'] = 1
        if f'DeviceProtection_{device_protection}' in input_dict:
            input_dict[f'DeviceProtection_{device_protection}'] = 1

        input_df = pd.DataFrame([input_dict], columns=feature_order)
        prob = model.predict_proba(input_df)[:, 1][0]
        threshold = 0.35
        prediction = "Likely to CHURN 🚨" if prob >= threshold else "Likely to STAY ✅"
        color = "danger" if prob >= threshold else "success"

        return render_template('index.html', prediction_text=f"{prediction} (Probability: {prob:.2f})", color=color)

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}", color="warning")

if __name__ == "__main__":
    app.run(debug=True)
