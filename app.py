from flask import Flask, render_template, request
import pandas as pd
import joblib, pickle
import os
from lime.lime_tabular import LimeTabularExplainer
import numpy as np

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


model = joblib.load(os.path.join(BASE_DIR, "models", "stacking_model_final.pkl"))
with open(os.path.join(BASE_DIR, "models", "feature_list.pkl"), "rb") as f:
    feature_order = pickle.load(f)


sample_data = pd.DataFrame(np.zeros((10, len(feature_order))), columns=feature_order)
explainer = LimeTabularExplainer(
    training_data=sample_data.values,
    feature_names=feature_order,
    class_names=['Stay', 'Churn'],
    mode='classification'
)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        
        tenure = float(request.form['tenure'])
        monthly_charges = float(request.form['MonthlyCharges'])
        total_charges = tenure * monthly_charges
        senior_citizen = 1 if request.form['SeniorCitizen'].lower() == 'yes' else 0
        contract = request.form['contract_type']
        dependents = request.form['dependents']
        device_protection = request.form['device_protection']

        
        input_dict = {col: 0 for col in feature_order}
        input_dict.update({
            'tenure': tenure,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges,
            'SeniorCitizen': senior_citizen
        })

        for cat_col, val in {'Contract': contract, 'Dependents': dependents, 'DeviceProtection': device_protection}.items():
            key = f"{cat_col}_{val}"
            if key in input_dict:
                input_dict[key] = 1

        input_df = pd.DataFrame([input_dict], columns=feature_order)

        
        prob = model.predict_proba(input_df)[:, 1][0]
        threshold = 0.366
        prediction = "🚨 Likely to CHURN" if prob >= threshold else "✅ Likely to STAY"

    
        exp = explainer.explain_instance(input_df.values[0], model.predict_proba, num_features=5)
        feature_impacts = [f"{feat} {impact:.2f} ({'increases' if val > 0 else 'decreases'} churn)" 
                           for feat, val, impact in exp.as_list()]

        return render_template(
            'index.html',
            prediction_text=f"{prediction} (Probability: {prob:.2f})",
            tenure=tenure,
            monthly_charges=monthly_charges,
            total_charges=round(total_charges, 2),
            lime_explanation=feature_impacts
        )

    except Exception as e:
        return render_template(
            'index.html',
            prediction_text=f"⚠️ Error: {str(e)}"
        )

if __name__ == "__main__":
    app.run(debug=True)
