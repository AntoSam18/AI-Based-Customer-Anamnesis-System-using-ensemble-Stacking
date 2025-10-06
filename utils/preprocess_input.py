import pandas as pd

def prepare_input(form_data, feature_order):
    tenure = float(form_data['tenure'])
    monthly_charges = float(form_data['MonthlyCharges'])
    total_charges = float(form_data['TotalCharges'])
    senior_citizen = int(form_data['SeniorCitizen'])
    contract = form_data['contract_type']
    dependents = form_data['dependents']
    device_protection = form_data['device_protection']

    input_dict = {col: 0 for col in feature_order}
    input_dict['tenure'] = tenure
    input_dict['MonthlyCharges'] = monthly_charges
    input_dict['TotalCharges'] = total_charges
    input_dict['SeniorCitizen'] = senior_citizen

    if f'Contract_{contract}' in input_dict:
        input_dict[f'Contract_{contract}'] = 1
    if f'Dependents_{dependents}' in input_dict:
        input_dict[f'Dependents_{dependents}'] = 1
    if f'DeviceProtection_{device_protection}' in input_dict:
        input_dict[f'DeviceProtection_{device_protection}'] = 1

    input_df = pd.DataFrame([input_dict], columns=feature_order)
    return input_df
