# AI-Driven Customer Attrition System using Stacking Ensemble (RF + XGB + LGBM)

![Churn Prediction](https://img.shields.io/badge/Status-Completed-green)

## **Project Overview**
The **AI-Driven Customer Attrition System** is a machine learning solution designed to predict customers likely to leave a telecom service. By proactively identifying potential attrition, businesses can take retention actions to improve customer satisfaction and lifetime value.

This project uses a **stacked ensemble of Random Forest, XGBoost, and LightGBM**, combined with a **logistic regression meta-learner** for superior predictive performance.It also uses LIME to show model interporablity.

---

## **Dataset**
- **Telco Customer Churn dataset**
- Contains customer information such as tenure, contract type, services subscribed, payment method, and monthly/total charges.
- Target variable: `Churn` (Yes/No)

---

## **Key Features**
- End-to-end **data preprocessing**: numeric/categorical imputation, one-hot encoding.
- **Class balancing** using SMOTE for equal representation of churned and non-churned customers.
- **Stacked ensemble model** with hyperparameter tuning for Random Forest, XGBoost, and LightGBM.
- **Threshold optimization** to maximize F1-score.
- **Cross-validation** (5-fold Stratified) ensuring robust and generalizable performance.
- **Real-time prediction** via a **Flask web application**.
- Predicts customer attrition probability and class label using input features.

---
## Advantages
- High predictive accuracy using **ensemble stacking**.
- **Real-time prediction** accessible via web interface.
- **Robust generalization** across multiple folds (5-fold cross-validation).
- Automatic preprocessing ensures **consistent feature handling**.
- Improves understanding of **customer behavior patterns**.
- **Interactive UI** with clear, actionable output.

---

## Uniqueness
- First project to **combine RF, XGB, and LGBM** in a single stacking ensemble for churn prediction in the Telco domain.
- Auto-calculates Total Charges in the UI — no manual input required.
- Focus on **balanced dataset training** for fair and unbiased predictions.
- **Probability-based thresholding** for optimized F1-score and decision-making.
- Web interface is **user-friendly, responsive, and visually appealing**.

- ### LIME-Based Explainability

This project uses **LIME (Local Interpretable Model-Agnostic Explanations)** to provide transparency into the model’s predictions. Key features:

- **Top Feature Impact**: For each customer prediction, the top 5 features influencing churn probability are highlighted.
- **Direction of Impact**: Each feature is labeled as either **"increases churn"** or **"decreases churn"**.
- **Numeric & Categorical Support**: Both numeric values (e.g., `MonthlyCharges`) and categorical selections (e.g., `Contract_Type`) are explained.
- **Interactive Understanding**: Users can see **which features contributed most** to the predicted churn/stay outcome, making the prediction more interpretable.
- **Model-Agnostic**: Works with any classifier, including stacking ensembles like **RF + XGB + LGBM**.

Example output for a customer:

- `MonthlyCharges (799.0) -> decreases churn`
- `TotalCharges (9588.0) -> decreases churn`
- `tenure (12.0) -> decreases churn`
- `SeniorCitizen (0) -> decreases churn`
- `gender_Female (0) -> decreases churn`

This integration helps **explain the model’s decision-making process**, boosting trust and usability in business contexts.


---

## **Installation**
1. Clone this repository:
   ```bash
   git clone https://github.com/YourUsername/AI-Customer-Attrition.git
   cd AI-Customer-Attrition
   ```
2.  Install required packages:
   ```
pip install -r requirements.txt
```
Make sure all model files are in the same directory:

stacking_model_final.pkl
numeric_imputer.pkl
categorical_imputer.pkl

feature_list.pkl

Usage

## 1.Run the Flask application:
```
python app.py
```
## 2.Open a browser and go to:
```
http://127.0.0.1:5000/
```
## Input Customer Details

- **Tenure (months)**  
- **Monthly Charges**  
- **Total Charges**  
- **Senior Citizen (Yes/No)**  
- **Contract Type**  
- **Dependents (Yes/No)**  
- **Device Protection (Yes/No)**  

The system provides **real-time prediction with probability score**.

## Sample Output
- 🚨 Likely to CHURN (Probability: 0.78)  
- ✅ Likely to STAY (Probability: 0.22)  


## Results

- **5-Fold Cross-Validation:**  
  - Mean ROC AUC: 0.937 ± 0.009  
  - Mean F1-score: 0.851 ± 0.007  

- **Optimized threshold** for churn prediction: 0.366  
- **Balanced training classes:** [3622, 3622]  
- Robust prediction performance across folds demonstrates **generalizability**.

## Project Structure
```
AI-Customer-Attrition-Prediction/
│
├── app.py                        # Flask application
├── README.md                      # Project description & instructions
├── requirements.txt               # Dependencies
│
├── models/
│   ├── stacking_model_final.pkl   # Pre-trained stacking ensemble model
│   └── feature_list.pkl           # Feature order for inference
│
├── utils/
│   └── preprocess_input.py        # Input preprocessing for prediction
│
├── templates/
│   └── index.html                 # Web interface for user input
│
└── static/                        # (Optional) CSS, images, JS for UI enhancement
```
## Tech Stack

- **Python 3.x**
- **scikit-learn**, **XGBoost**, **LightGBM**, **imbalanced-learn**
- **Flask** (Web App)
- **Pandas**, **NumPy**, **Matplotlib**, **Seaborn**
- **LIME**
## 📊 Model Architecture
![Model Architecture](images/arch.png)

## 🔍 LIME Explanation
![LIME Output](images/graph.png)

## 💻 Web App Screenshot
![App Screenshot](images/Output1.png)
![App Screenshot](images/Output2.png)

## Author
- **Anto Sam Christ A**  
- **Email:** antosamchrist18@gmail.com  
- **B.Tech, Computer Science & Engineering**  
- [Visit my Profile](https://antosamchrista.netlify.app)


