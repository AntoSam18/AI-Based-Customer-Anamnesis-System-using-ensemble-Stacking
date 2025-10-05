import joblib, pickle, os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_model_and_features():
    model = joblib.load(os.path.join(BASE_DIR, "stacking_model_final.pkl"))
    with open(os.path.join(BASE_DIR, "feature_list.pkl"), "rb") as f:
        feature_order = pickle.load(f)
    return model, feature_order
