from pathlib import Path

import pandas as pd
import joblib
from fastapi import FastAPI, status
from fastapi.staticfiles import StaticFiles
from credit_scoring.paths import MODEL_PATH
from credit_scoring.api.schemas import BorrowerFeatures
import shap

def explain_denial(shap_values_row, feature_names, top_k=4):
    contributions = list(zip(feature_names, shap_values_row.values))
    # get not only risk shap, but all features result
    risk_factors = [(name, val) for name, val in contributions]
    #risk_factors = [(name, val) for name, val in contributions if val > 0]
    risk_factors.sort(key=lambda x: -x[1])
    return risk_factors[:top_k]

pipeline = joblib.load(MODEL_PATH)
clip_step = pipeline.named_steps["clip"]
model = pipeline.named_steps["model"]
explainer = shap.TreeExplainer(model)

app = FastAPI()

@app.post(
    '/predict', # url
    response_model=dict, # тип данных, который возвращает эндоинт
    status_code=status.HTTP_200_OK, # статус, который возвращает в случаи успеха
    summary="Prediction of credit score" # что будет написанов /docs 
)
def predict(payload: BorrowerFeatures) -> dict:
    # преобразует Pydantic-объект в обычный словарь(dict)
    # be_alias=True - ключи будут как в алиасе(чтобы названия ключей совпадали)
    features_dict = payload.model_dump(by_alias=True)
    # преобразуем словарь в dataframe
    features_vector = pd.DataFrame([features_dict])
        
    pred = pipeline.predict_proba(features_vector)[:,1]
    
    feature_vector_clipper = clip_step.transform(features_vector)
    shap_values = explainer(feature_vector_clipper)
    
    return {
        'score': float(pred[0]),
        'risk_factors': [
            {'feature': name, 'contribution': float(val)}
            for name, val in explain_denial(shap_values[0], shap_values[0].feature_names, 10)
        ],
    }

STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
