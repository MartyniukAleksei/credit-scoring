import pandas as pd
import joblib
from fastapi import FastAPI, status
from credit_scoring.paths import PROJECT_ROOT
from credit_scoring.api.schemas import BorrowerFeatures

MODEL_PATH = PROJECT_ROOT / 'models/lgbm_pipeline.joblib'
pipeline = joblib.load(MODEL_PATH)

app = FastAPI()

@app.post(
    '/predict',
    response_model=float,
    status_code=status.HTTP_200_OK,
    summary="Prediction of credit score"
)
def predict(payload: BorrowerFeatures) -> float:
    features_dict = payload.model_dump(by_alias=True)
    features_vector = pd.DataFrame([features_dict])
        
    pred = pipeline.predict_proba(features_vector)[:,1]
    return pred[0]