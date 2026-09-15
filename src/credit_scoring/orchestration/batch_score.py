import joblib
from credit_scoring.paths import DATA_RAW, MODEL_PATH, DATA_PROCESSED
from credit_scoring.orchestration.tasks import ingest
from credit_scoring.data.schema import base_schema
from prefect import flow

pipeline = joblib.load(MODEL_PATH)

@flow
def score_batch(input_path, output_path) -> None:
    df = ingest(input_path)
    df = base_schema.validate(df)
    feature_cols = list(base_schema.columns.keys())
    X = df[feature_cols]
    df['probability'] = pipeline.predict_proba(X)[:,1]
    df.to_csv(output_path)
    
    
if __name__ == '__main__':
    score_batch(DATA_RAW / 'cs-test.csv', DATA_PROCESSED / 'preds.csv')