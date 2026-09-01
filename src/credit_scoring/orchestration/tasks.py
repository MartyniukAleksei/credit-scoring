from pathlib import Path
from typing import Literal
import pandas as pd
from sklearn.pipeline import Pipeline
from credit_scoring.models.pipeline import split_data, build_lgbm_pipeline, build_logreg_pipeline
from credit_scoring.data.prepare_dataset import prepare_dataset
from credit_scoring.paths import PROJECT_ROOT
from credit_scoring.data.schema import train_schema
from credit_scoring.models.tune import objective
import optuna
import joblib
from sklearn.metrics import roc_auc_score, average_precision_score

def ingest(raw_path: Path) -> pd.DataFrame:
    df = pd.read_csv(raw_path, index_col=0)
    return prepare_dataset(df)

def validate(df: pd.DataFrame) -> pd.DataFrame:
    df = train_schema.validate(df)
    return df
    
def split_target(df: pd.DataFrame, target_col: str = "SeriousDlqin2yrs") ->  tuple[pd.DataFrame, pd.Series]:
    y = df[target_col]
    X = df.drop(columns=[target_col])
    
    return (X, y)

def split_train_test(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X_train, X_test, y_train, y_test = split_data(X, y)
    return X_train, X_test, y_train, y_test

def tune(X_train: pd.DataFrame, y_train: pd.Series, n_trials: int=30) -> dict:
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=n_trials)
    
    return study.best_params

def train(X_train: pd.DataFrame, y_train: pd.Series, model_type: Literal["logreg", "lgbm"], params: dict | None = None) -> Pipeline:
    if model_type == "lgbm":
        lgbm = build_lgbm_pipeline(params)
        lgbm.fit(X_train, y_train)
        return lgbm
    else:
        logreg = build_logreg_pipeline()
        logreg.fit(X_train, y_train)
        return logreg
    
def evaluate(fitted_pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    preds = fitted_pipeline.predict_proba(X)[:,1]
    roc_auc = roc_auc_score(y, preds)
    pr_auc = average_precision_score(y, preds)
    mean_pred = preds.mean()
    return {"roc_auc": roc_auc, "pr_auc": pr_auc, "mean_pred": mean_pred}

def register(fitter_pipeline: Pipeline, path: Path) -> None:
    Path(PROJECT_ROOT / path.parent).mkdir(exist_ok=True)
    model_path = PROJECT_ROOT / path
    
    joblib.dump(fitter_pipeline, model_path)
    print(f"Saved to {model_path}")
