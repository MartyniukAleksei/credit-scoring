import optuna
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score
from lightgbm import LGBMClassifier
from credit_scoring.features.clipper import Clipper
from credit_scoring.models.pipeline import load_data, split_data
import joblib
from pathlib import Path
from credit_scoring.paths import PROJECT_ROOT

def objective(trial: optuna.Trial, X: pd.DataFrame, y: pd.Series) -> float:
    params = {
        "num_leaves": trial.suggest_int("num_leaves", 8, 128),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial. suggest_int("min_child_samples", 10, 200),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }
    
    pipeline = Pipeline([
        ("clip", Clipper(columns=["DebtRatio", "RevolvingUtilizationOfUnsecuredLines"])),
        ("model", LGBMClassifier(**params, random_state=42, verbosity=-1))
    ]
    )
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)# cross-validation
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring='average_precision')
    return scores.mean()

if __name__ == "__main__":
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=30)
    
    print("Best PR-AUC (CV):", study.best_value)
    print("Best params:", study.best_params)
    
    final_lgbm = Pipeline([
            ("clip", Clipper(columns=["DebtRatio", "RevolvingUtilizationOfUnsecuredLines"])),
            ("model", LGBMClassifier(**study.best_params, random_state=42, verbosity=-1))
        ]
        )
    final_lgbm.fit(X_train, y_train)
    
    for name, X_part, y_part in [('training', X_train, y_train), ('test', X_test, y_test)]:
        pred = final_lgbm.predict_proba(X_part)[:,1]
        roc = roc_auc_score(y_part, pred)
        pr_auc = average_precision_score(y_part, pred)
        print(f"Model {name}: fits OK | ROC-AUC={roc:.3f} | PR-AUC={pr_auc:.3f}")
    
    Path(PROJECT_ROOT / "models").mkdir(exist_ok=True)
    model_path= PROJECT_ROOT / "models/lgbm_pipeline.joblib"
    
    joblib.dump(final_lgbm, model_path)
    print(f"Saved to {model_path}")
    
    loaded = joblib.load(model_path)
    loaded_preds = loaded.predict_proba(X_test)[:, 1]
    original_preds = final_lgbm.predict_proba(X_test)[:, 1]
    assert (loaded_preds == original_preds).all(), "predictions before and after have to be equal"
    print("Round-trip OK: predictions are identic")