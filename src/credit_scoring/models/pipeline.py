import pandas as pd 
from credit_scoring.data.prepare_dataset import prepare_dataset
from credit_scoring.data.schema import train_schema
from credit_scoring.features.woeencoder import WOEEncoder
from credit_scoring.features.clipper import Clipper
from credit_scoring.paths import DATA_RAW
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from lightgbm import LGBMClassifier

def load_data() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(DATA_RAW / "cs-training.csv", index_col=0)
    
    df = prepare_dataset(df)
    df = train_schema.validate(df)#pandera returns new df with needed types
    
    y = df['SeriousDlqin2yrs']
    X = df.drop(columns=['SeriousDlqin2yrs'])
    
    return X, y

def build_logreg_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("woe", WOEEncoder(
                quantile_columns=["age", "RevolvingUtilizationOfUnsecuredLines", "DebtRatio"],
                discrete_columns=["NumberOfTime30-59DaysPastDueNotWorse", "NumberOfTime60-89DaysPastDueNotWorse", "NumberOfTimes90DaysLate"],
                sentinel_values={
                    "NumberOfTime30-59DaysPastDueNotWorse": [96, 98],
                    "NumberOfTime60-89DaysPastDueNotWorse": [96, 98],
                    "NumberOfTimes90DaysLate": [96, 98],
                }),
             ["age", "RevolvingUtilizationOfUnsecuredLines", "DebtRatio",
              "NumberOfTime30-59DaysPastDueNotWorse", "NumberOfTime60-89DaysPastDueNotWorse", "NumberOfTimes90DaysLate"]),
            ("impute_income", Pipeline([
                ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                ("scale", StandardScaler()),
            ]), ["MonthlyIncome"]),
            ("impute_dependents", Pipeline([
                ("impute", SimpleImputer(strategy="constant", fill_value=0)),
                ("scale", StandardScaler()),
            ]), ["NumberOfDependents"]),
            ("scale_rest", StandardScaler(), ["NumberOfOpenCreditLinesAndLoans", "NumberRealEstateLoansOrLines"]),
        ],
        remainder="drop",
    )
    
    logreg_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression()),
    ])
    
    return logreg_pipeline
    
def build_lgbm_pipeline() -> Pipeline:
    lgbm_pipeline = Pipeline([
        ("clip", Clipper(columns=["DebtRatio", "RevolvingUtilizationOfUnsecuredLines"])),
        ("model", LGBMClassifier()),
    ])
    
    return lgbm_pipeline

def split_data(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

if __name__ == '__main__':
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    logred = build_logreg_pipeline()
    logred.fit(X_train, y_train)
    for name, X_type, y_type in [('Train', X_train, y_train), ('Test', X_test, y_test)]:
        preds = logred.predict_proba(X_type)[:, 1]
        auc = roc_auc_score(y_type, preds)
        pr_auc = average_precision_score(y_type, preds)
        assert len(preds) == len(X_type), "the number of predictions have to be equal to number of rows"
        assert not pd.Series(preds).isna().any(), "не должно быть NaN в предсказаниях"
        assert ((preds >= 0) & (preds <= 1)).all(), "prohabilities have to be between [0,1]"
        print(f"LogReg {name}: fit OK, ROC-AUC={auc} | PR_AUC={pr_auc} | mean prediction = {preds.mean():.3f} (baseline ~0.07)")
    
    lgbm = build_lgbm_pipeline()
    lgbm.fit(X_train, y_train)
    for name, X_type, y_type in [('Train', X_train, y_train), ('Test', X_test, y_test)]:
        preds_lgbm = lgbm.predict_proba(X_type)[:, 1]
        auc = roc_auc_score(y_type, preds_lgbm)
        pr_auc = average_precision_score(y_type, preds_lgbm)
        assert len(preds_lgbm) == len(X_type)
        assert not pd.Series(preds_lgbm).isna().any()
        assert ((preds_lgbm >= 0) & (preds_lgbm <= 1)).all()
        print(f"LightGBM {name}: fit OK, ROC-AUC={auc} | PR-AUC={pr_auc} | mean prediction = {preds_lgbm.mean():.3f} (baseline ~0.07)")