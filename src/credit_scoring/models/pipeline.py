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

if __name__ == '__main__':
    X, y = load_data()
    
    logred = build_logreg_pipeline()
    logred.fit(X, y)
    preds = logred.predict_proba(X)[:, 1]
    assert len(preds) == len(X), "the number of predictions have to be equal to number of rows"
    assert not pd.Series(preds).isna().any(), "не должно быть NaN в предсказаниях"
    assert ((preds >= 0) & (preds <= 1)).all(), "prohabilities have to be between [0,1]"
    print(f"LogReg: fit OK, mean prediction = {preds.mean():.3f} (baseline ~0.07)")
    
    lgbm = build_lgbm_pipeline()
    lgbm.fit(X, y)
    preds_lgbm = lgbm.predict_proba(X)[:, 1]
    assert len(preds_lgbm) == len(X)
    assert not pd.Series(preds_lgbm).isna().any()
    assert ((preds_lgbm >= 0) & (preds_lgbm <= 1)).all()
    print(f"LightGBM: fit OK, mean prediction = {preds_lgbm.mean():.3f} (baseline ~0.07)")