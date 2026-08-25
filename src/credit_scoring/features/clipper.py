from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from typing import Self

class Clipper(BaseEstimator, TransformerMixin):
    def __init__(self, columns: list[str], upper_quantile: float = 0.99):
        self.columns = columns
        self.upper_quantile = upper_quantile
        
    def fit(self, X: pd.DataFrame, y=None) -> Self:
        self.upper_bounds_: dict[str, float] = {} # its important to create var here, not in __init__
        for col in self.columns:
            self.upper_bounds_[col] = X[col].quantile(self.upper_quantile)
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self.columns:
            X[col] = X[col].clip(upper=self.upper_bounds_[col])
            
        return X
    
