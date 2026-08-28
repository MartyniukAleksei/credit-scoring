from sklearn.base import BaseEstimator, TransformerMixin
from typing import Self
import pandas as pd
import numpy as np

class WOEEncoder(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        quantile_columns: list[str],
        discrete_columns: list[str],
        n_bins: int = 10,
        sentinel_values: dict[str, list[int]] | None = None,
        smoothing: float = 0.5,
        discrete_tail_threshold: int = 4
    ):
        self.quantile_columns = quantile_columns
        self.discrete_columns = discrete_columns
        self.n_bins = n_bins
        self.sentinel_values = sentinel_values
        self.smoothing = smoothing
        self.discrete_tail_threshold = discrete_tail_threshold
    
    def _woe_for_mask(self, mask: pd.Series, y: pd.Series, all_good: int, all_bad: int) -> float:
        y_bin = y[mask]
        good = (y_bin == 0).sum()
        bad = (y_bin == 1).sum()
        return np.log(((good + self.smoothing) / all_good) / ((bad + self.smoothing) / all_bad))
    
    def _fit_discrete_column(
        self,
        col: str,
        X: pd.DataFrame,
        y: pd.Series,
        all_good: int,
        all_bad: int,
        
    ) -> dict:
        values = X[col]
        sentinels = (self.sentinel_values or {}).get(col, [])
        
        if sentinels:
            sentinel_mask = values.isin(sentinels)
            woe_map = {"sentinel": self._woe_for_mask(sentinel_mask, y, all_good, all_bad)}
            values = values[~sentinel_mask]
            y = y[~sentinel_mask]
        else:
            woe_map = {}
        
        for k in range(self.discrete_tail_threshold):
            mask = values == k
            woe_map[k] = self._woe_for_mask(mask, y, all_good, all_bad)
            
        tail_mask = values >= self.discrete_tail_threshold
        woe_map['tail'] = self._woe_for_mask(tail_mask, y, all_good, all_bad)
        
        return woe_map
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> Self:
        self.woe_maps_: dict[str, dict] = {}
        self.bin_edges_: dict[str, np.ndarray] = {}
        
        all_good = (y == 0).sum()
        all_bad = (y == 1).sum()
        
        for col in self.quantile_columns:
            bin_labels, edges = pd.qcut(X[col], q=self.n_bins, duplicates="drop", retbins=True)
            self.bin_edges_[col] = edges
            
            woe_map = {}
            for interval in bin_labels.cat.categories:
                mask = bin_labels == interval
                woe_map[interval] = self._woe_for_mask(mask, y, all_good, all_bad)
            self.woe_maps_[col] = woe_map
            
        for col in self.discrete_columns:
            self.woe_maps_[col] = self._fit_discrete_column(col, X, y, all_good, all_bad)
        
        self.default_woe_ = {col: 0.0 for col in self.quantile_columns + self.discrete_columns}
        return self
                
    def _discrete_bin_label(self, col: str, value) -> object:
        sentinels = (self.sentinel_values or {}).get(col, [])
        if value in sentinels:
            return "sentinel"
        if value < self.discrete_tail_threshold:
            return value
        return "tail"

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        for col in self.quantile_columns:
            binned = pd.cut(X[col], bins=self.bin_edges_[col], include_lowest=True)
            X[col] = binned.map(self.woe_maps_[col]).fillna(self.default_woe_[col])

        for col in self.discrete_columns:
            binned = X[col].apply(lambda v: self._discrete_bin_label(col, v))
            X[col] = binned.map(self.woe_maps_[col]).fillna(self.default_woe_[col])

        return X
    