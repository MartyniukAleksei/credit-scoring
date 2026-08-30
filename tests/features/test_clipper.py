import pandas as pd
import pytest
from credit_scoring.features.clipper import Clipper

def test_fit_computes_expected_upper_bound():
    # 101 values 0..100: the 0.99 quantile falls exactly on 99.0 (no interpolation)
    df = pd.DataFrame({"x": range(101)})
    clipper = Clipper(columns=["x"], upper_quantile=0.99).fit(df)
    assert clipper.upper_bounds_["x"] == 99.0
    assert clipper.upper_bounds_["x"] == df["x"].quantile(0.99)

def test_transform_clips_above_bound_and_leaves_lower_values_intact():
    # 11 values 0..10: the 0.9 quantile is 9.0
    df = pd.DataFrame({"x": range(11)})
    clipper = Clipper(columns=["x"], upper_quantile=0.9).fit(df)
    result = clipper.transform(df)

    assert clipper.upper_bounds_["x"] == 9.0
    assert result["x"].max() == 9.0
    # everything at or below the bound is untouched
    assert result["x"].tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9]

def test_transform_uses_bound_from_fit_not_recomputed_on_new_data():
    train = pd.DataFrame({"x": range(11)})            # 0.9 quantile == 9.0
    unseen = pd.DataFrame({"x": [0, 500, 1000]})      # much wider spread

    clipper = Clipper(columns=["x"], upper_quantile=0.9).fit(train)
    result = clipper.transform(unseen)

    assert clipper.upper_bounds_["x"] == 9.0
    assert result["x"].tolist() == [0, 9, 9]

def test_transform_without_fit_raises_attribute_error():
    df = pd.DataFrame({"x": [1, 2, 3]})
    clipper = Clipper(columns=["x"])
    with pytest.raises(AttributeError):
        clipper.transform(df)

def test_transform_does_not_mutate_input_frame():
    df = pd.DataFrame({"x": [0, 1, 2, 3, 100], "y": [1, 1, 1, 1, 1]})
    before = df.copy(deep=True)

    clipper = Clipper(columns=["x"], upper_quantile=0.5).fit(df)
    result = clipper.transform(df)

    assert result is not df
    pd.testing.assert_frame_equal(df, before)
    assert result["x"].max() == clipper.upper_bounds_["x"]
