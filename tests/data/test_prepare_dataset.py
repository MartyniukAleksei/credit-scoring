import pandas as pd
from credit_scoring.data.prepare_dataset import prepare_dataset

def test_drops_non_positive_age_rows():
    df = pd.DataFrame({
        "age": [25, 0, 40, -1],
        "MonthlyIncome": [3000, 4000, 5000, 6000],
    })
    result = prepare_dataset(df)
    assert (result['age'] > 0).all()
    assert len(result) == 2
    
def test_drops_exact_duplicate_rows():
    df = pd.DataFrame({
        "age": [30, 30, 45],
        "MonthlyIncome": [5000, 5000, 7000],
    })
    assert len(prepare_dataset(df)) == 2
    
def test_same_age_different_income_is_not_a_duplicate():
    df = pd.DataFrame({"age": [30, 30], "MonthlyIncome": [5000, 6000]})
    assert len(prepare_dataset(df)) == 2 


def test_noop_on_clean_data():
    df = pd.DataFrame({"age": [25, 40, 60], "MonthlyIncome": [3000, 4000, 5000]})
    result = prepare_dataset(df).reset_index(drop=True)
    pd.testing.assert_frame_equal(result, df.reset_index(drop=True))


def test_dedup_runs_after_age_filter_not_before():
    df = pd.DataFrame({"age": [0, 30, 30], "MonthlyIncome": [5000, 5000, 5000]})
    assert len(prepare_dataset(df)) == 1
    
    