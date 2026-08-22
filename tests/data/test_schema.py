import pandas as pd
import pytest
from credit_scoring.data.schema import train_schema

@pytest.fixture
def raw_train():
    df = pd.read_csv('data/raw/cs-training.csv')
    return df.drop_duplicates()

def test_schema_passes_on_real_data(raw_train):
    train_schema.validate(raw_train, lazy=True)
