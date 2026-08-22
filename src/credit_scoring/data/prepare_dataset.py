import pandas as pd
import logging

logger = logging.getLogger(__name__)

def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    n_before = len(df)
    df = df[df['age']>0]
    logger.info("dropped %d rows with age <= 0", n_before - len(df))
    
    n_before_dedup = len(df)
    df = df.drop_duplicates()
    logger.info("dropped %d duplicate rows", n_before_dedup - len(df))
    
    return df
    
    
if __name__ == '__main__': 
    raw = pd.read_csv("../data/raw/cs-training.csv")
    clean = prepare_dataset(raw)
    clean.to_csv("..data/precessed/cs-training-clean.csv", index=False)