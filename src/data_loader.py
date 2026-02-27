import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
TARGET_COL = "default.payment.next.month"

def load_credit_data(csv_path: str):
    df = pd.read_csv(csv_path)

    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(int)

    return X, y


def make_split(X, y, test_size=0.2):
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y
    )
