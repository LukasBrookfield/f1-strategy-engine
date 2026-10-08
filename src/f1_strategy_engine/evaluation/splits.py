import pandas as pd


def race_split(
    df: pd.DataFrame,
    last_train_round: int = 18,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split one season by whole race, in time order."""
    is_train = df["round"] <= last_train_round
    train, test = df[is_train], df[~is_train]
    return train, test
