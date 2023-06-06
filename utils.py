import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def onehotencode(df, col: str):
    """One-hot encode a column in a dataframe

    Args:
        df (pd.DataFrame): pandas dataframe
        col (str): categorical column to one-hot encode

    Returns:
        df: dataframe with ohe features
        ohe_features: names of the ohe features
        categories: all the categories of the ohe column
    """
    ohe = OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)
    ohe_df = pd.DataFrame(ohe.fit_transform(df[col].values.reshape(-1, 1)))
    ohe_features = [x.replace("x0_", "") for x in ohe.get_feature_names_out()]
    ohe_df.columns = ohe_features
    categories = ohe.categories_[0].tolist()
    df.drop(col, axis=1, inplace=True)
    df = pd.concat([df, ohe_df], axis=1)

    return df, ohe_features, categories
