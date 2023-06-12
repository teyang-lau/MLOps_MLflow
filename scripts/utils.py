import logging
import warnings
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from mlflow import MlflowClient


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


def fetch_logged_data(run_id):
    # params, metrics, tags, artifacts = fetch_logged_data(run_id)
    client = MlflowClient()
    data = client.get_run(run_id).data
    tags = {k: v for k, v in data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in client.list_artifacts(run_id, "model")]
    return data.params, data.metrics, tags, artifacts


def infer_schema(df):
    schema = {}
    for col in df.columns:
        dt = df[col].dtype
        if dt.kind in "iufc":
            schema[col] = {
                "type": str(dt),
                "min": float(df[col].min()),
                "max": float(df[col].max()),
            }
        else:  # object
            schema[col] = {"type": str(dt), "domain": sorted(list(set(df[col])))}

    return schema


def compare_data_to_schema(data, schema):
    # logger = logging.getLogger(__name__)
    # warnings_logger = logging.getLogger("py.warnings")
    logging.captureWarnings(True)

    data_cols = data.columns
    for col in schema.keys():
        if col not in data_cols:
            print("Column `{}` is missing from dataset".format(col))
            return "Failed"
    for col in data_cols:
        if col not in schema:
            print("Column `{}` does not exist in schema".format(col))
            return "Failed"
        dt = data[col].dtype
        if dt.kind in "iufc":  # if numeric
            if dt != schema[col]["type"]:
                print(
                    'Expected column `{}` to be "{}" but got "{}"'.format(
                        col, schema[col]["type"], dt
                    )
                )
                print("Terminating data validation ...")
                return "Failed"
            d_max, d_min = data[col].max(), data[col].min()
            if d_max > schema[col]["max"]:
                warnings.warn(
                    "Column `{}` has values ({}) higher than max of schema ({})".format(
                        col, d_max, schema[col]["max"]
                    )
                )
            if d_min < schema[col]["min"]:
                warnings.warn(
                    "Column `{}` has values ({}) lower than min of schema ({})".format(
                        col, d_min, schema[col]["min"]
                    )
                )
        else:  # if object
            if dt != schema[col]["type"]:
                print(
                    'Expected column `{}` to be "{}" but got "{}"'.format(
                        col, schema[col]["type"], dt
                    )
                )
                print("Terminating data validation ...")
                return "Failed"
            # check if each columns contain values not present in previous domain
            old_set = set(schema[col]["domain"])
            diff = sorted(set(data[col]).difference(old_set))
            if diff:
                warnings.warn(
                    "Column `{}` domain contains {} that are not present in schema".format(
                        col, diff
                    )
                )

    return "Passed"
