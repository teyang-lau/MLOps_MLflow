import logging
from urllib.parse import unquote, urlparse
import os
import tempfile
import mlflow
import click
import pandas as pd
from utils import onehotencode
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


@click.command(help="Preprocess HDB resale dataset and saves it as mlflow artifact")
@click.option("--filepath", type=str, default="data/resale-flat-prices-2022-jan.csv")
@click.option("--train-ratio", type=float, default=0.7)
@click.option("--val-ratio", type=float, default=0.2)
@click.option("--test-ratio", type=float, default=0.1)
def preprocess(filepath, train_ratio, val_ratio, test_ratio):
    with mlflow.start_run() as mlrun:
        artifact_uri = mlrun.info.artifact_uri
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        logger.addHandler(logging.StreamHandler())
        logger.addHandler(
            logging.FileHandler(
                unquote(urlparse(os.path.join(artifact_uri, "log.log")).path)
            )
        )

        logger.info("Reading data from {}".format(filepath))
        data = pd.read_csv(filepath)

        tmpdir = tempfile.mkdtemp()
        train_output_path = os.path.join(tmpdir, "train.csv")
        validation_output_path = os.path.join(tmpdir, "validation.csv")
        test_output_path = os.path.join(tmpdir, "test.csv")

        columns = [
            "resale_price",
            "town",
            "flat_type",
            "storey_range",
            "floor_area_sqm",
            "flat_model",
            "lease_commence_date",
            "remaining_lease",
        ]
        data = data[columns]

        data = data.replace(regex=[r".*[mM]aisonette.*", "foo"], value="Maisonette")
        data["remaining_lease"] = data["remaining_lease"].str.extract(
            r"(\d+)(?= years)"
        )
        data = data.astype({"remaining_lease": "int16"})

        logger.debug("Label encoding categorical columns - flat_type")
        flat_type_map = {
            "1 ROOM": 0,
            "2 ROOM": 1,
            "3 ROOM": 2,
            "4 ROOM": 3,
            "5 ROOM": 4,
            "MULTI-GENERATION": 5,
            "EXECUTIVE": 6,
        }
        data = data.replace({"flat_type": flat_type_map})
        # save mappings as artifacts!!!

        logger.debug("Label encoding categorical columns - storey_range")
        storey_range_le = LabelEncoder()
        data["storey_range"] = storey_range_le.fit_transform(data["storey_range"])
        # print(storey_range_le.classes_)

        logger.debug("One hot encoding categorical features")
        data, town_features, town_cat = onehotencode(data, "town")
        data, flat_model_features, flat_model_cat = onehotencode(data, "flat_model")
        # print(data.columns)
        # save encoders as artifacts!!!

        # log categorical features schema
        mlflow.log_dict(
            {
                "town": {
                    "categories": town_cat,
                    "ohe_features": town_features,
                },
                "flat_model": {
                    "categories": flat_model_cat,
                    "ohe_features": flat_model_features,
                },
            },
            os.path.join("cat_features_schema", "cat_features_schema.json"),
        )

        data_processed = data.copy()
        # Split into train, val, test
        y = data_processed["resale_price"]
        X = data_processed.drop(["resale_price"], axis=1)

        logger.debug("Splitting data into train, validation, and test sets")
        X_train, X_val_test, y_train, y_val_test = train_test_split(
            X,
            y,
            test_size=1 - train_ratio,
            random_state=2023,
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_val_test,
            y_val_test,
            test_size=(test_ratio / (test_ratio + val_ratio)),
            random_state=2023,
        )

        # set y as first column
        train_df = pd.concat([y_train, X_train], axis=1)
        val_df = pd.concat([y_val, X_val], axis=1)
        test_df = pd.concat([y_test, X_test], axis=1)
        # dataset_df = pd.concat([y, X], axis=1)

        logger.info("Train data shape after preprocessing: {}".format(train_df.shape))
        logger.info(
            "Validation data shape after preprocessing: {}".format(val_df.shape)
        )
        logger.info("Test data shape after preprocessing: {}".format(test_df.shape))

        train_df.to_csv(train_output_path, index=False)
        val_df.to_csv(validation_output_path, index=False)
        test_df.to_csv(test_output_path, index=False)

        # log train, validation, test df to artifact store
        train_artifact_uri = os.path.join(
            artifact_uri, "trainvaltest_data", "train.csv"
        )
        mlflow.log_artifact(train_output_path, "trainvaltest_data")
        logger.debug("Uploaded train data to artifact store: %s" % train_artifact_uri)
        val_artifact_uri = os.path.join(
            artifact_uri, "trainvaltest_data", "validation.csv"
        )
        mlflow.log_artifact(validation_output_path, "trainvaltest_data")
        logger.debug(
            "Uploaded validation data to artifact store: %s" % val_artifact_uri
        )
        test_artifact_uri = os.path.join(artifact_uri, "trainvaltest_data", "test.csv")
        mlflow.log_artifact(test_output_path, "trainvaltest_data")
        logger.debug(
            "Uploaded validation data to artifact store: %s" % test_artifact_uri
        )


if __name__ == "__main__":
    preprocess()
