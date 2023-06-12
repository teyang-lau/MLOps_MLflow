import logging
import hashlib
from urllib.parse import unquote, urlparse
import os
import sys
from typing import Literal, Union, Any
import mlflow
from mlflow.models.signature import infer_signature
import click
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


@click.command(help="Trains a random forest regressor")
@click.option("--datadir", type=str)
@click.option("--n-estimators", type=int, default=10)  # change to 100!
@click.option(
    "--max-features", type=click.Choice(["sqrt", "log2", None]), default="sqrt"
)
@click.option("--max-depth", type=click.IntRange(1), default=click.types.UNPROCESSED)
@click.option("--min-samples-split", type=int, default=2)
@click.option("--min-samples-leaf", type=int, default=1)
def train(
    datadir, n_estimators, max_features, max_depth, min_samples_split, min_samples_leaf
):
    with mlflow.start_run() as mlrun:
        artifact_uri = mlrun.info.artifact_uri

        # logging
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        logging.captureWarnings(True)
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S"
        )

        file_handler = logging.FileHandler(
            unquote(urlparse(os.path.join(artifact_uri, "log.log")).path)
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)

        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(logging.DEBUG)
        stdout_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(stdout_handler)

        # hash current file and log it as artifact
        curr_file_hash = hashlib.md5(open("scripts/train.py", "rb").read()).hexdigest()
        mlflow.log_text(curr_file_hash, "entrypoint_hash/hash.txt")

        train_path = os.path.join(datadir, "train.csv")
        validation_path = os.path.join(datadir, "validation.csv")
        logger.info("Reading train data from {}".format(train_path))
        train = pd.read_csv(train_path)
        logger.info("Reading validation data from {}".format(validation_path))
        validation = pd.read_csv(validation_path)

        y_train = train[["resale_price"]]
        X_train = train.drop("resale_price", axis=1)
        y_validation = validation[["resale_price"]]
        X_validation = validation.drop("resale_price", axis=1)

        rfr = RandomForestRegressor(
            n_estimators=n_estimators,
            max_features=max_features,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            criterion="absolute_error",
            n_jobs=-1,
            random_state=2023,
        )
        logger.debug("Fitting random forest regressor")
        rfr.fit(X_train, y_train.values.ravel())
        train_mae = mean_absolute_error(y_train, rfr.predict(X_train))
        validation_mae = mean_absolute_error(y_validation, rfr.predict(X_validation))
        logger.info("Train MAE: %.2f" % train_mae)
        logger.info("Validation MAE: %.2f" % validation_mae)
        mlflow.log_metric("train_mae", train_mae)
        mlflow.log_metric("validation_mae", validation_mae)
        signature = infer_signature(X_validation, rfr.predict(X_validation))
        mlflow.sklearn.log_model(
            rfr,
            "model",
            signature=signature,
            # input_example=X_train.iloc[0]
        )


if __name__ == "__main__":
    train()
