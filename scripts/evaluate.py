import logging
import hashlib
from urllib.parse import unquote, urlparse
import os
import sys
import mlflow
import click
import pandas as pd
from sklearn.metrics import mean_absolute_error


@click.command(help="Evaluate the trained model")
@click.option("--datadir", type=str)
@click.option("--modeldir", type=str)
def evaluate(datadir, modeldir):
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
        curr_file_hash = hashlib.md5(
            open("scripts/evaluate.py", "rb").read()
        ).hexdigest()
        mlflow.log_text(curr_file_hash, "entrypoint_hash/hash.txt")

        # load test data
        test_path = os.path.join(datadir, "test.csv")
        logger.info("Reading test data from {}".format(test_path))
        test = pd.read_csv(test_path)
        y_test = test[["resale_price"]]
        X_test = test.drop(["resale_price"], axis=1)

        # load model
        logger.info("Loading model from {}".format(modeldir))
        model = mlflow.sklearn.load_model(modeldir)

        # evaluate on test set
        test_mae = mean_absolute_error(y_test, model.predict(X_test))
        logger.info("Test MAE: %.2f" % test_mae)
        mlflow.log_metric("test_mae", test_mae)


if __name__ == "__main__":
    evaluate()
