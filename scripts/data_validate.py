import logging
import hashlib
from urllib.parse import unquote, urlparse
import os
import sys
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import RunStatus
import click
import pandas as pd
from utils import infer_schema, compare_data_to_schema


@click.command(help="Preprocess HDB resale dataset and saves it as mlflow artifact")
@click.option("--filepath", type=str, default="data/resale-flat-prices-2022-jan.csv")
def data_validate(filepath):
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
            open("scripts/data_validate.py", "rb").read()
        ).hexdigest()
        mlflow.log_text(curr_file_hash, "entrypoint_hash/hash.txt")

        logger.info("Reading data from {}".format(filepath))
        data = pd.read_csv(filepath)

        exp_id = mlrun.info.experiment_id
        client = MlflowClient()
        all_runs = reversed(client.search_runs([exp_id]))

        # infer schema
        schema_curr = infer_schema(data)
        found_old_schema = False

        if all_runs:  # experiment have previous runs:
            # validate current data with newest schema
            # get latest main run, get its data_validate run, and check for the schema, if any
            for run in all_runs:
                tags = run.data.tags
                if tags.get("mlflow.project.entryPoint") == "main" and tags.get(
                    "data_validate"
                ):
                    run_id = tags.get("data_validate")
                    old_run = client.get_run(run_id)
                    if old_run.info.to_proto().status != RunStatus.FINISHED:
                        break
                    schema_old_path = os.path.join(
                        client.get_run(run_id).info.artifact_uri,
                        "data_schema/schema.json",
                    )
                    # might fail here if there is no schema dict found
                    schema_old = mlflow.artifacts.load_dict(schema_old_path)
                    found_old_schema = True
                    # data.at[0, "resale_price"] = 10000000 # for testing
                    # data.rename({"month": "date"}, axis=1, inplace=True) # for testing
                    # data.at[0, "flat_type"] = "Bungalow" # for testing
                    data_val_status = compare_data_to_schema(data, schema_old)
                    if data_val_status == "Failed":
                        logger.error("Data validation with previous schema failed!")
                        raise RuntimeError(
                            "Data validation with previous schema failed!"
                        )
                    logger.info("Data validation with previous schema passed!")
                    break

        if not found_old_schema:
            logger.info(
                "Found no previous schema from successful data validation runs. Proceeding to log current schema ..."
            )
        else:
            logger.info("Logging current schema ...")
        mlflow.log_dict(schema_curr, "data_schema/schema.json")

        # check for missing data
        missing = (
            pd.concat([data.isnull().any(), data.isnull().sum()], axis=1)
            .T.apply(tuple)
            .to_dict("list")
        )
        for col, [miss, num] in missing.items():
            if miss:
                logger.error("Column `{}` has ({}) missing data".format(col, num))
                logger.error("Data validation for missing data has failed!")
                raise RuntimeError("Data validation for missing data has failed!")
        logger.info("Data validation for missing data has passed!")

        mlflow.set_tags({"validation_status": "pass"})


if __name__ == "__main__":
    data_validate()


"""
Missing data, such as features with empty values.
Labels treated as features, so that your model gets to peek at the right answer during training.
Features with values outside the range you expect.
Data anomalies.If you cannot immediately regenerate your protos, some other possible workarounds are:
 1. Downgrade the protobuf package to 3.20.x or lower.
 2. Set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python (but this will use pure-Python parsing and will be much slower).
Transfer learned model has preprocessing that does not match the training data.
"""
