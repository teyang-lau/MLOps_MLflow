import logging
import hashlib
from urllib.parse import unquote, urlparse
import os
import tempfile
import json
import mlflow
from mlflow.tracking import MlflowClient
import click
import pandas as pd
from utils import infer_schema


@click.command(help="Preprocess HDB resale dataset and saves it as mlflow artifact")
@click.option("--filepath", type=str, default="data/resale-flat-prices-2022-jan.csv")
@click.option("--data-schema", type=str, default="data/resale-flat-prices-2022-jan.csv")
def data_validate(filepath, data_schema):
    with mlflow.start_run() as mlrun:
        artifact_uri = mlrun.info.artifact_uri
        logger = logging.getLogger("mlflow")
        logger.setLevel(logging.DEBUG)
        logger.addHandler(
            logging.FileHandler(
                unquote(urlparse(os.path.join(artifact_uri, "log.log")).path)
            )
        )

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
        if not all_runs:  # experiment does not have previous runs
            # infer schema and log it
            schema = infer_schema(data)
            # with open("schema.json", "w") as file:
            #     json.dump(schema, file)
            mlflow.log_dict(schema, "data_schema/schema.json")
        else:
            # validate current data with newest schema
            # get latest data_validate run, check if there is schema in there. if no, means it used a previous
            # cached run. So get to the cached run folder (how to do this?)
            pass


"""
Missing data, such as features with empty values.
Labels treated as features, so that your model gets to peek at the right answer during training.
Features with values outside the range you expect.
Data anomalies.If you cannot immediately regenerate your protos, some other possible workarounds are:
 1. Downgrade the protobuf package to 3.20.x or lower.
 2. Set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python (but this will use pure-Python parsing and will be much slower).
Transfer learned model has preprocessing that does not match the training data.
"""
