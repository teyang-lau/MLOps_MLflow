"""
Downloads the MovieLens dataset, ETLs it into Parquet, trains an
ALS model, and uses the ALS model to train a Keras neural network.

See README.rst for more details.
"""

import click
import os
import hashlib

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.utils import mlflow_tags
from mlflow.entities import RunStatus
from mlflow.utils.logging_utils import eprint

from mlflow.tracking.fluent import _get_experiment_id


def _already_ran(entry_point_name, parameters, git_commit, experiment_id=None):
    """Best-effort detection of if a run with the given entrypoint name,
    parameters, and experiment id already ran. The run must have completed
    successfully and have at least the parameters provided.
    """
    experiment_id = experiment_id if experiment_id is not None else _get_experiment_id()
    curr_file_hash = hashlib.md5(
        open(entry_point_name + ".py", "rb").read()
    ).hexdigest()
    client = MlflowClient()
    all_runs = reversed(client.search_runs([experiment_id]))
    for run in all_runs:
        tags = run.data.tags
        if tags.get(mlflow_tags.MLFLOW_PROJECT_ENTRY_POINT, None) != entry_point_name:
            continue
        match_failed = False
        for param_key, param_value in parameters.items():
            run_value = run.data.params.get(param_key)
            if run_value != param_value:
                match_failed = True
                break
        if match_failed:
            continue

        if run.info.to_proto().status != RunStatus.FINISHED:
            eprint(
                ("Run matched, but is not FINISHED, so skipping (run_id=%s, status=%s)")
                % (run.info.run_id, run.info.status)
            )
            continue

        previous_version = tags.get(mlflow_tags.MLFLOW_GIT_COMMIT, None)
        if git_commit != previous_version:
            eprint(
                (
                    "Run matched, but has a different source version, so skipping "
                    "(found=%s, expected=%s)"
                )
                % (previous_version, git_commit)
            )
            continue

        # checking git commit above does not work for locally changed files.
        # so check hash of previous run's entrypoint file with with current entrypoint file hash
        try:
            previous_hash = mlflow.artifacts.load_text(
                os.path.join(run.info.artifact_uri, "entrypoint_hash/hash.txt")
            )
        except Exception:
            continue
        if curr_file_hash != previous_hash:
            continue

        return client.get_run(run.info.run_id)
    eprint("No matching run has been found.")
    return None


# TODO(aaron): This is not great because it doesn't account for:
# - changes in code (can save .py files as artifacts and compare hashes against current one)
# - changes in dependant steps
def _get_or_run(entrypoint, parameters, git_commit, use_cache=True):
    # get hash of current run entrypoint file
    existing_run = _already_ran(entrypoint, parameters, git_commit)
    if use_cache and existing_run:
        print(
            "Found existing run for entrypoint={} and parameters={}".format(
                entrypoint, parameters
            )
        )
        return existing_run
    print(
        "Launching new run for entrypoint={} and parameters={}".format(
            entrypoint, parameters
        )
    )
    submitted_run = mlflow.run(
        ".", entrypoint, parameters=parameters, env_manager="local"
    )
    print("\n" * 3)

    return MlflowClient().get_run(submitted_run.run_id)


@click.command()
@click.option("--eval-mae-threshold", default=150000, type=int)
@click.option("--keras-hidden-units", default=20, type=int)
@click.option("--max-row-limit", default=100000, type=int)
def pipeline(eval_mae_threshold, keras_hidden_units, max_row_limit):
    # Note: The entrypoint names are defined in MLproject. The artifact directories
    # are documented by each step's .py file.
    with mlflow.start_run() as active_run:
        git_commit = active_run.data.tags.get(mlflow_tags.MLFLOW_GIT_COMMIT)

        # preprocess run
        preprocess_run = _get_or_run(
            "preprocess",
            {"filepath": "data/resale-flat-prices-2022-jan.csv"},
            git_commit,
        )
        datadir_uri = os.path.join(
            preprocess_run.info.artifact_uri, "trainvaltest_data"
        )

        # train run
        train_run = _get_or_run("train", {"datadir": datadir_uri}, git_commit)
        # modeldir_uri = os.path.join(train_run.info.artifact_uri, "model")
        modeldir_uri = "runs:/{}/model".format(train_run.info.run_id)

        # evaluate run
        evaluate_run = _get_or_run(
            "evaluate", {"datadir": datadir_uri, "modeldir": modeldir_uri}, git_commit
        )

        # model validation run
        test_mae = evaluate_run.data.metrics.get("test_mae", float("inf"))
        model_validation_run = _get_or_run(
            "model_validate",
            {
                "datadir": datadir_uri,
                "modeldir": modeldir_uri,
                "test_score": test_mae,
                "eval_threshold": eval_mae_threshold,
            },
            git_commit,
        )

        # register model based on condition (checked in validation run)
        if model_validation_run.data.tags.get("validation_status") != "pass":
            return
        # register
        model_version = mlflow.register_model(
            modeldir_uri,
            "random_forest_regressor_HDB_Resale_Price",
        )
        # print("Name: {}, Version: {}".format(model_version.name, model_version.version))


if __name__ == "__main__":
    pipeline()
