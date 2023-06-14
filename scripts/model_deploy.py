import os
import click
import yaml
from mlflow.tracking import MlflowClient


@click.command(help="Deploy a model to staging or production or archive it")
@click.option("--modelname", type=str)
@click.option("--version", type=int)
@click.option("--stage", type=str)
@click.option("--archive_existing", type=bool, default=False)
def model_transition(
    modelname: str,
    version: int,
    stage: str,
    archive_existing: bool = False,
):
    client = MlflowClient()
    client.transition_model_version_stage(
        name=modelname,
        version=version,
        stage=stage,
        archive_existing_versions=archive_existing,
    )

    if stage in ["Staging", "Production"]:
        model_yaml = "mlruns/models/{}/version-{}/meta.yaml".format(modelname, version)
        with open(model_yaml, "r") as f:
            meta = yaml.safe_load(f)
        model_uri = meta.get("source")
        os.system(
            "mlflow models serve -m models:/{}/{} -p 1234".format(modelname, stage)
        )


if __name__ == "__main__":
    model_transition()
