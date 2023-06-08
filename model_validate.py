import logging
import hashlib
from urllib.parse import unquote, urlparse
import os
import mlflow
import click
import pandas as pd
import shap
import tempfile
import matplotlib.pyplot as plt


@click.command(help="Validate the trained model")
@click.option("--datadir", type=str)
@click.option("--modeldir", type=str)
@click.option("--test-score", type=float)
@click.option("--eval-threshold", type=float)
def model_validate(datadir, modeldir, test_score, eval_threshold):
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
        curr_file_hash = hashlib.md5(open("model_validate.py", "rb").read()).hexdigest()
        mlflow.log_text(curr_file_hash, "entrypoint_hash/hash.txt")

        # check if test score satisfy threshold, if no, end model validation
        if test_score > eval_threshold:
            logger.info(
                "Model did not pass threshold. Model performance score {} is larger than {}".format(
                    test_score, eval_threshold
                )
            )
            mlflow.set_tags({"validation_status": "fail"})
            return

        logger.info("Model has passed threshold")
        mlflow.set_tags({"validation_status": "pass"})

        # load test data
        test_path = os.path.join(datadir, "test.csv")
        logger.info("Reading test data from {}".format(test_path))
        test = pd.read_csv(test_path)
        y_test = test[["resale_price"]]
        X_test = test.drop(["resale_price"], axis=1)

        # load model
        logger.info("Loading model from {}".format(modeldir))
        model = mlflow.sklearn.load_model(modeldir)

        # model bias check

        # model explanability check
        # shap not working with numpy > 1.24
        # check whether to use log_explainer, log_explanation, or save_explainer
        logger.debug("Performing SHAP computations for model explanability")
        explainer = shap.Explainer(model.predict, X_test)
        shap_values = explainer(X_test)
        # log the shap plots
        shap.plots.beeswarm(shap_values, show=False)
        fig = plt.gcf()
        fig.tight_layout()
        tmpdir = tempfile.mkdtemp()
        fig.savefig(os.path.join(tmpdir, "beeswarm_plot.png"))
        plt.clf()
        shap.plots.bar(shap_values, show=False)
        fig = plt.gcf()
        fig.tight_layout()
        fig.savefig(os.path.join(tmpdir, "summary_bar_plot.png"))
        mlflow.log_artifacts(tmpdir, artifact_path="model_explanations_shap")
        mlflow.shap.log_explainer(explainer, "model_explanations_shap/explainer")


# check new model performs better than current model or baseline model
# test model calculation. eg. for neural networks, check the weights, no anomalies etc
# check for feature importance, fairness (compute performance metrics on all slices of data, https://medium.com/responsibleml/what-fairness-in-regression-285e3f2a549e)
# compare feature importance between train and test set https://stats.stackexchange.com/questions/475567/permutation-feature-importance-on-train-vs-validation-set
# post-training dat and model bias https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-measure-post-training-bias.html
# adversarial attacks
# sensitivity analysis (can mean vulnerability)
# see how model reacts to data which it has never seen before (eg. house with floor_area_sqm of 300?)
# simulate data and see what model predicts
# Subgroup analysis
# test model outputs

if __name__ == "__main__":
    model_validate()
