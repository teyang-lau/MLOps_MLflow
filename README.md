
# <div align="center">End-to-end MLOps Using MLflow</div>

<div align="center">

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-blue.svg)](https://www.python.org/)
[![made-with-MLflow](https://img.shields.io/badge/Made%20with-MLflow-9cf.svg?)](https://mlflow.org/)
[![made-with-Markdown](https://img.shields.io/badge/Made%20with-Markdown-1f425f.svg)](http://commonmark.org)
[![Generic badge](https://img.shields.io/badge/STATUS-INPROGRESS-<COLOR>.svg)](https://shields.io/)
[![GitHub license](https://img.shields.io/github/license/teyang-lau/HDB_Resale_Prices.svg)](https://github.com/teyang-lau/YOListenO/blob/main/LICENSE)
<br><br><br>

</div>

<p align="center">
  <img src="./images/mlflow_mlops_chart.PNG" width="900">
  <br><br>

## About

* Developing and implementing a MLOps pipeline for a ML problem, leveraging the capabilities of MLflow

    * Robust automated workflow that covers the entire ML lifecycle, including data validation, processing, model training, evaluation, validation and deployment

    * Monitoring and logging functionalities within the pipeline along with a ML metadata store to provide insights into model performance and ensure data lineage and reproducibility
</p>

<br>

## How to run
1. Run
    ```
    mlflow experiments create -n experiment_name # create a new experiment
    mlflow run --experiment-name experiment_name -P eval_mae_threshold=150000 .

<br>

## To Do
- [ ] Data & Model Validation
- [ ] Jensen Shannon divergence for data drift
- [ ] Hyperparameter tuning
- [ ] Log immediate parent run
- [ ] Pass multiple params into subsequent runs down the pipeline
- [ ] Get mlflow logs written in log file
- [ ] Using caching for runs that were ran before. Still does not work for dependant runs. For eg., if data in preprocess changes, only preprocess run will run, while training will not, since train script is not modified
- [ ] Try to keep track of whether a run was modified before. If so, all subsequent runs will have to run! (maybe have a var to keep track if previous runs were ran. If so, all subsequent children runs must run again)
- [x] Try moving all scripts to scripts folder and make it work
- [x] Create experiments name for the pipeline

<br>

## Logs
* 9 Jun — logged all `run_id`s in main.py and created function to infer basic schema of pandas dataframe for data validation. Wrote prelim schema checks that will check for data types and range of values for numeric columns and domain values for object columns 
* 8 Jun — evaluation.py and model_validate.py works. Also integrated SHAP model explanability check into model validation step, which will log the shap plots and the explainer
* 7 Jun — Got MLproject pipeline with just preprocess step working. Can just run `mlflow run .` in directory. Created hash checking to make the run runs if code changes. train.py works for taking data from previous step and training and validating model
* 6 Jun — preprocess.py works to preprocess and split data into train, val, test. Logged category features schema and data artifacts using mlflow. Saved python logs to log file


