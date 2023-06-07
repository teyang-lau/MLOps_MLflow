## To Do
- [ ] Pass multiple params into subsequent runs down the pipeline
- [x] Create experiments name for the pipeline

## Logs
* 7 Jun — Got MLproject pipeline with just preprocess step working. Can just run `mlflow run .` in directory. Created hash checking to make the run runs if code changes
* 6 Jun — preprocess.py works to preprocess and split data into train, val, test. Logged category features schema and data artifacts using mlflow. Saved python logs to log file.

## How to run
1. Run
    ```
    mlflow experiments create -n experiment_name # create a new experiment
    mlflow run --experiment-name experiment_name -P als_max_iter=2 .
    ```