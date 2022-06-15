from typing import List

import click
import joblib as jb
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings

warnings.filterwarnings("ignore")


@click.command()
@click.argument("input_paths", type=click.Path(), nargs=4)
@click.argument("output_paths", type=click.Path(), nargs=2)
def evaluate_model(input_paths: List[str], output_paths: List[str]):
    """
    Function evaluates fitted model and makes submission on test dataset.
    :param input_paths: Paths to test dataset, to validation dataset, to target of
    validation dataset and to fitted model.
    :param output_paths: Paths to save scores and submission on test dataset.
    :return: None.
    """
    (
        input_path_X_test,
        input_path_X_val,
        input_path_y_val,
        input_path_model,
    ) = input_paths
    output_path_score, output_path_submission = output_paths
    # Load data
    model = jb.load(input_path_model)
    X_test = pd.read_csv(input_path_X_test)
    X_val = pd.read_csv(input_path_X_val)
    y_val = pd.read_csv(input_path_y_val)
    # Make validation
    y_predicted = model.predict(X_val)
    score = pd.DataFrame(
        dict(
            mae=mean_absolute_error(y_val, y_predicted),
            rmse=mean_squared_error(y_val, y_predicted),
            r2_score=r2_score(y_val, y_predicted),
        ),
        index=[0],
    )
    score.to_csv(output_path_score, index=False)
    # Make a prediction
    y_test_predicted = model.predict(X_test)
    # Make a submission
    submission = pd.DataFrame(np.exp(y_test_predicted), columns=["price"]).reset_index()
    submission.columns = ["Id", "price"]
    submission.set_index("Id").to_csv(output_path_submission)


if __name__ == "__main__":
    evaluate_model()
