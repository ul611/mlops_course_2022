from typing import List
import json

import click
import joblib as jb
import pandas as pd
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.neighbors import KNeighborsRegressor
import warnings

warnings.filterwarnings("ignore")


@click.command()
@click.argument("input_paths", type=click.Path(), nargs=4)
@click.argument("output_path", type=click.Path())
def train_model(input_paths: List[str], output_path: str):
    """
    Function fits stacked (KNN and Lasso regressors) estimator.
    :param input_paths: Paths to train part of dataset, to its target, to the best
    parameters for Lasso model, to the best parameters for KNN model.
    :param output_path: Path to trained model.
    :return: None.
    """
    input_path_X, input_path_y, input_path_lasso, input_path_knn = input_paths
    # Load data
    with open(input_path_lasso, "r") as fd:
        best_params_lasso = json.load(fd)
    with open(input_path_knn, "r") as fd:
        best_params_knn = json.load(fd)
    X_train = pd.read_csv(input_path_X)
    y_train = pd.read_csv(input_path_y)
    # Define the base models
    level0 = list()
    level0.append(("knn", KNeighborsRegressor(**best_params_knn)))
    level0.append(("lasso", Lasso(**best_params_lasso)))
    # Define meta learner model
    level1 = LinearRegression()
    # Define the stacking ensemble
    model = StackingRegressor(estimators=level0, final_estimator=level1, cv=5)
    # Fit the model
    model.fit(X_train, y_train)
    # Save data
    jb.dump(model, output_path)


if __name__ == "__main__":
    train_model()
