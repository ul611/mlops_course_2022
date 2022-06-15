import json
from typing import List

import click
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
import warnings

warnings.filterwarnings("ignore")


@click.command()
@click.argument("random_state", type=int, default=42)
@click.argument("input_paths", type=click.Path(), nargs=2)
@click.argument("output_paths", type=click.Path(), nargs=2)
def find_best_model_parameters(
    random_state: int, input_paths: List[str], output_paths: List[str]
):
    """
    Function finds the best parameters for Lasso and KNN estimators.
    :param random_state: Random state. By default, random_state = 42.
    :param input_paths: Paths to the train part of train dataset and its target.
    :param output_paths: Paths to the best parameters for Lasso model and to the best
    parameters for KNN model.
    :return: None.
    """
    input_path_X, input_path_y = input_paths
    output_path_lasso, output_path_knn = output_paths
    # Load data
    X_train = pd.read_csv(input_path_X)
    y_train = pd.read_csv(input_path_y)
    # Find the best parameters for regression
    grid_search_lasso = GridSearchCV(
        Lasso(),
        {
            "max_iter": range(10, 150, 10),
            "alpha": np.logspace(-9, -5),
            "random_state": [random_state],
        },
        scoring="r2",
    )
    grid_search_lasso.fit(X_train, y_train)
    # Find the best parameters for KNN
    grid_search_knn = GridSearchCV(
        KNeighborsRegressor(),
        {
            "metric": [
                "cosine",
                "euclidean",
                "manhattan",
                "chebyshev",
                "hamming",
                "canberra",
                "braycurtis",
            ],
            "weights": ["distance"],
            "n_neighbors": range(3, 8),
        },
        scoring="r2",
    )
    grid_search_knn.fit(X_train, y_train)
    # Save data
    with open(output_path_lasso, "w") as fd:
        json.dump(grid_search_lasso.best_params_, fd)
    with open(output_path_knn, "w") as fd:
        json.dump(grid_search_knn.best_params_, fd)


if __name__ == "__main__":
    find_best_model_parameters()
