import click
from typing import List
import pandas as pd
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")


@click.command()
@click.argument("input_paths", type=click.Path(), nargs=2)
@click.argument("output_paths", type=click.Path(), nargs=2)
def scale_features(input_paths: List[str], output_paths: List[str]):
    """
    Function processes standard scaling procedure on train and test datasets.
    :param input_paths: Paths to train and test datasets.
    :param output_paths: Paths to scaled train and test datasets.
    :return: None.
    """
    input_path_X, input_path_X_to_predict = input_paths
    output_path_X, output_path_X_to_predict = output_paths
    # Load data
    X = pd.read_csv(input_path_X)
    X_to_predict = pd.read_csv(input_path_X_to_predict)
    # Scale data with Standard Scaler
    scaler = StandardScaler()
    scaler.fit(X)
    X = pd.DataFrame(scaler.transform(X), columns=X.columns)
    X_to_predict = pd.DataFrame(
        scaler.transform(X_to_predict), columns=X_to_predict.columns
    )
    # Save data
    X.to_csv(output_path_X, index=False)
    X_to_predict.to_csv(output_path_X_to_predict, index=False)


if __name__ == "__main__":
    scale_features()
