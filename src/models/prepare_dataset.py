from typing import List
import click
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore")


@click.command()
@click.argument("random_state", type=int, default=42)
@click.argument("input_paths", type=click.Path(), nargs=2)
@click.argument("output_paths", type=click.Path(), nargs=4)
def prepare_dataset(random_state: int, input_paths: List[str], output_paths: List[str]):
    """
    Function divides dataset to train and validation parts.
    :param random_state: Random state. By default, random_state = 42.
    :param input_paths: Paths to full train dataset and to its target.
    :param output_paths: Paths to train part of train dataset, to validation part, and
    to their targets, respectively.
    :return: None.
    """
    input_path_X, input_path_y = input_paths
    (
        output_path_X_train,
        output_path_X_val,
        output_path_y_train,
        output_path_y_val,
    ) = output_paths
    # Load data
    X_train = pd.read_csv(input_path_X)
    y_train = pd.read_csv(input_path_y)
    # Divide dataset to train and validation
    X_train_train, X_val, y_train_train, y_val = train_test_split(
        X_train, y_train, random_state=random_state
    )
    # Save data
    X_train_train.to_csv(output_path_X_train, index=False)
    y_train_train.to_csv(output_path_y_train, index=False)
    X_val.to_csv(output_path_X_val, index=False)
    y_val.to_csv(output_path_y_val, index=False)


if __name__ == "__main__":
    prepare_dataset()
