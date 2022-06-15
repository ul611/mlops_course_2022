import click
import pandas as pd
import warnings

warnings.filterwarnings("ignore")


@click.command()
@click.argument("input_paths", type=click.Path(), nargs=3)
@click.argument("output_path", type=click.Path())
def read_merge_data(input_paths: str, output_path: str):
    """
    Function reads initial datasets and merges them into one dataset.
    :param input_paths: Paths to initial datasets.
    :param output_path: Path to full merged dataset.
    :return: None.
    """
    input_path_x_train, input_path_x_test, input_path_y_train = input_paths
    # Import data
    X_train = pd.read_csv(input_path_x_train)
    X_test = pd.read_csv(input_path_x_test)
    y_train = pd.read_csv(input_path_y_train)
    # Merge data
    X_train["dataset"] = "train"
    X_test["dataset"] = "test"
    df = pd.concat([pd.concat([X_train, y_train], axis=1), X_test])
    # Save data
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    read_merge_data()
