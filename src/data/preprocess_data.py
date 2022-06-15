import click
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")


@click.command()
@click.argument("input_path", type=click.Path())
@click.argument("output_path", type=click.Path())
def preprocess_data(input_path: str, output_path: str):
    """
    Function modifies target feature and removes outliers.
    :param input_path: Path to full dataset.
    :param output_path: Path to processed dataset.
    :return: None.
    """
    # Load data
    data = pd.read_csv(input_path)
    # Take the logarithm of y
    data["price"] = np.log(data["price"])
    # Take away outlier "bedrooms == 33" (its area for bedroom is too small: 1620 sq ft)
    data = data.query("bedrooms != 33").copy()
    # Save data
    data.to_csv(output_path, index=False)


if __name__ == "__main__":
    preprocess_data()
