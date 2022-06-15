import click
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")


@click.command()
@click.argument("input_path", type=click.Path())
@click.argument("output_path", type=click.Path())
def modify_features(input_path: str, output_path: str):
    """
    Function processes features modification.
    :param input_path: Path to full dataset.
    :param output_path: Path to processed dataset.
    :return: None.
    """
    # Load data
    data = pd.read_csv(input_path)
    # As there are a few of values for "view", "sqft_basement", "yr_renovated" features,
    # Let's make them boolean
    for col in ["view", "sqft_basement", "yr_renovated"]:
        data[col + "_bool"] = (data[col] > 0).astype(int)
        data.drop(col, axis=1, inplace=True)
    # Take the square root of area features and then take their logarithm
    for col in data.columns:
        if col.startswith("sqft") and not col.endswith("bool"):
            data[col[2:]] = np.log(np.sqrt(data[col]))
            data.drop(col, axis=1, inplace=True)
    # Take sale date and calculate the number of days from the earliest sale
    # to that date
    data.date = pd.to_datetime(
        data.date.str.split("T", expand=True)[0], format="%Y%m%d"
    )
    min_data = data.date.min()
    data.date = (data.date - min_data).apply(lambda x: x.days)
    # Save data
    data.to_csv(output_path, index=False)


if __name__ == "__main__":
    modify_features()
