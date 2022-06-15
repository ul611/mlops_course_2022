import json
from typing import List
import click
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")


@click.command()
@click.argument("input_paths", type=click.Path(), nargs=2)
@click.argument("output_path", type=click.Path())
def add_external_features(input_paths: List[str], output_path: str):
    """
    Function adds features related to the information based on zipcodes.
    :param input_paths: Paths to full dataset and to the dictionary with information
    based on zipcodes.
    :param output_path: Path to processed dataset.
    :return: None.
    """
    input_df_path, input_dict_path = input_paths
    info_fields = [
        "Population",
        "Population Density",
        "Housing Units",
        "Median Home Value",
        "Land Area",
        "Water Area",
        "Occupied Housing Units",
        "Median Household Income",
        "Median Age",
    ]
    # Load data
    data = pd.read_csv(input_df_path)
    # Load dictionary with addition info based on zipcodes
    with open(input_dict_path, "r") as fd:
        d_zipcodes_modified_keys = json.load(fd)
    d_zipcodes_info = {int(key): val for key, val in d_zipcodes_modified_keys.items()}
    # Create features based on external info
    for field in info_fields:
        colname = "_".join(field.split())
        data[colname] = data.zipcode.apply(lambda x: d_zipcodes_info[x][field])
    # Take the square root of population number
    data["Population"] = np.sqrt(data["Population"])
    # Create the feature with occupied house units ratio
    data["pers_houses_occupied"] = data.Occupied_Housing_Units / data.Housing_Units
    # Save data
    data.to_csv(output_path, index=False)


if __name__ == "__main__":
    add_external_features()
