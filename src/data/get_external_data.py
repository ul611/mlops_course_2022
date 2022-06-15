import json
import time
import click
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")


@click.command()
@click.argument("scroll_pause_time", type=int, default=1)
@click.argument("input_path", type=click.Path())
@click.argument("output_path", type=click.Path())
def get_external_data(scroll_pause_time: int, input_path: str, output_path: str):
    """
    Function collects demographic data based on zipcode into dictionary and prints
    the progress bar to STDOUT.
    :param scroll_pause_time: Pause time between webpages opening.
    By default, scroll_pause_time = 1.
    :param input_path: Path to full dataset.
    :param output_path: Path to the dictionary with information based on zipcodes.
    :return: None.
    """
    # Load data
    data = pd.read_csv(input_path)
    # Collect demographic data based on zipcode into dictionary
    d_zipcodes_info = {}
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
    driver = webdriver.Chrome()

    for zipcode in tqdm(data.zipcode.unique()):
        url = f"https://www.unitedstateszipcodes.org/{zipcode}/"
        driver.get(url)

        time.sleep(scroll_pause_time)
        list_p_element = driver.find_elements(By.XPATH, "//tr")

        d_zipcodes_info[zipcode] = {}
        fulltxt = []

        # Find the beginning of useful information
        for i_start, el in enumerate(list_p_element):
            txt = el.text
            if txt.startswith("Population"):
                break

        i = -1
        # Collect useful information
        for el in list_p_element:
            i += 1
            if i < i_start or i_start + 8 < i:
                continue
            txt = el.text
            if txt:
                fulltxt += [txt]
        # Extract data and write it to dictionary
        for txt, field in zip(fulltxt, info_fields):
            value = float(
                txt.split(field)[1]
                .replace(":", "")
                .strip()
                .split()[0]
                .replace(",", "")
                .replace("$", "")
            )
            d_zipcodes_info[zipcode][field] = value
    # Write dictionary into file
    d_zipcodes_modified_keys = {str(key): val for key, val in d_zipcodes_info.items()}
    with open(output_path, "w") as fd:
        json.dump(d_zipcodes_modified_keys, fd)


if __name__ == "__main__":
    get_external_data()
