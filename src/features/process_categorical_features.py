import click
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings("ignore")


@click.command()
@click.argument("input_path", type=click.Path())
@click.argument("output_path", type=click.Path())
def process_categorical_features(input_path: str, output_path: str):
    """
    Function processes label encoding on categorical features.
    :param input_path: Path to full dataset.
    :param output_path: Path to processed dataset.
    :return: None.
    """
    # Load data
    data = pd.read_csv(input_path)
    # Relabel categorical features
    cat_features = [
        "zipcode",
        # 'view_bool', 'sqft_basement_bool', 'waterfront', 'yr_renovated_bool'
    ]
    for col in cat_features:
        le = LabelEncoder()
        le.fit(data[col])
        data[col] = le.transform(data[col])
    # Save data
    data.to_csv(output_path, index=False)


if __name__ == "__main__":
    process_categorical_features()
