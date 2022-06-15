import itertools
from collections import Counter
import click
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")


@click.command()
@click.argument("corr_thres", type=float, default=0.8)
@click.argument("input_path", type=click.Path())
@click.argument("output_paths", type=click.Path(), nargs=3)
def process_correlated_features(corr_thres: float, input_path: str, output_paths: str):
    """
    Function detects and filters correlated features.
    :param corr_thres: Threshold for correlation coefficient.
    :param input_path: Path to full dataset.
    :param output_paths: Paths to the train dataset, to its target, and to the test
    dataset.
    :return: None.
    """
    output_path_X, output_path_y, output_path_X_to_predict = output_paths
    # Load data
    data = pd.read_csv(input_path)
    # Divide data to train dataset, predict dataset and target feature
    X_to_predict = data.query("price.isna()").drop(
        [
            # 'lat', 'long', 'zipcode',
            "dataset",
            "price",
        ],
        axis=1,
    )
    X = data.query("~price.isna()").drop(
        [
            # 'lat', 'long', 'zipcode',
            "dataset",
            "price",
        ],
        axis=1,
    )
    y = data.query("~price.isna()")[["price"]]
    # Detect correlated features
    correlated_features = pd.DataFrame(
        X.corr(method="pearson")
        .abs()
        .unstack()
        .reset_index()
        .query("level_0 != level_1")
        .sort_values(0, ascending=False)
    )
    # Form pairs of correlated features
    correlated_features["pairs"] = correlated_features.apply(
        lambda x: tuple(sorted([x.level_0, x.level_1])), axis=1
    )
    correlated_features = (
        correlated_features.drop_duplicates(subset="pairs")
        .drop(["level_0", "level_1"], axis=1)
        .rename(columns={0: "corr_value"})
    )
    # Collect the list of features that are the most frequent in correlated pairs with
    # corr_value greater than threshold
    f_to_drop = []
    most_correlated_df = correlated_features.query("corr_value > @corr_thres").copy()
    # While there are some pairs in correlated features dataset
    while len(most_correlated_df.index):
        # Get the list of correlated features in that dataset
        most_corr = list(itertools.chain(*most_correlated_df.pairs))
        # Define the most frequent feature there, add it to the list of
        # features to drop and delete all the pairs from most_correlated_df where
        # there is current most frequent correlated feature
        f = np.unique(sorted(most_corr, key=lambda x: Counter(most_corr)[x]))[-1]
        f_to_drop += [f]
        most_correlated_df["f_in_pairs"] = most_correlated_df.pairs.apply(
            lambda x: f in x
        )
        most_correlated_df = most_correlated_df.query("not f_in_pairs")

    X.drop(f_to_drop, axis=1, inplace=True)
    X_to_predict.drop(f_to_drop, axis=1, inplace=True)
    # Save data
    X.to_csv(output_path_X, index=False)
    X_to_predict.to_csv(output_path_X_to_predict, index=False)
    y.to_csv(output_path_y, index=False)


if __name__ == "__main__":
    process_correlated_features()
