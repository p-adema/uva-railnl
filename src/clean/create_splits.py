import numpy as np
import polars as pl

from .constants import data_dir, with_suffix

SHUFFLE_SEED = 42


def split_data(
    file_name: str,
    input_columns: list[pl.type_aliases.IntoExprColumn] = None,
    target_column: str = "sensor_voltage",
    split_file: str = None,
) -> None:
    # noinspection PyShadowingNames
    """
    Splits the data file into train, tune, and test sets.
    :param split_file: Output file for splits
    :param file_name: The name of the file to be split
    :param input_columns: The columns to be kept as inputs
    :param target_column: The name of the column containing the target data
    :return: None, but makes three separate numpy arrays.

    Later usage:

    >>> splits = np.load("simple_joined_splits.npz")  # doctest: +SKIP
    >>> i_train, i_tune, i_test, t_train, t_tune, t_test = [  # doctest: +SKIP
    ...     splits[part]
    ...     for part in ["i_train", "i_tune", "i_test", "t_train", "t_tune", "t_test"]
    ... ]

    """
    if input_columns is None:
        input_columns = ["volt_1", "volt_2", "volt_7", "distance_to_sensor"]
    if split_file is None:
        split_file = with_suffix(file_name, "_splits.npz")

    df: pl.DataFrame = (
        pl.read_parquet(data_dir(f"samples/{file_name}"))
        .sample(fraction=1, shuffle=True, seed=SHUFFLE_SEED)
        .select(input_columns, target=target_column)
    )

    arr_target = df.drop_in_place("target").to_numpy()
    arr_input = df.to_numpy()

    splits = [int(len(df) * 0.6), int(len(df) * 0.9)]

    i_train, i_tune, i_test = np.split(arr_input, splits)
    t_train, t_tune, t_test = np.split(arr_target, splits)

    np.savez_compressed(
        data_dir(f"samples/{split_file}"),
        i_train=i_train,
        i_tune=i_tune,
        i_test=i_test,
        t_train=t_train,
        t_tune=t_tune,
        t_test=t_test,
    )
