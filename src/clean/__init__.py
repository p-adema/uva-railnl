import os
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from .constants import data_dir
from .create_splits import split_data
from .link_rtm_sas import ensure_linked
from .space_pad import ensure_space_padded
from .space_window import ensure_space_window
from .svd_kernels import ensure_kernels
from .time_window import ensure_time_window

if TYPE_CHECKING:
    from numpy.lib.npyio import NpzFile


def get_base_splits(name: str = "simple_splits.npz") -> "NpzFile":
    """
    Makes sure that the system has the 'simple_splits' numpy file for the
    training tuning and testing of the models. If it does not yet exist
    it is made.
    :param name: The name of the split data file.
    :returns: A NpZ file-mapped dictionary containing the 6 splits
    """
    if not os.path.isfile(data_dir(f"samples/{name}")):
        ensure_linked(
            "train_joined.pq",
            original_rtm="train_preprocessed.pq",
            original_sas="avg_cleaned.pq",
        )
        split_data(
            "train_joined.pq",
            split_file=name,
            input_columns=["volt_1", "volt_2", "volt_7", "distance_to_sensor"],
            target_column="sensor_voltage",
        )
    return np.load(data_dir(f"samples/{name}"))


def get_time_splits(
    name: str = "train_splits.npz", include_interpolated: bool = True
) -> "NpzFile":
    """
    Makes sure that the splits file for the extra time dimension data set exists for
    training, tuning, and testing of the neural networks using this dataset. If the
    file does not exist it is made.
    :param name: The name of the splits file.
    :param include_interpolated: A parameter stating whether datapoints that are
    centred on an interpolated point should be included
    :returns: A NpZ file-mapped dictionary containing the 6 splits
    """
    if name == "train_splits.npz" and not include_interpolated:
        name = "train_ni_splits.npz"
    if not os.path.isfile(data_dir(f"samples/{name}")):
        ensure_linked(
            "train_joined.pq",
            original_rtm="train_cleaned.pq",
            original_sas="avg_cleaned.pq",
        )
        ensure_time_window("train_joined.pq", include_interpolated=include_interpolated)
        split_data(
            "interpolated_train_joined.pq",
            split_file=name,
            input_columns=[
                "volt_1",
                "volt_2",
                "volt_7",
                "distance_to_sensor",
                "pre_volt_1",
                "pre_volt_2",
                "pre_volt_7",
                "pre_distance_to_sensor",
                "pos_volt_1",
                "pos_volt_2",
                "pos_volt_7",
                "pos_distance_to_sensor",
            ],
            target_column="sensor_voltage",
        )
    return np.load(data_dir(f"samples/{name}"))


def get_space_splits(name: str = "space_splits.npz") -> "NpzFile":
    """
    Makes sure that the splits file for the extra space dimension data set exists for
    training, tuning, and testing of the neural networks using this dataset. If the
    file does not exist it is made.
    :param name: The name of the splits file.
    :returns: A NpZ file-mapped dictionary containing the 6 splits
    """
    if not os.path.isfile(data_dir(f"samples/{name}")):
        print("space split file missing")
        print("checking space window")
        ensure_space_window(
            "space_window.pq",
            original_train="train.pq",
            original_train_preprocessed="train_preprocessed.pq",
        )
        print("checking linked")
        ensure_linked(
            "space_joined.pq",
            original_rtm="space_window.pq",
            original_sas="avg_cleaned.pq",
        )
        print("checking padding")
        ensure_space_padded("space_padded.pq", original="space_joined.pq")
        print("splitting")
        split_data(
            "space_padded.pq",
            split_file=name,
            input_columns=[pl.exclude("sensor_voltage")],
            target_column="sensor_voltage",
        )

    return np.load(data_dir(f"samples/{name}"))


def get_kernel_splits(
    name: str = "kernel_splits.npz", original: str = "train_joined.pq"
):
    """
    Makes sure that the data splits exist with all the available kernels for
    training, tuning, and testing for the support vector machine. If it does not
    yet exist it is made.
    :param name: The name of the split data file.
    :param original: The name of the original data file to process
    :returns: A NpZ file-mapped dictionary containing the 6 splits
    """
    if not os.path.isfile(data_dir(f"samples/{name}")):
        print("no kernel file")
        print("linking with sas")
        ensure_linked(
            original,
            original_rtm="train_preprocessed.pq",
            original_sas="avg_cleaned.pq",
        )
        print("making kernels")
        ensure_kernels("kernels.pq", original=original)
        print("splitting")
        split_data(
            "kernels.pq",
            input_columns=[pl.exclude("sensor_voltage")],
            target_column="sensor_voltage",
            split_file=name,
        )
    return np.load(data_dir(f"samples/{name}"))


__all__ = [
    data_dir,
    get_base_splits,
    get_space_splits,
    get_time_splits,
    get_kernel_splits,
]
