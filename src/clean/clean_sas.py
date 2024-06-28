import polars as pl

from .constants import SENSOR_POSITIONS, data_dir


def add_sensor_num(sensor: tuple, num: int) -> pl.Expr:
    """
    Based on the sensor coordinates in the SAS data set assigns the correct sensor
    number to each row in the data.
    :param sensor: The coordinates of the sensor currently being checked for.
    :param num: The number of the sensor being checked for.
    :return: A Polars Expression that adds a new column that contains either a
    number or None.
    """
    return (
        pl.when((pl.col("latitude") == sensor[0]) & (pl.col("longitude") == sensor[1]))
        .then(num)
        .alias(f"s_dist_{num}")
    )


def clean_sas(filename: str, cleaned_file: str = None) -> None:
    """
    Selects only the necessary columns from the SAS data for model training.
    :param filename: The name of the raw SAS file.
    :param cleaned_file: The name of the new cleaned SAS file.
    :return: None, but makes a new file on the system.
    """
    if cleaned_file is None:
        cleaned_file = "avg_cleaned.pq"
    (
        pl.scan_parquet(data_dir(f"sas/{filename}"))
        .select(
            "latitude",
            "longitude",
            sensor_voltage="max",
            time=pl.from_epoch(pl.col("t_max"), time_unit="s")
            .cast(pl.Datetime)
            .dt.replace_time_zone("Europe/Amsterdam", non_existent="null"),
        )
        .with_columns(
            [
                add_sensor_num(sensor, num)
                for num, sensor in enumerate(SENSOR_POSITIONS, 1)
            ]
        )
        .with_columns(sensor=pl.min_horizontal(r"^s_dist_.\d$"))
        .drop([f"s_dist_{num+1}" for num in range(len(SENSOR_POSITIONS))])
        .drop_nulls()
        .sort("time")
        .sink_parquet(data_dir(f"sas/{cleaned_file}"), compression_level=10)
    )


def ensure_sas(cleaned: str, *, original: str) -> None:
    """
    Makes sure the cleaned version of the SAS data exists on the system.
    :param cleaned: The name of the cleaned version of the file.
    :param original: The name of the raw SAS data file.
    :return: None, but potentially makes a new file.
    """
    import os

    if os.path.isfile(data_dir(f"sas/{cleaned}")):
        return

    clean_sas(original, cleaned_file=cleaned)
