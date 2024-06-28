from datetime import timedelta

import polars as pl
import tqdm

from .constants import data_dir


def interpolate_per_trip(
    df: pl.DataFrame, include_interpolated: bool = False
) -> pl.DataFrame:
    """
    Interpolates measurements between captured times for all trips present in the
    dataset. By interpolating for every second a clear time window is created
    for each trip with which more thorough models can be trained.
    :param df: The dataset which is to be interpolated on.
    :param include_interpolated: Whether the full interpolated dataset should be
    returned, or just the rows on the trip step.
    :return: The interpolated polars dataframe.
    """
    interpolated_dfs = []

    # This should be done using .over() and explicit block chunking instead of manually
    # grouping by trip_id, if this is going to be run often.
    for trip_id, group in tqdm.tqdm(df.group_by(["trip_id"])):
        group = group.sort("time")
        new_df: pl.DataFrame = (
            pl.DataFrame(
                {
                    "time": pl.datetime_range(
                        group.select(pl.min("time")),
                        group.select(pl.max("time")),
                        timedelta(seconds=1),
                        eager=True,
                    )
                }
            )
            .with_columns(pl.lit(trip_id[0]).cast(pl.UInt32).alias("trip_id"))
            .join(group, on=["trip_id", "time"], how="left", coalesce=True)
            .with_columns(
                pl.col("^volt_.$").interpolate(),
                pl.col("train_nr").interpolate(),
                pl.col("mat_nr").interpolate(),
                pl.col("distance_to_sensor").interpolate(),
                pl.col("sensor").interpolate(),
                pl.col("latitude").interpolate(),
                pl.col("longitude").interpolate(),
                pl.col("sensor_voltage").interpolate(),
            )
        )
        second_df: pl.DataFrame = (
            new_df.lazy()
            .rolling(
                index_column="time",
                period="8s",
                offset="-4s",
                closed="none",
            )
            .agg(
                pl.col("time").first().alias("pre_time"),
                pl.col("time").last().alias("pos_time"),
                pl.col("volt_1").first().alias("pre_volt_1"),
                pl.col("volt_1").last().alias("pos_volt_1"),
                pl.col("volt_2").first().alias("pre_volt_2"),
                pl.col("volt_2").last().alias("pos_volt_2"),
                pl.col("volt_7").first().alias("pre_volt_7"),
                pl.col("volt_7").last().alias("pos_volt_7"),
                pl.col("distance_to_sensor").first().alias("pre_distance_to_sensor"),
                pl.col("distance_to_sensor").last().alias("pos_distance_to_sensor"),
            )
            .drop("time")
            .collect()
        )
        new_df = new_df.hstack(second_df)
        new_df = new_df[3:-2]
        interpolated_dfs.append(new_df)

    if include_interpolated:
        return pl.concat(interpolated_dfs)
    else:
        return (
            pl.concat(interpolated_dfs)
            .lazy()
            .sort("trip_id")
            .filter(pl.col("trip_step").is_not_null())
            .collect()
        )


def ensure_time_window(name: str, include_interpolated: bool = False):
    """
    Makes sure the time expanded data file exists on the system. If it does not
    it is made.
    :param name: The name of the file of the time expanded dataset.
    :param include_interpolated: A boolean indicating whether the interpolated
    data should be included in the dataset.
    :return: None, but potentially makes a new file.
    """
    import os

    filename = f"time_{name}" if include_interpolated else f"time_ni_{name}"
    if os.path.isfile(data_dir(f"samples/{filename}")):
        return

    (
        pl.read_parquet(f"../data/samples/{name}")
        .pipe(interpolate_per_trip, include_interpolated)
        .write_parquet(data_dir(f"samples/{filename}"), compression_level=10)
    )
