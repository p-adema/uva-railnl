import polars as pl

from .constants import LAT_TO_KM, LON_TO_KM, SENSOR_POSITIONS, data_dir, with_suffix
from .preprocess_rtm import ensure_rtm_preprocessed

# FYI: this lovely thing uses ~200GB of RAM if you run in on the full 3-month dataset.
# If you don't have a 256GB box, you can split the operation into blocks
# (as in 'link_rtm_mtps.py' and 'space_extra_gps.py')


# noinspection DuplicatedCode
def space_window(
    train_name: str = "train.pq",
    preprocessed_train_name: str = "train_preprocessed.pq",
    window_name: str = "space_window.pq",
    window_size_m: int = 5_000,
):
    """
    Create training samples using a window in space around each sensor
    :param train_name: The name of the original train information data file.
    :param preprocessed_train_name: The name of the preprocessed train information file.
    :param window_name: The name of the file to be produced in the function.
    :param window_size_m: The radius of the window in metres
    :return: None, but makes a new datafile.
    """
    ensure_rtm_preprocessed(preprocessed_train_name, original=train_name)

    # SENSOR_POSITIONS as a Pandas DataFrame
    sensor_df: pl.DataFrame = (
        pl.DataFrame(
            pl.Series(
                "pos",
                SENSOR_POSITIONS,
                dtype=pl.Array(pl.Float64, shape=2),
            )
        )
        .select_seq(s_lat=pl.col("pos").arr.get(0), s_lon=pl.col("pos").arr.get(1))
        .with_row_index("sensor", 1)
    )

    # Distance to the 'primary' row, except the window is now right-closed
    # see link_rtm_mtps.py
    window_time_distance: pl.Expr = (
        pl.col("time").last().sub(pl.col("time")).dt.total_seconds().abs().mul(70 / 3.6)
    )

    train_sensor_distance: pl.Expr = (
        (pl.col("s_lat").last().sub(pl.col("lat")).mul(LAT_TO_KM).pow(2))
        .add(pl.col("s_lon").last().sub(pl.col("lon")).mul(LON_TO_KM).pow(2))
        .sqrt()
        .round()
        .cast(pl.UInt32)
    )

    linked_rtm: pl.LazyFrame = pl.scan_parquet(data_dir(f"rtm/{train_name}"))

    (
        linked_rtm.join(
            pl.scan_parquet(data_dir(f"rtm/{preprocessed_train_name}")),
            on=linked_rtm.columns,
            how="left",
            coalesce=True,
        )
        .cast({"sensor": pl.UInt32})
        .join(sensor_df.lazy(), on="sensor", how="left", coalesce=True)
        .sort("time")
        .with_row_index()
        .with_columns(
            roll_time=pl.col("time")
            .dt.cast_time_unit("ns")
            .add(pl.duration(nanoseconds="index"))
        )
        .collect()
        .lazy()
        .rolling("roll_time", period="1m")
        .agg(
            pl.col("time", "lat", "lon", "sensor").last(),
            # For all preprocessed rows (ones with an associated sensor), gather all
            # other measurements in the rolling window and mark them with `keep` for
            # whether they are within the space window
            trains=pl.when(pl.col("sensor").last().is_not_null()).then(
                pl.struct(
                    pl.col(r"^volt_\d$"),
                    distance_to_sensor=train_sensor_distance,
                    keep=train_sensor_distance.add(window_time_distance)
                    .lt(window_size_m)
                    .and_(
                        pl.any_horizontal(pl.col("^volt_.$").is_last_distinct()),
                    ),
                )
                .reverse()
                .gather(pl.col("trip_id").reverse().arg_unique())
            ),
        )
        .filter(pl.col("sensor").is_not_null())
        # Throw away trains that are not `keep` (outside the space window)
        .with_columns(
            pl.col("trains")
            .list.eval(
                pl.when(pl.element().struct.field("keep")).then(
                    pl.struct(
                        pl.element().struct.field(
                            "volt_1", "volt_2", "volt_7", "distance_to_sensor"
                        )
                    )
                )
            )
            .list.drop_nulls()
        )
        .filter(pl.col("trains").list.len().gt(0))
        .drop("roll_time", "latitude", "longitude")
        .collect()
        .write_parquet(data_dir(f"rtm/{window_name}"))
    )


def ensure_space_window(
    cleaned: str,
    *,
    original_train: str,
    original_train_preprocessed: str = None,
    window_size_m: int = 5_000,
):
    """
    Makes sure the file containing the dataset with the expanded space window
    exists on the system. If it does not, it is made.
    :param cleaned: The name of the file containing the dataset with the expanded
    space window.
    :param original_train: The filename of the original
    :param original_train_preprocessed:
    :param window_size_m:
    :return:
    """
    import os

    if os.path.isfile(data_dir(f"rtm/{cleaned}")):
        return

    if original_train_preprocessed is None:
        original_train_preprocessed = with_suffix(original_train, "_preprocessed.pq")

    space_window(
        original_train,
        original_train_preprocessed,
        window_name=cleaned,
        window_size_m=window_size_m,
    )
