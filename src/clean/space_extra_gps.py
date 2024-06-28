import math

import polars as pl
import tqdm

from .constants import LAT_TO_KM, LON_TO_KM, SENSOR_POSITIONS, data_dir, with_suffix
from .preprocess_rtm import ensure_rtm_preprocessed

# This thing is an extension of `space_window.py` to make it also merge in the MTPS/GPS
# data a second time, so that it can count how many trains there are in the area while
# performing the space window (including those without an RTM measurement)
# It's pretty slow though, and didn't seem to help the models much.


# noinspection DuplicatedCode
def space_window(
    train_name: str = "train.pq",
    preprocessed_train_name: str = "train_preprocessed.pq",
    window_name: str = "space_window.pq",
    window_size_m: int = 5_000,
    block_size: int = 10_000,
    mtps_name: str = "gps_preprocessed.pq",
):
    ensure_rtm_preprocessed(preprocessed_train_name, original=train_name)

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

    roll_df = (
        pl.concat(
            (
                linked_rtm.join(
                    pl.scan_parquet(data_dir(f"rtm/{preprocessed_train_name}")),
                    on=linked_rtm.columns,
                    how="left",
                    coalesce=True,
                ),
                pl.scan_parquet(data_dir(f"mtps/{mtps_name}")),
            ),
            how="diagonal",
        )
        .cast({"sensor": pl.UInt32})
        .join(sensor_df.lazy(), on="sensor", how="left", coalesce=True)
        .sort("time")
        .with_row_index()
        .with_columns(
            roll_time=pl.col("time")
            .dt.cast_time_unit("ns")
            .add(pl.duration(nanoseconds="index")),
            rtm_trip=pl.when(pl.col("volt_1").is_not_null()).then("trip_id"),
        )
        .collect()
    )
    res = [pl.DataFrame()]
    for block in tqdm.trange(math.ceil(len(roll_df) / block_size), desc="Space window"):
        if (
            roll_df.lazy()
            .slice(block * block_size, block_size)
            .select(pl.col("sensor").null_count())
            .collect()["sensor"][0]
            == block_size
        ):
            continue

        res.append(
            roll_df.lazy()
            .slice(max(block * block_size - 10_000, 0), block_size)
            .rolling("roll_time", period="1m")
            .agg(
                pl.col("time", "lat", "lon", "sensor", "index").last(),
                trains=pl.when(pl.col("sensor").last().is_not_null()).then(
                    pl.struct(
                        pl.col(r"^volt_\d$"),
                        distance_to_sensor=train_sensor_distance,
                        keep=train_sensor_distance.add(window_time_distance)
                        .lt(window_size_m)
                        .and_(
                            pl.any_horizontal(pl.col("^volt_.$").is_last_distinct()),
                            pl.col("volt_1").is_not_null(),
                        ),
                    )
                    .reverse()
                    .gather(pl.col("rtm_trip").reverse().arg_unique())
                ),
                train_count=pl.col("trip_id")
                .filter(
                    train_sensor_distance.add(window_time_distance).lt(window_size_m)
                )
                .n_unique(),
            )
            .filter(pl.col("sensor").is_not_null())
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
        )

    (
        pl.concat(res)
        .lazy()
        .unique("index")
        .drop("index")
        .sort("time")
        .sink_parquet(data_dir(f"rtm/{window_name}"))
    )


def ensure_space_window(
    cleaned: str,
    *,
    original_train: str,
    original_train_preprocessed: str = None,
    window_size_m: int = 5_000,
):
    import os

    if os.path.isfile(data_dir(f"sas/{cleaned}")):
        return

    if original_train_preprocessed is None:
        original_train_preprocessed = with_suffix(original_train, "_preprocessed.pq")

    space_window(
        original_train,
        original_train_preprocessed,
        window_name=cleaned,
        window_size_m=window_size_m,
    )
