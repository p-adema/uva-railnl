import gc
import math

import polars as pl
from tqdm import trange

from .constants import data_dir, with_suffix


def link_rtm_mtps(
    rtm_file: str,
    mtps_file: str,
    linked_file: str = None,
    block_size: int = 10_000,
    max_blocks: int = None,
):
    """
    Links the rtm and mtps data in order to get all the useful train based data in
    one joined file.
    :param rtm_file:
    :param mtps_file:
    :param linked_file:
    :param block_size:
    :param max_blocks:
    :return:
    """
    if linked_file is None:
        linked_file = with_suffix(rtm_file, "_train.pq")

    # Number of blocks we need to process
    rtm_blocks = math.ceil(
        (
            pl.scan_parquet(data_dir(f"rtm/{rtm_file}"))
            .select(pl.col("time").len())
            .collect()
            / block_size
        )["time"][0]
    )

    # We will later be rolling over the dataframe, comparing all rows in a time window
    # with the 'primary' row (namely .first() ) to see if they're close

    # The distance between this row and the 'primary' row in meters
    coord_distance: pl.Expr = (
        (pl.col("lat").first().sub(pl.col("lat").slice(1)).mul(111 * 1000).pow(2)).add(
            (pl.col("lon").first().sub(pl.col("lon").slice(1)).mul(68 * 1000)).pow(2)
        )
    ).sqrt()

    # The equivalent distance in meters that a difference in time would have,
    # if the two trains would be moving away from each other at 70 km/h
    time_distance: pl.Expr = (
        pl.col("real_time")
        .first()
        .sub(pl.col("time").slice(1))
        .dt.total_seconds()
        .abs()
        .mul(70 / 3.6)
    )

    # We add the two distances, as well as a 'marker' column which is only valid
    # for MTPS measurements. As such, total_dist is 'null' for RTM rows
    total_dist = coord_distance.add(time_distance).add(pl.col("marker").slice(1))

    block_results = []

    for block in trange(max_blocks or rtm_blocks):
        # Get the current RTM block to process
        # We offset the primary time index by -30 seconds, so that we can later
        # roll with a window of 1 minute and get 30 seconds before and after
        # every RTM measurement
        df_rtm = (
            pl.scan_parquet(data_dir(f"rtm/{rtm_file}"))
            .slice(block * block_size, block_size)
            .with_columns(offset_time=pl.col("time").dt.offset_by("-30s"))
            .rename(
                {
                    "offset_time": "time",
                    "time": "real_time",
                }
            )
            .collect()
        )
        # Get the MTPS data around this block
        df_time_window = (
            pl.scan_parquet(data_dir(f"mtps/{mtps_file}"))
            .filter(
                pl.col("time").is_between(
                    df_rtm.select(pl.col("time").min()),
                    df_rtm.select(pl.col("time").max().dt.offset_by("1m")),
                )
            )
            .sort("time")
            .head(20 * block_size)  # Don't die if the time window is miscalculated
            .collect()
        )
        # Combine the two dataframes. This could be done better, but it works
        df_outer = (
            df_time_window.lazy()
            .join(df_rtm.lazy(), on=["time", "lat", "lon"], how="full")
            .select(
                pl.when(pl.col("time").is_null())
                .then(pl.col("time_right"))
                .otherwise(pl.col("time"))
                .alias("time"),
                pl.coalesce("lat", "lat_right"),
                pl.coalesce("lon", "lon_right"),
                pl.col(
                    "train_nr",
                    "mat_nr",
                    "^volt_.$",
                    "real_time",
                    "trip_id",
                ),
                marker=pl.when(pl.col("volt_1").is_null()).then(0),
            )
            .sort("time")
            .with_row_index("id")
            .with_columns(
                # make sure every entry in time is unique
                pl.col("time").dt.cast_time_unit("ns")
                + pl.duration(nanoseconds=pl.col("id"))
            )
            .collect()
        )
        del df_rtm, df_time_window

        df_joined: pl.DataFrame = (
            df_outer.lazy()
            .rolling(
                check_sorted=False,
                index_column="time",
                period="60s",
                offset="0s",
                closed="left",
            )
            # Since we have a positive period and are left closed, .first() is always
            # the window origin (our 'primary' row), and we therefore compare all
            # other rows with this .first()
            .agg(
                pl.col("id").first(),
                pl.col("lat").first().alias("real_lat"),
                pl.col("lon").first().alias("real_lon"),
                pl.col("real_time").first().alias("real_time"),
                pl.col("volt_1").first().is_not_null().alias("is_measurement"),
                pl.col("^volt_.$").first(),
                # When the window origin is an RTM row (has a not-null voltage), and
                # also has a MTPS measurement which is relatively close, then we use
                # that MTPS measurement to set train_nr, mat_nr and trip_id
                pl.when(
                    pl.col("volt_1").first().is_not_null() & total_dist.min().lt(1000)
                ).then(
                    pl.col("train_nr", "mat_nr", "trip_id").get(
                        total_dist.arg_min().add(1).fill_null(0)
                    ),
                ),
            )
            # We then only take RTM measurements that were linked to an MTPS row
            .filter(
                pl.col("is_measurement").and_(
                    pl.all_horizontal(pl.col("train_nr", "mat_nr").is_not_null())
                )
            )
            .select(
                pl.col("time")
                .sub(pl.duration(nanoseconds=pl.col("id")))
                .dt.cast_time_unit("us"),
                pl.col("^volt_.$").cast(pl.UInt32),
                pl.col("^.*_nr$", "trip_id"),
                lat="real_lat",
                lon="real_lon",
            )
            .collect()
        )
        del df_outer

        block_results.append(df_joined)
        del df_joined

        if len(block_results) >= 20:
            block_results = [pl.concat(block_results, rechunk=True)]

        gc.collect()

    # Finally, we group all the blocks together, and recalculate the trip_step
    # (as we might have lost some MTPS measurements that were too far from their
    #  corresponding RTM measurement)
    pl.concat(block_results).with_columns(
        trip_step=pl.col("time").rle_id().over("trip_id")
    ).write_parquet(data_dir(f"rtm/{linked_file}"))


def ensure_linked(
    cleaned: str, *, original_rtm: str, original_mtps: str, block_size: int = 10_000
):
    """
    Makes sure the linked sas and mtps file exists on the system, if it does not it is
    made.
    :param cleaned: The name of the cleaned file, to be given if the file does not
    exist.
    :param original_rtm: The name of the original rtm file.
    :param original_mtps: The name of the original mtps file.
    :param block_size: The block size passed onto the link_rtm_mtps function.
    :return: None, but makes a new file if needed.
    """
    import os

    if os.path.isfile(data_dir(f"rtm/{cleaned}")):
        return

    link_rtm_mtps(original_rtm, original_mtps, cleaned, block_size)
