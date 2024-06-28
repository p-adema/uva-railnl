import polars as pl

from .constants import LAT_TO_KM, LON_TO_KM, data_dir, with_suffix


def preprocess_mtps(
    file: str,
    preprocessed_file: str = None,
    min_avg_speed: float = 20.0,
    min_count: int = 50,
    min_total_dur_min: int = 10,
    max_gap_min: int = 10,
    min_total_dist_km: float = 1.0,
    min_local_dist_m: int = 50,
    local_time_window: str = "5m",
    act_file: str = "act.pq",
):
    """

    :param file: Input GPS filename in data/mtps
    :param preprocessed_file: Output filename in data/mtps
    :param min_avg_speed: Average speed a trip needs to have to be kept
    :param min_count: Number of measurements a trip needs to have
    :param min_total_dur_min: Duration a trip needs to have
    :param max_gap_min: Maximum minutes between measurements in a trip
    :param min_total_dist_km: Minimum distance a trip needs to span
    :param min_local_dist_m: Minimum distance a trip needs to go every local_time_window
    :param local_time_window: Time window for checking if a train is stationary
    :param act_file: Unused, for merging the act (train type dataset) in the future
    """
    if preprocessed_file is None:
        preprocessed_file = with_suffix(file, "_preprocessed.pq")

    window_distance = (
        (pl.col("lat").first().sub(pl.col("lat").last()).mul(LAT_TO_KM).pow(2))
        .add(pl.col("lon").first().sub(pl.col("lon").last()).mul(LON_TO_KM).pow(2))
        .sqrt()
        .gt(min_local_dist_m / 1_000)
        .fill_null(False)
    )

    # We haven't merged the GPS and ACT data, so the current MTPS data does not
    # include train_type. However, using rail_day, train_nr and mat_nr it's possible
    # to join these (we didn't have the time, and it didn't seem useful).
    #       rail_day = (
    #           pl.when(pl.col("time").dt.hour().gt(2))
    #           .then(pl.col("time").dt.date())
    #           .otherwise(pl.col("time").dt.date().sub(pl.duration(days=1)))
    #       )

    with pl.StringCache():
        # Most of this code is identifying which measurements belong to the same train
        (
            pl.scan_parquet(data_dir(f"mtps/{file}"))
            .drop("null")
            .filter(pl.col("train_nr").ne(0))
            .sort("train_nr", "mat_nr", "time")
            # We assume that successive measurements within 20m with the same
            # train_nr and mat_nr belong to the same train
            .with_columns(
                trip_id=(
                    pl.struct("train_nr", "mat_nr").rle_id().diff().cast(pl.Boolean)
                )
                .or_(pl.col("time").diff().gt(pl.duration(minutes=max_gap_min)))
                .fill_null(True)
                .cum_sum(),
            )
            .rolling("time", period=local_time_window, group_by="trip_id")
            .agg(
                pl.exclude("time", "trip_id").last(),
                moved_bck=window_distance,
            )
            .rolling(
                "time",
                period=local_time_window,
                offset="0",
                group_by="trip_id",
                closed="left",
            )
            .agg(
                pl.exclude("time", "trip_id").first(),
                moved_fwd=window_distance,
            )
            # unless they haven't moved in the previous or upcoming local_time_window
            .select(
                (pl.col("trip_id").rle_id().diff().cast(pl.Boolean))
                .or_(
                    pl.col("moved_bck").or_(pl.col("moved_fwd")).not_().fill_null(True)
                )
                .cum_sum(),
                pl.exclude("trip_id"),
            )
            .drop("moved_fwd", "moved_bck")
            # we only keep trips with more than min_count measurements, which last at
            # least min_total_dur_min, go at least min_total_dist_km with min_avg_speed
            .filter(pl.col("time").count().over("trip_id").gt(min_count))
            .with_columns(
                trip_dur=(pl.col("time").max().over("trip_id")).sub(
                    pl.col("time").min().over("trip_id")
                ),
                trip_dist=(pl.col("lat").diff().mul(LAT_TO_KM).pow(2))
                .add(pl.col("lon").diff().mul(LON_TO_KM).pow(2))
                .sqrt()
                .sum()
                .over("trip_id"),
            )
            .filter(
                pl.col("trip_dur").dt.total_minutes().gt(min_total_dur_min)
                & pl.col("trip_dist").gt(min_total_dist_km)
                & pl.col("trip_dist")
                .truediv(pl.col("trip_dur").dt.total_seconds())
                .mul(60 * 60)
                .gt(min_avg_speed)
            )
            .with_columns(pl.col("trip_id").rle_id())
            .with_columns(trip_step=pl.col("time").rle_id().over("trip_id"))
            .drop("trip_dur", "trip_dist", "speed")
            .sort("trip_id", "time")
            .collect()
            .write_parquet(data_dir(f"mtps/{preprocessed_file}"), compression_level=10)
        )


def ensure_mtps_preprocessed(cleaned: str, *, original: str) -> None:
    """
    Makes sure the preprocessed cleaned mtps data file exists on the system. If it does
    not, it is made.
    :param cleaned: The name of the cleaned mtps file.
    :param original: The name of the original non-processed mtps file.
    :return: None, but makes a new file if needed.
    """
    import os

    if os.path.isfile(data_dir(f"mtps/{cleaned}")):
        return

    preprocess_mtps(original, preprocessed_file=cleaned)


if __name__ == "__main__":
    preprocess_mtps("gps_all.pq")
