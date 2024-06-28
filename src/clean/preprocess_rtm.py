import polars as pl

from .constants import LAT_TO_KM, LON_TO_KM, SENSOR_POSITIONS, data_dir, with_suffix


def is_min(num: int) -> pl.Expr:
    return (
        pl.when(pl.col(f"sensor{num}_distance").eq(pl.col("min")))
        .then(num)
        .alias(f"sensor{num}_distance")
    )


def preprocess_rtm(
    filename: str, cleaned_file: str = None, window_dist: int = 10_000
) -> None:
    """
    Takes the filename of an RTM dataset check which sensor is the closest

    :param filename: A string containing the filename of the
            RTM dataset without the extension.
    :param cleaned_file: Destination filename for cleaned data
    :param window_dist: Maximum distance a measurement can have to the closest sensor
    :return: None, but makes a new parquet file.
    """

    if cleaned_file is None:
        cleaned_file = with_suffix(filename, "_preprocessed.pq")
    (
        pl.scan_parquet(data_dir(f"rtm/{filename}"))
        .with_columns(
            [
                (
                    (pl.col("lat").sub(s_lat).mul(LAT_TO_KM).pow(2))
                    .add(pl.col("lon").sub(s_lon).mul(LON_TO_KM).pow(2))
                    .sqrt()
                    .alias(f"sensor{i}_distance")
                )
                for i, (s_lat, s_lon) in enumerate(SENSOR_POSITIONS, start=1)
            ]
        )
        # .drop("lat", "lon")
        .with_columns(min=pl.min_horizontal("^sensor.*$"))
        .with_columns([is_min(i) for i in range(1, 13)])
        .with_columns(sensor=pl.min_horizontal("^sensor.*$"))
        .drop([f"sensor{i}_distance" for i in range(1, 13)])
        .filter(pl.col("min") <= window_dist)
        .rename({"min": "distance_to_sensor"})
        .sort("time")
        .sink_parquet(data_dir(f"rtm/{cleaned_file}"), compression_level=10)
    )


def ensure_rtm_preprocessed(cleaned: str, *, original: str):
    import os

    from .clean_rtm import ensure_rtm

    if os.path.isfile(data_dir(f"rtm/{cleaned}")):
        return

    ensure_rtm(original, original="rtm.pq.nosync")
    preprocess_rtm(original, cleaned_file=cleaned)


if __name__ == "__main__":
    preprocess_rtm("cleaned.pq")
