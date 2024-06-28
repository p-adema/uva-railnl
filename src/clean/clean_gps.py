import polars as pl

from .constants import data_dir


def clean_gps(filename: str, cleaned_file: str = None) -> None:
    """
    A cleaning function that takes a GPS data file and select the usable columns for
    later use.
    :param filename: The name of the gps file to be cleaned.
    :param cleaned_file: The new name of the file the cleaned data will be stored in.
    :return: None, but makes a new data file.
    """
    if cleaned_file is None:
        cleaned_file = "gps.pq"
    with open(data_dir(f"mtps/{filename}")) as file:
        # The sherlock format files are different from the GPS_filter files,
        # and must be parsed differently
        is_sherlock = file.read(8) == b"Sherlock"

    if is_sherlock:
        (
            pl.scan_csv(
                data_dir(f"mtps/{filename}"),
                has_header=True,
                separator=";",
                skip_rows=1,
                truncate_ragged_lines=True,
            )
            .select(
                train_nr=pl.col("Treinnr").cast(pl.UInt32),
                mat_nr=pl.col("Matnr").cast(pl.UInt32),
                time=pl.col("Tijdstip").str.to_datetime("%F %T%.3f"),
                lat=pl.col("GPS_latitude").str.replace(",", ".").cast(pl.Float64),
                lon=pl.col("GPS_longitude").str.replace(",", ".").cast(pl.Float64),
            )
            # Many coordinates appear invalid, we throw those away
            .filter(pl.col("lat").is_between(50, 60) & pl.col("lon").is_between(3, 7))
            .sink_parquet(data_dir(f"mtps/{cleaned_file}"))
        )
    else:
        (
            pl.scan_csv(data_dir(f"mtps/{filename}"), has_header=True, separator=";")
            .select(
                train_nr=pl.col("Treinnummer").cast(pl.UInt32),
                mat_nr=pl.col("Mat-nummer").cast(pl.UInt32),
                time=pl.col("Tijdstip").str.to_datetime("%F %T"),
                lat=pl.col("Latitude").str.replace(",", ".").cast(pl.Float64),
                lon=pl.col("Longitude").str.replace(",", ".").cast(pl.Float64),
            )
            .sink_parquet(data_dir(f"mtps/{cleaned_file}"))
        )


def ensure_sas(cleaned: str, *, original: str):
    """
    Used to makes sure that the cleaned version of the GPS data exists on the system,
    if it does not exist it is made.
    :param cleaned: The name of the cleaned file.
    :param original: The name of the raw GPS file.
    :return: None, but potentially makes a new cleaned GPS file.
    """
    import os

    if os.path.isfile(data_dir(f"mtps/{cleaned}")):
        return

    clean_gps(original, cleaned_file=cleaned)
