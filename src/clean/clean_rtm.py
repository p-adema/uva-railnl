import polars as pl

from .constants import data_dir, with_suffix

measurement_names = {
    "lijnspanning 10 4 v bit 3a2 mbvk1": "volt_1",
    "lijnspanning 10 4 v bit 3a2 mbvk2": "volt_2",
    "lijnspanning 10 4 v bit 3a2 mbv7": "volt_7",
    "positie noorderbreedte graden abv6": "n_grad",
    "positie noorderbreedte minuten abv6": "n_min",
    "positie noorderbreedte seconden abv6": "n_sec",
    "positie noorderbreedte 0 01 seconden abv6": "n_sub_sec",
    "positie oosterlengte graden abv6": "e_grad",
    "positie oosterlengte minuten abv6": "e_min",
    "positie oosterlengte seconden abv6": "e_sec",
    "positie oosterlengte 0 01 seconden abv6": "e_sub_sec",
}


def list_find(name: str, alias: str) -> pl.Expr:
    """
    This function looks for a specific measurement key within the structs of the dataset
    and extracts the corresponding values.
    :param name: The measurement key.
    :param alias: The new name of the column to be added.
    :return: A Polars Expression which extracts the needed values from the dataset.
    """
    return (
        pl.col("measurements_filtered_normalized")
        .list.eval(
            pl.when(
                pl.element().struct.field("key").eq(name)
                & pl.element().struct.field("value").struct.field("valid")
            ).then(
                pl.element()
                .struct.field("value")
                .struct.field("value")
                .round()
                .cast(pl.Int16),
            )
        )
        .list.drop_nulls()
        .list.first()
    ).alias(alias)


def coord_calc(prefix: str) -> pl.Expr:
    """
    Creates a new column which turns the DMS coordinates to DD coordinates
    :param prefix: Either 'lat' or 'lon', to indicate which coordinates are converted
    :return: A Polars Expression for the DD coordinates
    """
    return (
        pl.col(f"{prefix}_grad")
        .add(pl.col(f"{prefix}_min").truediv(60))
        .add(pl.col(f"{prefix}_sec").truediv(60 * 60))
        .add(pl.col(f"{prefix}_sub_sec").truediv(60 * 60 * 100))
    )


def voltage_calc(num: int) -> pl.Expr:
    """
    When any of the volt values are nonexistent they are made by taking the mean of the
    other two voltages. Additionally, outlier voltages are filtered.
    :param num: The voltage number to be composed.
    :return: A Polars Expression that adds this new value to the dataset.
    """
    name = f"volt_{num}"
    other = [
        "volt_1",
        "volt_2",
        "volt_7",
    ]
    other.remove(name)
    val = pl.col(name).fill_null(pl.mean_horizontal(*other))
    return pl.when(val.lt(100)).then(0).when(val.is_between(1_000, 2_200)).then(val)


def clean_rtm(dir_or_file: str, cleaned_file=None, is_dir=True) -> None:
    """
    Cleans the raw RTM data for later linking and use. Can be run on a directory, in
    which case this directory should contain several parquet files.
    :param dir_or_file: The name or path to the raw data.
    :param cleaned_file: The name of the new file to be made.
    :param is_dir: Whether the given dir_or_file is a directory or file name.
    :return: None, but makes a new file.
    """
    if cleaned_file is None:
        cleaned_file = with_suffix(f"rtm/{dir_or_file}", "_cleaned.pq")
    if is_dir:
        dir_or_file = f"{dir_or_file}/*.parquet"
    (
        pl.scan_parquet(data_dir(dir_or_file))
        .lazy()
        .select(
            *[list_find(m, n) for m, n in measurement_names.items()],
            time=pl.col("datetime")
            .struct.field("local")
            .str.to_datetime("%+")
            .dt.convert_time_zone("Europe/Amsterdam"),
        )
        .select(
            "time",
            lat=coord_calc("n"),
            lon=coord_calc("e"),
            volt_1=voltage_calc(1),
            volt_2=voltage_calc(2),
            volt_7=voltage_calc(7),
        )
        .filter(pl.col("lat").ne(0) & pl.col("lon").ne(0))
        .drop_nulls()
        .sink_parquet(data_dir(f"rtm/{cleaned_file}"))
    )


def ensure_rtm(cleaned: str, *, original: str) -> None:
    """
    Makes sure that a cleaned version of the rtm data exists on the system.
    If this file does not exist it will be made from the raw RTM data.
    :param cleaned: Supposed name of the cleaned data file.
    :param original: Name of the raw RTM data file.
    :return: None, but potentially makes a new file.
    """
    import os

    if os.path.isfile(data_dir(f"rtm/{cleaned}")):
        return

    print(f"No cleaned rtm file ( can't find '{data_dir(f'rtm/{cleaned}')}'")
    print("Trying to clean RTM. This might take a while")
    try:
        clean_rtm(original, cleaned_file=cleaned, is_dir=False)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Missing raw RTM file rtm/{original}") from e
