import polars as pl

from .constants import data_dir

# For the models (even the LSTM) it's good practice to pad to a fixed length.
# Therefore, we pad all the samples to length 10


def space_window_pad(
    joined_name: str = "space_window.pq",
    out_name: str = "space_padded.pq",
    target_column: str = "sensor_voltage",
    pad_size: int = 10,
) -> None:
    """
    Takes the extended space window file
    :param joined_name: The name of the file to be padded.
    :param out_name: The name of the file that is made in the function
    :param target_column:
    :param pad_size:
    :return: None, but makes a new file.
    """
    pl.scan_parquet(data_dir(f"samples/{joined_name}")).select(
        target_column,
        pl.col("trains").list.len().cast(pl.UInt8).alias("length"),
        *[
            pl.col("trains")
            .list.get(i)
            .struct.field("*")
            .name.prefix(f"train_{i + 1:0>2}_")
            for i in range(pad_size)
        ],
    ).sink_parquet(data_dir(f"samples/{out_name}"))


def ensure_space_padded(cleaned: str, *, original: str):
    """
    Makes sure the space padded data file exists on the system. If it does not,
    it is made.
    :param cleaned: The filename of the presumed padded file.
    :param original: The name of the file the padded file is supposed to be made
    from.
    :return: None, but potentially makes a new file.
    """
    import os

    if os.path.isfile(data_dir(f"samples/{cleaned}")):
        return

    space_window_pad(original, out_name=cleaned)
