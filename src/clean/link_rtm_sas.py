import polars as pl

from .constants import data_dir


def link_rtm_sas(
    rtm_name: str, sas_name: str, linked_file: str = "simple_joined.pq"
) -> None:
    """
    Joins the cleaned rtm and sas data based on sensor and approximate time.
    :param linked_file: Output file for result.
    :param rtm_name: The name of the rtm file.
    :param sas_name: The name of the sas file.
    :return: None, but makes a new parquet file.
    """
    # It might be best to lower the tolerance if there's more data available
    (
        pl.scan_parquet(data_dir(f"rtm/{rtm_name}"))
        .sort("time")
        .cast({"sensor": pl.UInt8})
        .join_asof(
            pl.scan_parquet(data_dir(f"sas/{sas_name}"))
            .sort("time")
            .cast({"sensor": pl.UInt8}),
            by="sensor",
            on="time",
            strategy="nearest",
            tolerance="20m",
        )
        .drop_nulls()
        .collect()
        .write_parquet(data_dir(f"samples/{linked_file}"), compression_level=10)
    )


def ensure_linked(linked: str, *, original_rtm: str, original_sas: str) -> None:
    """
    Makes sure the linked rtm and sas file exists on the system. If it does not
    it is made.
    :param linked: The name of the linked sas and rtm file.
    :param original_rtm: The name of the original rtm file.
    :param original_sas: The name of the original sas file.
    :return: None, but potentially makes a new file.
    """
    import os

    from .clean_sas import ensure_sas
    from .preprocess_rtm import ensure_rtm_preprocessed

    if os.path.isfile(data_dir(f"samples/{linked}")):
        return

    ensure_rtm_preprocessed(original_rtm, original="train.pq")
    ensure_sas(original_sas, original="voltage-avg-feb-april.pq")
    print(f"linking {original_rtm=} and {original_sas=}, into {linked}")
    link_rtm_sas(original_rtm, original_sas, linked_file=linked)
