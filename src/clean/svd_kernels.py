from itertools import chain, combinations

import polars as pl

from .constants import data_dir, with_suffix


def add_kernels(
    sample: str,
    out_file: str = None,
    input_columns: list[pl.type_aliases.IntoExprColumn] = None,
    target_column: str = "sensor_voltage",
) -> None:
    """
    Adds numerous new kernels to the joined sas and rtm file. These kernels will allow
    for a support vector machine to be trained and used to predict sas measurement
    values.
    :param sample: The name of the file to which the kernels are to be assigned.
    :param out_file: The name of the file in which this new dataset will be stored.
    :param input_columns: The names of the columns which are the original input columns.
    :param target_column: The name of the column which contains the target values.
    :return: None, but makes a new file with the added kernels.
    """
    if input_columns is None:
        input_columns = ["volt_1", "volt_2", "volt_7", "distance_to_sensor"]

    if out_file is None:
        out_file = with_suffix(sample, "_kernels.pq")

    expressions: list[pl.Expr] = []

    for column_combination in chain.from_iterable(
        combinations(input_columns, i) for i in range(1, len(input_columns))
    ):
        expr = pl.lit(1)
        for col in column_combination:
            expr = expr.mul(col)

        expressions.append(expr.alias("*".join(column_combination)))

    for col in input_columns:
        expressions.append((pl.col(col).pow(2)).alias(f"{col}_squared"))

    (
        pl.scan_parquet(data_dir(f"samples/{sample}"))
        .select(*input_columns, target_column)
        .with_columns(expressions)
        .sink_parquet(data_dir(f"samples/{out_file}"))
    )


def ensure_kernels(cleaned: str, *, original: str) -> None:
    """
    Makes sure the file with the additional kernels exists on the system. If it does not
    it is made.
    :param cleaned: The name of the new kernel file.
    :param original: The name of the original file, in this case the linked
    sas and rtm file.
    :return: None, but potentially makes a new file.
    """
    import os

    if os.path.isfile(data_dir(f"samples/{cleaned}")):
        return

    add_kernels(original, out_file=cleaned)
