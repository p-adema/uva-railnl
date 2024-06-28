# Regularly used functions and constant values such as the positions of the sensors and
# values used for calculations.

SENSOR_POSITIONS = [
    (52.341436, 5.151784),
    (52.309062, 5.073484),
    (51.996604, 5.984491),
    (51.946364, 4.388627),
    (51.427086, 4.132634),
    (52.164124, 4.991159),
    (51.428747, 4.238175),
    (51.998169, 5.988049),
    (52.361899, 5.178191),
    (52.350055, 6.566841),
    (52.414185, 5.355776),
    (52.338693, 4.826204),
]

LAT_TO_KM = 111_139
LON_TO_KM = 87_578


def data_dir(file: str) -> str:
    """
    Finds the correct path to a specific file in the data directory and returns that
    path for an easy way to load files.
    :param file: The name of the file of which the location has to be found
    :return: A string containing the path to the data file.
    """
    import os

    cwd = os.getcwd()
    src_pos = cwd.rfind("/src" if os.name != "nt" else r"\src")
    project_root = cwd if src_pos == -1 else cwd[:src_pos]
    path = f"{project_root}/data/{file}"

    if os.name == "nt":
        path = path.replace("/", "\\")
    return path


def with_suffix(file: str, suffix: str) -> str:
    """
    Makes a new file name based on an old one and a specific suffix.
    :param file: The name of the file.
    :param suffix: The suffix to be added to the end of the file, replacing the current.
    :return: The new file name.
    """
    return file.rsplit(".", maxsplit=1)[0] + suffix
