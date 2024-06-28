# Predicting rail-ground voltage using AI
As part of the UvA Second Year Project for ProRail, this project was created
as an attempt to predict rail-ground voltages using train metrics. All graphs
in the report and presentation can be found in the notebooks in [src/](src/README.md), and all the code
used to preprocess the datasets can be found in [src/clean/](src/clean/README.md).

Further documentation can be found in the README's of the relevant directories.

### Directory contents:
- [data](data/README.md): All parquet, csv and Numpy saved files containing the datasets.
- [models](models/README.md): Saved Keras models produced by the notebooks.
- [src](src/README.md) All project code, with the notebooks at the top-level.
  - [src/clean](src/clean/README.md) Data preprocessing code, written with Polars.


# Setup:

1. Create a virtual environment using `conda` or `mamba`:
    ```shell
    conda create -n uva-railnl python=3.11 numpy polars[all] matplotlib pandas pyarrow jupyter jax keras keras-tuner ruff ruff-lsp pydot
    ```
   ```shell
    conda activate uva-railnl
   ```
2. If you want to create the graphs present in e.g. [sas_graphs.ipynb](src/sas_graphs.ipynb), install [graphviz](https://www.graphviz.org/download/) via your package manager (`apt`, `brew`, etc.)

3. Ensure the base data files are present (see [the data readme](data/README.md)):
   - `sas/voltage-avg-feb-april.pq`
   - `rtm/cleaned.pq` (or the raw RTM datafiles, but these are very large)
   - `mtps/GPS_filter.csv`

4. Start Jupyter, and run the notebooks!

Note: the Git history for this project has been erased, as some commits contained traces of the data, and we are not permitted to share this publicly under our NDA.