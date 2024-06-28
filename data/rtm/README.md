# RTM data

Data from the RTM datasource, containing catenary voltage measurements. Also contains the 'train data' (RTM merged with
MTPS)

The base of the RTM chain is a collection of parquet files from all relevant dates. We
gathered all the parquet partitions (from `<day>/05/dataset-prorail/virm/measurements/data.parquet/*`)
into a single (27GB) file as this led to orders of magnitude faster processing in the later steps.

[clean_rtm.py](../../src/clean/clean_rtm.py) takes this large file (or directory) and
turns it into a tabular format, namely `cleaned.pq`. This is also the alternative chain base,
for if you don't want to store the full 27GB of RTM measurement data.

This is then fed into [link_rtm_mtps.py](../../src/clean/link_rtm_mtps.py), which combines
`cleaned.pq` and `mtps/gps_preprocessed.pq` to form `train.pq`
(this is very slow, about 8 hours for the full dataset).

`train.pq` is then fed into [preprocess_rtm.py](../../src/clean/preprocess_rtm.py) to produce `train_preprocessed.pq`.

This can either be directly linked with the SAS data to form `samples/simple_joined.pq`,
or first windowed using [space_window.py](../../src/clean/space_window.py) to produce `space_window.pq` before it is joined to form `samples/space_joined.pq`

(Time expansion happens on the final sample dataset `simple_joined.pq`, not at the RTM level)
