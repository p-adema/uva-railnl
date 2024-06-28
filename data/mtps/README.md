# MTPS data
Data from the MTPS datasource, containing info for identifying trains

The base of the MTPS chain is either a Sherlock GPS file (`gps_2024-04-22.csv`) or a filtered GPS output (`GPS_filter.csv`). 
The script [clean_gps.py](../../src/clean/clean_gps.py) turns this into `gps.pq`,
which is used by [preprocess_mtps.py](../../src/clean/preprocess_mtps.py) to produce `gps_preprocessed.pq` (where `trip_id` has been identified).

`gps_preprocessed` is later combined with `rtm/cleaned.pq` to produce `rtm/train.pq`