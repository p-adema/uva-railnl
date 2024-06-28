# SAS data
Data from the SAS datasource, containing sensor voltage measurements

The base of the SAS chain is the `voltage-avg-feb-april.pq` file (renamed from `.parquet`)
This is cleaned into `avg_cleaned.pq`, which is later joined with RTM data.

The voltage bands were not used, as there was not enough data present.