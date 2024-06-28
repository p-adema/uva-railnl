# Data

All data files are stored in the appropriate subdirectories,
but are ignored by default in the [gitignore](../.gitignore).

For the preprocessing chain to start, the following files need to be present:
- `sas/voltage-avg-feb-april.pq`
- `rtm/cleaned.pq` (or the raw RTM datafiles, but these are very large)
- `mtps/GPS_filter.csv`

More detals are available in the README's of the relevant subdirectories:

### Directory contents:
- [mtps](mtps/README.md): Data from the MTPS datasource, containing info for identifying trains
- [rtm](rtm/README.md): Data from the RTM datasource, containing catenary voltage measurements. Also contains the 'train data' (RTM merged with MTPS)
- [sas](sas/README.md): Data from the SAS datasource, containing sensor voltage measurements
- [samples](samples/README.md): Fully joined training data and the resulting train/tune/test splits
- [pred](pred/README.md): Predictions done by the models, saved for later visualisation in the [sas graphs notebook](../src/sas_graphs.ipynb)
