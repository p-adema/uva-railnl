# Cleaning scripts

Data preprocessing code, written with the Polars dataframe and expression library.

The filenames should be self-explanatory in most cases

### Cleaning
- [clean_gps.py](clean_gps.py)
- [clean_rtm.py](clean_rtm.py)
- [clean_sas.py](clean_sas.py)

### Preprocessing
- [preprocess_mtps.py](preprocess_mtps.py)
- [preprocess_rtm.py](preprocess_rtm.py)

### Merging
- [link_rtm_mtps.py](link_rtm_mtps.py)
- [link_rtm_sas.py](link_rtm_sas.py)

### Expanding
- [svd_kernels.py](svd_kernels.py)
- [space_window.py](space_window.py)
  - [space_pad.py](space_pad.py):  ensure padded inputs for space data
- [time_window.py](time_window.py)

### Generating samples
- [create_splits.py](create_splits.py): create `.npz` archives with the train/tune/test splits in `data/samples`

### Other
- [constants.py](constants.py): utilities used by the other scripts
- [space_extra_gps.py](space_extra_gps.py): an alternative (far slower) implementation of the space windowing that also counts the number of non-RTM trains nearby