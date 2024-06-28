# Training samples
Fully joined training data and the resulting train/tune/test splits

There are two types of files: `.pq` files to cache the preprocessing results, and `.npz` files that contain the actual splits and final training samples (but no column names: pure NumPy arrays).

All files here should be automatically generated, and the only process that might be time-consuming is the
interpolation that occurs in [time_expansion.py](../../src/clean/time_window.py)
to turn `simple_joined.pq` (or `train_joined.pq`, same data) into time windowed data.
This takes a bit under 10 minutes on a relatively new 12-core machine.