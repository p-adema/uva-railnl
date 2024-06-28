# Prediction results

Predictions done by the models, saved for later visualisation in the [sas graphs notebook](../src/sas_graphs.ipynb)

These should all have the form of `<model>_<loss>.npy`, e.g. `time_mse.npy`, and contain
the predictions of the best models, fitted on the training data, of the test split. Since
the preprocessing steps might result in different inputs being presented in the test set,
the only comparisons between model types that should be made are distributional comparisons.