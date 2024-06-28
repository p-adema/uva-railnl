# Project source

In this directory, all the code for the project is present.
Preprocessing scripts are in the module [clean](clean/README.md)

### Models:
- [svm_model_search.ipynb](svm_model_search.ipynb): Support Vector Machine model (also contains hyperparameter search)
- [base_model.ipynb](base_model.ipynb): The base (simple) neural network model
- [space_model.ipynb](space_model.ipynb): Neural network with LSTM trained on space-expanded data
- [time_model.ipynb](time_model.ipynb): Neural network trained on time-expanded (interpolated) data

### Hyperparameter search
- [svm_model_search.ipynb](svm_model_search.ipynb): SVM hyperparameter search (and main models) 
- [base_search.ipynb](base_search.ipynb): Architecture search for the base model
- [space_search.ipynb](space_search.ipynb): Architecture search for the space-expanded model
- [time_search.ipynb](time_search.ipynb): Architecture search for the time-expanded model

### Extra
- [sas_graphs.ipynb](sas_graphs.ipynb): Graphs from the progress meetings, report and presentation
- [resample.ipynb](resample.ipynb): A short experiment showing that down-sampling the data is ineffective
- [non_mtps_trips.ipynb](non_mtps_trips.ipynb): A prototype method for identifying train trips without the MTPS identifiers. Abandoned after the MTPS data was provided, but might be of interest
