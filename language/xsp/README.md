# Exploring Unexplored Generalization Challenges for Cross-Database Semantic Parsing

This directory contains code necessary to replicate the training and evaluation for the ACL 2020 paper ["Exploring Unexplored Generalization Challenges for Cross-Database Semantic Parsing](https://www.aclweb.org/anthology/2020.acl-main.742/) (Alane Suhr, Ming-Wei Chang, Peter Shaw, and Kenton Lee).

## Directory Structure

Our code is organized into four subdirectories:

* `data_preprocessing`: Code necessary for taking the raw training and evaluation data and creating preprocessed examples.
* `evaluation`: Code for evaluating the trained model.
* `model`: Defines the model architecture.
* `training`: Main script for running training.

Throughout the lifetime of training and evaluating a model, the files associated with a run (e.g., model save, preprocessed data) will be saved into a single directory associated with the experiment, which should have the following structure:

```
xsp_experiment_run/
    base_model_config.json
    assets/
    examples/
    experiment_trial_0/
    experiment_trial_1/
    experiment_trial_2/
    tf_examples/
```

The rest of the README will refer to these directories:

* `xsp_experiment_run/` is the parent directory that contains all the information about the run.
* `base_model_config.json` is the config file that contains hyperparameters for the run, similar to the file `model/model_config.json`.
* `assets/` contains data like the vocabulary.
* `examples/` is the directory containing text-based examples.
* `experiment_trial_*` contain saves for one or more trials of the experiment.
* `tf_examples` contains the TFRecords files for the input examples.

## (1) Downloading the data

TODO

## (2) Data preprocessing

### For training

There are two steps for preprocessing the training data: (1) converting to a standard JSON serialized version of examples, and (2) converting those examples to TFRecords for use as input to a model.

#### (a) Converting raw data to JSON

`data_preprocessing:convert_to_examples` will convert from the original format of each dataset  to a JSON list file containing serialized `NLToSQLExample` objects (see `data_preprocessing/nl_to_sql_example.py`).

This target takes several arguments:

* `dataset_name`: The name of the dataset (e.g., for training, `spider` or `wikisql`).
* `input_filepath`: The path containing the examples (usually pointing to a JSON file). The file should have been downloaded in the data download step above. The parent directory containing the JSON file should contain a CSV that defines the schema(s) of the database.
* `splits`: A list of the dataset splits that should be processed (e.g., for training, `train`).
* `output_filepath`: The name of the file to write the JSON to.
* `tokenizer_vocabulary`: The location of a vocabulary file that the tokenizer will use to tokenize natural language strings.
* `generate_sql`: Whether to process the gold SQL queries and include them in the JSON list file. If training, this should be `True`; if evaluating, this should be `False`.
* `anonymize_values`: Whether to anonymize the values (e.g., strings and numerical values) present in the gold SQL queries by replacing them with a placeholder value. Only relevant if `generate_sql` is `True`.

#### (b) Creating the output vocabulary


### For evaluation

The steps to preprocess the evaluation data are similar to above, except for the following:

* (a) Converting raw data to JSON
    * The value of `split` is dependent on the data being evaluated. E.g, for ATIS, this would be `dev`; for Restaurants, this would be `0,1,2,3,4,5,6,7,8,9'.
    * `generate_sql` should be set to `False`.
* (b) Creating the output vocabulary

