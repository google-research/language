# Exploring Unexplored Generalization Challenges for Cross-Database Semantic Parsing

This directory contains code necessary to replicate the training and evaluation for the ACL 2020 paper ["Exploring Unexplored Generalization Challenges for Cross-Database Semantic Parsing"](https://www.aclweb.org/anthology/2020.acl-main.742/) (Alane Suhr, Ming-Wei Chang, Peter Shaw, and Kenton Lee).

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

The rest of the README will refer to these directories, referenced as being in the top-level of the directory:

* `xsp_experiment_run/` is the parent directory that contains all the information about the run.
* `base_model_config.json` is the config file that contains hyperparameters for the run, similar to the file `model/model_config.json`.
* `assets/` contains data like the vocabulary.
* `examples/` is the directory containing text-based examples.
* `experiment_trial_*` contain saves for one or more trials of the experiment.
* `tf_examples` contains the TFRecords files for the input examples.

## (0) Environment setup and running python scripts

We suggest creating a conda environment for installation of dependencies.

```
conda create --name xsp tensorflow==1.15
source activate xsp
pip install -r language/xsp/requirements.txt
```

Run all python scripts while in the top-level of this repository using `-m`. For example,

`python -m language.xsp.data_preprocessing.convert_to_examples`

## (1) Downloading resources

The script `data_download.sh` contains directives to download all the data necessary to train and evaluate a model. For some operations, it is automatic, but for others it requires the user to manually download data from specific URLs. See the below for running this script for training and evaluation.

### For training

To download the training data (Spider and WikiSQL), run the `data_download.sh` script. For better organization, we suggest creating a separate directory to store the data, e.g., `data/`, then running the download script inside the directory.

To download only Spider and WikiSQL, run the script with the value `train_only` as the first argument. For example, in this directory:

```
mkdir data/
cd data/
sh language/xsp/data_download.sh train_only
```

You must also download resources for training the models (e.g., a pre-trained BERT model). Clone the [official BERT repository](https://github.com/google-research/bert) and download the BERT-Large, uncased model. We didn't use the original BERT-Large model in our main experimental results, but performance using BERT-Large is slightly behind BERT-Large+ on the Spider development set (see Table 3 in the main paper). You can ignore the vocabulary file in the zipped directory.

Finally, for the input training vocabulary, please download the text file from [this link](https://storage.googleapis.com/xsp-files/input_bert_vocabulary.txt) or `gs://xsp-files/input_bert_vocabulary.txt` via `gsutils`. We recommend to save it in the `assets` directory for each run.

### For evaluation

Before downloading the evaluation data, you need to prepare your environment for both MySQL and SQLite:

* You will need to download, install, and initialize a MySQL server. Details on how to do this are available on [the offical MySQL website](https://dev.mysql.com/doc/refman/5.7/en/installing.html). You need to set up a server because the databases that will be downloaded run on MySQL and will be added to your MySQL server.
* You will also need to install `sqlite3`, e.g., from [the official repo](https://www.sqlite.org/download.html). SQLite does not require running a server. We run inference on the sqlite3 databases, and only use the MySQL server in order to convert from the original database files to sqlite3 files.

To download the evaluation data (the eight additional datasets), run the `data_download.sh` script. Again, we suggest storing this data in the `data/` directory and running the script inside the directory. To download the evaluation data, leave out the `train_only` argument.

For evaluation, the script is a bit more complex than just downloading annotations. It does the following:
1. Downloads the annotations from [Finegan-Dollak & Kummerfeld's repository associated with the 2018 ACL paper](https://github.com/jkkummerfeld/text2sql-data).
1. Downloads and installs the actual databases into a separate directory, `databases`. *NOTES and WARNINGS*:
    * This may take a while to download, and may require you to manually download some databases yourself.
    * Some of these databases are quite large (several GB of storage).
    * The database installation function (`database_installation`) in `database_download.sh` assumes that the MySQL server you are running uses the username `root` and no password. You may need to modify this function if you used a different username and/or added a password.
1. Each database also must be converted from MySQL to SQLite3 format. We use the conversion tool provided by [Jean-Luc Lacroix](https://gist.github.com/esperlu/943776).
1. An empty copy of each database is created. We perform beam search re-ranking by throwing items out of the beam that are not executable on the database. To test this, we use empty databases to ensure that no database contents are used during inference time.
1. (TODO) Indices are added to some of the larger databases to make execution faster in these databases.
1. (TODO) Caches of gold query executions are created for each dataset. This makes evaluation faster if evaluating multiple model checkpoints. *NOTE*: On some datasets and databases, this process can take a very long time to complete.

After installation is complete, you don't need the MySQL databases anymore. We recommend dropping these databases from the server, because they take up a lot of disk space.

## (2) Data preprocessing

### For training

There are three steps for preprocessing the training data: (a) converting to a standard JSON serialized version of examples, (b) computing the output vocabulary from these examples, and (c) converting those examples to TFRecords for use as input to a model.

#### (a) Converting raw data to JSON

`data_preprocessing/convert_to_examples.py` will convert from the original format of each dataset  to a JSON list file containing serialized `NLToSQLExample` objects (see `data_preprocessing/nl_to_sql_example.py`).

This target takes several arguments:

* `dataset_name`: The name of the dataset (e.g., for training, `spider` or `wikisql`).
* `input_filepath`: The path containing the examples (usually pointing to a JSON file). The file should have been downloaded in the data download step above. The parent directory containing the JSON file should contain a CSV that defines the schema(s) of the database.
* `splits`: A list of the dataset splits that should be processed (e.g., for training, `train`).
* `output_filepath`: The name of the file to write the JSON to.
* `tokenizer_vocabulary`: The location of a vocabulary file that the tokenizer will use to tokenize natural language strings. This file contains special placeholder vocabulary tokens for new tokens we added, including tokens for table separators and column types. TODO: Commit the vocabulary file that we used.
* `generate_sql`: Whether to process the gold SQL queries and include them in the JSON list file. If training, this should be `True`; if evaluating, this should be `False`.
* `anonymize_values`: Whether to anonymize the values (e.g., strings and numerical values) present in the gold SQL queries by replacing them with a placeholder value. Only relevant if `generate_sql` is `True`.

An example of running this for creating the Spider and WikiSQL training data is:

```
# Spider
python -m language.xsp.data_preprocessing.convert_to_examples --dataset_name=spider --input_filepath=data/spider/ --splits=train --output_filepath=xsp_experiment_run/examples/spider_train.json --generate_sql=True --tokenizer_vocabulary=xsp_experiment_run/assets/input_bert_vocabulary.txt

# WikiSQL
python -m language.xsp.data_preprocessing.convert_to_examples --dataset_name=wikisql --input_filepath=data/wikisql/ --splits=train --output_filepath=xsp_experiment_run/examples/wikisql_train.json --generate_sql=True --tokenizer_vocabulary=xsp_experiment_run/assets/input_bert_vocabulary.txt
```

For training, you should also convert a validation set. For example, for the Spider development set:

```
python -m language.xsp.data_preprocessing.convert_to_examples --dataset_name=spider --input_filepath=data/spider/ --splits=dev --output_filepath=xsp_experiment_run/examples/spider_dev.json --tokenizer_vocabulary=xsp_experiment_run/assets/input_bert_vocabulary.txt
```

#### (b) Creating the output vocabulary

`data_preprocessing/create_vocabularies.py` will create output vocabulary (as a `.txt` file with a single word type per line) for the provided training data JSON(s).

This target takes several arguments:

* `data_dir`: A directory containing the JSON files to process (e.g., in the example above, `xsp_experiment_run/examples/`).
* `input_filenames`: A list of filenames containing the JSON information describing the training examples, as generated by step (a) (e.g., from the example above, `spider_examples.json` and `wikisql_examples.json`).
* `output_path`: The filepath where the vocabulary file will be saved.

For example:

```
python -m language.xsp.data_preprocessing.create_vocabularies --data_dir=xsp_experiment_run/examples/ --input_filenames=spider_train.json,wikisql_train.json --output_path=xsp_experiment_run/assets/output_vocab.txt
```

#### (c) Converting to TFRecords

`data_preprocessing/convert_to_tfrecords.py` will convert the JSON data created by `convert_to_examples` into TFRecords. This is the format that the model takes as input for training and inference.

This target takes several arguments:

* `examples_dir`: A directory containing the the JSON files to process (e.g., in the example above, `xsp_experiment_run/examples/`).
* `filenames`: The name(s) of the file(s) that should be converted to TFRecords. Example files are `spider_examples.json` and `wikisql_examples.json` as above.
* `output_vocab`: A file containing the output vocabulary comptued by `create_vocabularies`; one line per output token.
* `tf_examples_dir`: The location to save the TFRecords. This directory will be created if it doesn't exist.
* `generate_output`: Whether to process the gold SQL queries and include them in the output TFRecords.
* `permute`: Whether to permute the training schemas (table and column name serialization) for Spider data only.
* `num_spider_repeats`: The number of permutations to create. We use 7 repeats in our experiments.
* `config`: A filepath to a model config. An example of such a config is in `model/model_config.json`.

For example:

```
# Convert the Spider and WikiSQL training data
python -m language.xsp.data_preprocessing.convert_to_tfrecords --examples_dir=xsp_experiment_run/examples/ --filenames=spider_train.json,wikisql_train.json --output_vocab=xsp_experiment_run/assets/output_vocab.txt --generate_output=True --permute=True --num_spider_repeats=7 --config=xsp_experiment_run/model/model_config.json --tf_examples_dir=xsp_experiment_run/tf_records

# Convert the Spider evaluation data
python -m language.xsp.data_preprocessing.convert_to_tfrecords --examples_dir=xsp_experiment_run/examples/ --filenames=spider_dev.json --output_vocab=xsp_experiment_run/assets/output_vocab.txt --permute=False --config=xsp_experiment_run/model/model_config.json --tf_examples_dir=xsp_experiment_run/tf_records
```

*Note*: This target may take a long time to finish.


### For evaluation

The steps to preprocess the evaluation data are similar to above, except for the following:

* (a) Converting raw data to JSON
    * The value of `split` is dependent on the data being evaluated. E.g, for ATIS, this would be `dev`; for Restaurants, this would be `0,1,2,3,4,5,6,7,8,9`.
    * `generate_sql` should be set to `False`.
* (b) Creating the output vocabulary: not necessary for evaluation data. Use the output vocabulary computed during training.
* (c) Converting to TFRecords
    * `output_vocab` should be the file computed before training from the training data.
    * `generate_output` should be set to `False`.
    * `permute` should be set to `False`.

## (3) Model training

After creating TFRecord files for the Spider and WikiSQL training data, and a TFRecord file for the Spider development set (using the evaluation procedure), you can now train the model.

`training/train_model.py` will train the model given several arguments:

* `do_train` and `do_eval` specify the mode of the model. Exactly one must be set.
* `tf_examples_dir` is the directory containing the TFRecords which should be used for training (generated in step 2c above)
* `config` is a filepath specifying the model config as a `json` file (e.g., `model/model_config.json` or `model/local_model_config.json`).
* `output_vocab` is a filepath pointing to the output vocabulary (generated in step 2b above).
* `training_filename` is a list of filenames containing TFRecords to use for training (generated in step 2c above)
* `eval_filename` is the filename containing TFRecords used for evaluation during training (usually the Spider development set).
* `model_dir` is the directory to save the model in. It will save the model checkpoints and evaluation results.
* `eval_batch_size` is the batch size to use during evaluation.
* `steps_between_saves` is the number of model steps between saving checkpoints.
* `max_eval_steps` is the number of evaluation steps to train on before terminating evaluation. When not set, all of the evaluation data is used for evaluating.

Make sure to edit your local model config to point to the correct path for the pretrained BERT (`pretrained_bert_dir`).

For example:

```
python -m language.xsp.training.train_model --do_train --tf_examples_dir=tf_records/ --config=language/xsp/model/local_model_config.json --output_vocab=assets/output_vocab.txt --training_filename=spider_train.tfrecords,wikisql_train.tfrecords --eval_filename=spider_dev.tfrecords --model_dir=language/xsp/experiment_trial_0
```
## (4) Model inference and evaluation

Once you have a model checkpoint that you would like to run inference on, you need to make sure you have evaluation data prepared to evaluate. This means you need to:

* Download the relevant evaluation data (see #1 above)
* Preprocess the relevant evaluation data inputs (see #2 above)

### (a) Running inference
