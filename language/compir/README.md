
# Unlocking Compositional Generalization in Pre-trained Models Using Intermediate Representations

This directory contains code related to the paper "Unlocking Compositional Generalization in Pre-trained Models Using Intermediate Representations"
(Jonathan Herzig, Peter Shaw, Ming-Wei Chang, Kelvin Guu, Panupong Pasupat, Yuan Zhang).

The current version of this library contains:

1. Code for reproducing the train and test sets used in the paper for the different transformations (reversible and lossy).
2. Code for evaluating model predictions against gold programs.


## Datasets

Below are instructions for downloading all splits for all of the datasets we experimented with

We use a standard format for representing all splits before applying transformations, where each
line corresponds to an example and is formatted as:

`IN: <source> OUT: <target>\n`

Where `<source>` is the input utterance and `<target>` is the program.

The datasets downloaded from resources below should be parsed to this format before applying transformations.

### CFQ

Instructions to produce the SCAN MCD splits are here:

https://github.com/google-research/google-research/tree/master/cfq


### Text-to-SQL

The i.i.d and template splits for ATIS, GeoQuery and Scholar are here:

https://github.com/jkkummerfeld/text2sql-data/tree/master/data

### SCAN

The "turn left" and "length" splits, as well as the original "simple" splits, are available here:

https://github.com/brendenlake/SCAN

Instructions to produce the SCAN MCD splits are here:

https://github.com/google-research/google-research/tree/master/cfq#scan-mcd-splits


## Intermediate representations

### Data generation

The script `transform/apply_transformation.py` prepares the train and test data for a specific transformation. For running it, provide a train and test set in the format described above, along with the desired transformation.
Example usage:

```shell
python -m language.compir.transform.apply_transformation \
--transformation="rir" \
--dataset="cfq" \
--split="mcd1" \
--train_data_path="/path/to/.../cfq_mcd1_train.txt" \
--test_data_path="/path/to/.../cfq_mcd1_test.txt" \
--output_path="/path/to/.../output_dir"
```
For this example, the files `cfq_mcd1_rir_train.tsv` and `cfq_mcd1_rir_test.tsv`, which have programs in their reversible intermediate representation, will be created under the `output_path` directory.

Note that when preparing data for `seq2seq_2` (corresponds to transformations with a `2` suffix), the test set predictions that `seq2seq_1` outputs should be given to `apply_transformation.py` as the extra `--prediction_path` parameter. In this prediction file, the i<sup>th</sup> row should hold the prediction for the i<sup>th</sup> test example.

### Evaluation
The script `evaluate/evaluate_predictions.py` evaluates model predictions against the gold test data. For running it, provide a train and test set in the format described above, along with a prediction file and the transformation that was used to create the predictions.
Example usage:

```shell
python -m language.compir.evaluate.evaluate_predictions \
--transformation="rir" \
--dataset="cfq" \
--train_data_path="/path/to/.../cfq_mcd1_train.txt" \
--test_data_path="/path/to/.../cfq_mcd1_test.txt" \
--prediction_path="/path/to/.../predictions.txt" \
```
For this example, the files `predictions.txt` holds the predictions for CFQ MCD1 test examples in a reversible intermediate representation.


## T5

Instructions for fine-tuning T5 given a dataset in the TSV format described
above are here:

https://github.com/google-research/text-to-text-transfer-transformer#using-a-tsv-file-directly

This document also contains instructions for generating predictions:

https://github.com/google-research/text-to-text-transfer-transformer#decode


