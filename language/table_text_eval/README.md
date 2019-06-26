# PARENT: Table to Text Evaluation

This folder contains scripts for computing the PARENT metric for table-to-text
evaluation, which is described in the following paper:

[Handling Divergent Reference Texts when Evaluating Table-to-Text Generation](https://arxiv.org/abs/1906.01081)\
Bhuwan Dhingra, Manaal Faruqui, Ankur Parikh, Ming-Wei Chang, Dipanjan Das, William W. Cohen\
ACL, 2019

## Computing PARENT

PARENT evaluates the generated text against both references and the table
itself. To compute PARENT on the generations in a file `<generation_file>`:

```
python -m language.table_text_eval.table_text_eval \
  --references <reference_file> \
  --generations <generation_file> \
  --tables <table_file>
```

Where the tables and references are in the `<table_file>` and `<reference_file>`,
respectively. See the `table_text_eval.py` script for more details on the
format for these files. To compute the
metric from another python program see `table_text_eval_test.py`.

By default, the scripts use the "Word Overlap Model" for determining
entailment. If you want to use the "Co-occurrence Model", pass in the
flags `--entailment_fn=cooccurrence` and `--cooccurrence_counts=<path_to_counts>`.
The latter is a JSON file mapping tokens in the table and pairs of tokens
from the table and text to their counts in a dataset, as follows:

```
for table, text in dataset:
  text_toks = _parse_text(text)
  table_toks = _parse_table(table)
  for btok in table_toks:
    counts[btok] += 1
    for xtok in text_toks:
      counts[btok + "|||" + xtok] += 1
```

Pre-computed co-occurrence counts for the WikiBio and WebNLG datasets are
included at the data links below.

## Reproducing the Paper's Results

This folder also contains the scripts for reproducing the correlation numbers
for PARENT in the paper.

### WikiBio

For WikiBio due to legal constraints we cannot directly release the human annotated
pairwise judgments. So we are releasing a pre-processed version where we compute
the aggregated score for each model across 500 bootstrap samples and release
only the aggregated scores:

1. WikiBio-Systems -- [bootstrap.json](https://storage.googleapis.com/table-text-eval/wikibio-systems/bootstrap.json), [processed_data.json](https://storage.googleapis.com/table-text-eval/wikibio-systems/processed_data.json).
2. WikiBio-Hyperparams -- [bootstrap.json](https://storage.googleapis.com/table-text-eval/wikibio-hyperparams/bootstrap.json), [processed_data.json](https://storage.googleapis.com/table-text-eval/wikibio-hyperparams/processed_data.json).
3. WikiBio [co-occurrence counts (gzipped)](https://storage.googleapis.com/table-text-eval/co-occurrence-counts/wikibio_cooccurrence_counts.json.gz)

These can be used to compute the correlations for PARENT:

```
python -m language.table_text_eval.wikibio_correlations \
  --bootstrap_file=<path_to_bootstrap.json> \
  --data_file=<path_to_processed_data.json> \
  --save_output=<output_file>
```

The correlations across all the bootstrap samples will be stored in
`<output_file>.correlations.json`.

### WebNLG Experiments

For WebNLG, first download the challenge results from the official repository
[here](https://gitlab.com/shimorina/webnlg-human-evaluation).
Then use the script provided here to pre-process it:

```
python preprocess_webnlg.py --data_dir=<path_to_repo> \
  --out_file=<processed_file.json>
```

To compute the correlations of PARENT for this data:

```
python webnlg_correlations.py \
  --data_file=<processed_file.json> \
  --save_output=<output_file>
```

Co-occurrence counts for the WebNLG data are available [here (gzipped)](https://storage.googleapis.com/table-text-eval/co-occurrence-counts/webnlg_cooccurrence_counts.json.gz).

## Citation

```
@inproceedings{dhingra2019handling,
  title={Handling divergent reference texts in table-to-text generation},
  author={Dhingra, Bhuwan and Faruqui, Manaal and Parikh, Ankur and Chang, Ming-Wei and Das, Dipanjan and Cohen, William W},
  booktitle={Proc. of ACL},
  year={2019}
}
```
