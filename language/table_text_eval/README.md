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

## Reproducing the Paper's Results

This folder also contains the scripts for reproducing the correlation numbers
for PARENT in the paper.

### WikiBio

For WikiBio due to legal constraints we cannot directly release the human annotated
pairwise judgments. So we are releasing a pre-processed version where we compute
the aggregated score for each model across 500 bootstrap samples and release
only the aggregated scores.

You can find the pre-processed data [here]() for the "Systems" category
and [here]() for the "Hyperparams" category in the paper. There are two
files for each category -- `bootstrap.json` and `processed_data.json`.
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

For WebNLG, a pre-processed version of the data available from the
[WebNLG challenge](http://webnlg.loria.fr/pages/results.html) can be
found [here](). Download the data and unzip it to a folder `<data_dir>`.

To compute the correlations of PARENT for this data:

```
python webnlg_correlations.py \  
  --data_dir=<data_dir> \  
  --save_output=<output_file>  
```

## Citation

```
@inproceedings{dhingra2019handling,  
  title={Handling divergent reference texts in table-to-text generation},  
  author={Dhingra, Bhuwan and Faruqui, Manaal and Parikh, Ankur and Chang, Ming-Wei and Das, Dipanjan and Cohen, William W},  
  booktitle={Proc. of ACL},  
  year={2019}  
}
```
