# BoolQ

This repo contains code to train and evaluate the main results from the BoolQ
paper (https://arxiv.org/abs/1905.10044). This includes the recurrent model for
MultiNLI and BoolQ, and BERT for BoolQ.

## Data
The code uses data from the following sources:

* BoolQ: https://github.com/google-research-datasets/boolean-questions

* MultiNLI: https://www.nyu.edu/projects/bowman/multinli

* fastText embeddings: https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip

## Recurrent Model

Use `python -m language.boolq.run_recurrent_model_boolq`

For example (BoolQ only):

```
python -m language.boolq.run_recurrent_model_boolq \
--train_data_path /path/to/boolq/train.json \
--dev_data_path /path/to/boolq/dev.jsonl \
--fasttext_embeddings /path/to/fasttext/embeddings \
--num_train_steps 22000 --batch_size 24  --dataset boolq
```

or (MultiNLI only)

```
python -m language.boolq.run_recurrent_model_boolq \
--train_data_path /path/to/multinli/train.json \
--dev_data_path /path/to/multinli/dev.jsonl \
--fasttext_embeddings /path/to/fasttext/embeddings \
--num_train_steps 180000 --batch_size 32 --dataset multinli
```

The model can be re-trained from an existing checkpoint using the
`--checkpoint_file` flag.

The model can be evaluated using:

```
python -m language.boolq.run_recurrent_model_boolq \
--train=false \
--dev_data_path /path/to/multinli/dev.jsonl \
--fasttext_embeddings /path/to/fasttext/embeddings
```

## BERT

Use `python -m language.boolq.run_bert_boolq`

This script takes a similar set of parameters to the original
`run_classifier.py` from the BERT codebase, along with flags that point to the
location of the BoolQ dataset. For example:

```
python -m language.boolq.run_bert_boolq \
--vocab_file vocab.txt \
--bert_config_file bert_config.json \
--init_checkpoint bert_model.ckpt \
--boolq_train_data_path /path/to/train.jsonl \
--boolq_dev_data_path /path/to/dev.jsonl \
--do_train --do_eval_dev --output_dir /path/to/output-dir
```

It defaults to the hyper-parameters we used to get our BoolQ results.

To reproduce the best results with BERT pre-trained on MultiNLI, the
`init_checkpoint` flag should point to a MultiNLI checkpoint trained with the
original BERT codebase (https://github.com/google-research/bert).
