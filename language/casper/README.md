# CASPER: ControllAble Semantic Parser via Exemplar Retrieval

## Introduction

CASPER is a retrieval-augmented semantic parser: given an input query, the parser retrieves related exemplars (e.g., training examples with similar inputs) from a retrieval index, augments them to the query, and then applies a seq2seq model to produce an output parse.

The retrieved exemplars give additional information to the parser, which leads to an **improved accuracy** in the standard train-test setup. However, the main power of retrieval augmentation is **controllability**: by manipulating the retrieval index or how the augmented query is constructed, we can change the behavior of the parser without having to re-train the model. We demonstrate this in three use cases:

1. **Domain bootstrapping:** By adding examples from a new domain to the retrieval index at test time, CASPER can parse examples in the new domain without model re-training, while also preserving performance on other domains.

2. **Parse guiding:** We can train CASPER to follow the semantic template of the manually provided exemplars when asked to do so, while maintaining accuracy on the standard setup. This can be used to override the parses of problematic queries.

3. **Schema refactoring:** By editing the retrieval index, CASPER can, without re-training, adapt to a new semantic schema where some semantic labels are renamed/split into unseen labels.

For details, please check out our paper:

> [**Controllable Semantic Parsing via Retrieval Augmentation**](https://casperparser.page.link/paper) (EMNLP 2021)<br>
> Panupong Pasupat, Yuan Zhang, Kelvin Guu.

If you find the paper or code useful, please consider citing:

```
@inproceedings{pasupat2021casper,
  title={Controllable Semantic Parsing via Retrieval Augmentation},
  author={Panupong Pasupat and Yuan Zhang and Kelvin Guu},
  booktitle={EMNLP},
  year={2021}
}
```

## Code

This codebase contains scripts for the experiments in the paper. Note that we did not intend to demonstrate how CASPER should be deployed in an actual system. For instance, we cache the retrieval of dev/test examples for convenience; a real system should deploy an online retriever.

The following instructions will train a CASPER<sub>orig</sub> model on the standard train-test setup (Section 3 in the paper). See the [Experiments in the paper](#experiments-in-the-paper) section for how to run other experiments.

### Setup

1. **Download the code** by cloning the repository. Note that since the code does not depend on other `language` modules, it is fine to download just this subdirectory:

  ```shell
  mkdir language && cd language
  svn export https://github.com/google-research/language/trunk/language/casper language/casper
  ```

2. **Set up the environment:** The dependencies include:
  * Python 3.7+
  * Tensorflow 2.1+
  * Tensorflow Hub
  * Tensorflow Text (with the matching version as Tensorflow)

  Additionally, you will need access to T5 ([Official](https://github.com/google-research/text-to-text-transfer-transformer) / [HuggingFace](https://huggingface.co/transformers/model_doc/t5.html)) or any seq2seq model of your choice.

3. **Make sure `language` is in `PYTHONPATH`:** Either run the code while at the root of this repository, or set the following environment variable:

  ```shell
  # This is the root `language` directory, which contains `language/casper`
  export PYTHONPATH="/absolute/path/to/language/:${PYTHONPATH}"
  ```

4. **Download [the MTOP dataset](https://fb.me/mtop_dataset)** and extract it.

5. **Set up the data directories:**

  ```shell
  export ORIG_DIR="/path/to/mtop/dataset/"
  export DATA_DIR="/your/output/directory/"
  mkdir -p "${DATA_DIR}/raw" "${DATA_DIR}/t5"
  ```


### Preprocess the MTOP dataset

Run the following command to convert MTOP TSV files into preprocessed JSONL files.

```shell
# Train data
python -m language.casper.utils.mtop_tsv_to_jsonl \
  --alsologtostderr \
  --infile="${ORIG_DIR}/en/train.txt" \
  --outfile="${DATA_DIR}/raw/train.no_exemplars.jsonl"

# Dev data
python -m language.casper.utils.mtop_tsv_to_jsonl \
  --alsologtostderr \
  --infile="${ORIG_DIR}/en/eval.txt" \
  --outfile="${DATA_DIR}/raw/dev.no_exemplars.jsonl"

# Test data
python -m language.casper.utils.mtop_tsv_to_jsonl \
  --alsologtostderr \
  --infile="${ORIG_DIR}/en/test.txt" \
  --outfile="${DATA_DIR}/raw/test.no_exemplars.jsonl"
```

### Cache retrievals

We use the cosine distance between query embeddings as the retrieval score.  The following command constructs a retrieval index from the training examples, and then dumps the top 100 neighbors (excluding the trivial retrieval) of each train/dev example into JSONL files.

```shell
python -m language.casper.retrieve.cache_query_retrievals \
  --alsologtostderr \
  --retriever=use --embedder_size=large --neighbor_filter=simple \
  --index_files="${DATA_DIR}/raw/train.no_exemplars.jsonl" \
  --example_files="${DATA_DIR}/raw/train.no_exemplars.jsonl","${DATA_DIR}/raw/dev.no_exemplars.jsonl" \
  --output_files="${DATA_DIR}/raw/train.use-large.jsonl","${DATA_DIR}/raw/dev.use-large.jsonl"
```

### Analysis: Intrinsic retrieval metrics

The following command evaluates the retrievals.

```shell
python -m language.casper.evaluate.evaluate_retrieval \
  --alsologtostderr \
  --index_file="${DATA_DIR}/raw/train.no_exemplars.jsonl" \
  --retrieval_file="${DATA_DIR}/raw/dev.use-large.jsonl" \
  --max_num_neighbors=5
```

The reported metrics are:

*   Percentage of examples where one of the top K neighbors has the gold intent.

*   Percentage of examples where one of the top K neighbors has the gold template (called "frame" in the code).

*   Percentage of examples where all gold labels (intents and slots) are in the top K neighbors.

*   Percentage of gold labels covered by the top K neighbors.

where K runs from 1 to `max_num_neighbors`.

### Generate a retrieval-augmented dataset

We now generate a retrieval-augmented dataset for the standard train-test setup (no index manipulation).

The first command augments each **training** query with k = 5 **sampled** exemplars. Sampling is done n = 20 times per example to add diversity (so the dataset is 20 times bigger than the original). The results are sharded TSV files with prefixes `train.tsv` in `"${DATA_DIR}/t5/demo-run/`.

```shell
python -m language.casper.augment.cached_retrieval_to_dataset \
  --alsologtostderr \
  --train_data_paths="${DATA_DIR}/raw/train.use-large.jsonl" \
  --index_paths="${DATA_DIR}/raw/train.no_exemplars.jsonl" \
  --output_dir="${DATA_DIR}/t5/demo-run" \
  --file_format="tsv" \
  --example_converter=add_samp \
  --converter_kwargs='{"k":5,"n":20}'
```

The second command augments each **development** query with k = 5 **top** exemplars. The results are sharded TSV files with prefixes `dev.tsv`.

```shell
python -m language.casper.augment.cached_retrieval_to_dataset \
  --alsologtostderr \
  --dev_data_paths="${DATA_DIR}/raw/dev.use-large.jsonl" \
  --index_paths="${DATA_DIR}/raw/train.no_exemplars.jsonl" \
  --output_dir="${DATA_DIR}/t5/demo-run" \
  --file_format="tsv" \
  --example_converter=add_top \
  --converter_kwargs='{"k":5}'
```

### Train a model

Train a seq2seq model on the training data (`train.tsv*`) and produce predictions for the development data (`dev.tsv*`)

This step depends on the model you use. For T5, follow the instructions for [fine-tuning a model on TSV files](https://github.com/google-research/text-to-text-transfer-transformer#using-a-tsv-file-directly) and [producing predictions](https://github.com/google-research/text-to-text-transfer-transformer#decode).

### Evaluate the predictions

To evaluate the model, prepare a prediction file `pred.tsv` containing one prediction per line, with the lines aligning with the development data file (`dev.tsv`). Then run:

```shell
python -m language.casper.evaluate.evaluate_mtop_predictions \
  --alsologtostderr \
  --gold_file="${DATA_DIR}/t5/demo-run/dev.tsv-00000-of-00001" \
  --pred_file="/path/to/pred.tsv"
```

## Experiments in the paper

After [preprocessing the MTOP dataset](#preprocess-the-mtop-dataset), run the following bash scrips in `scripts/` to cache retrievals and generate datasets. (Make sure the environment variable `${DATA_DIR}` is set.)

* `standard_cache_retrievals.sh`: With training data as the retrieval index, cache the top 100 retrievals of training, development, and test examples. The resulting files will be used in all setups.

* `domain_bootstrap_cache_retrievals.sh`: Cache the retrievals for domain bootstrapping experiments. Care is taken to exclude unseen examples from the retrieval index. This is only used for the domain bootstrapping setup.

* `standard_gen_datasets.sh`: Generate datasets for the standard setup and ablations. Some datasets here are used in other setups as well.

* `domain_bootstrap_gen_datasets.sh`: Generate datasets for the domain bootstrapping setup.

* `parse_guiding_gen_datasets.sh`: Generate datasets for the parse guiding setup.

* `schema_refactor_gen_datasets.sh`: Generate datasets for the schema refactoring setup.

The tables in `EXPERIMENTS.md` list the training and evaluation data for each experiment in the paper.

* The **finetune_on** column specifies the subdirectories in `${DATA_DIR}/t5/` to get training data from. The model should be fine-tuned on the `train.*` files from such subdirectories, with a uniform mixture (i.e., an equal chance of picking each directory).

* The **predict_on** column specifies the evaluation data. The model should perform inference on the `dev.*` files from the specified subdirectory (or the `test.*` files for test set evaluation in the standard setup).
