# REALM: Retrieval-Augmented Language Model Pre-Training

- [Introduction](#introduction)
- [What's in this release](#whats-in-this-release)
- [Pre-training on a single machine](#pre-training-on-a-single-machine)
- [Fine-tuning on open domain question answering](#fine-tuning-on-open-domain-question-answering)

## Introduction

REALM is a method for augmenting neural networks with a **knowledge retrieval
mechanism**. For example, if a question-answering neural network is given a
question like *"What is the angle of an equilateral triangle?"*,
it could retrieve [this Wikipedia page](https://en.wikipedia.org/wiki/Equilateral_triangle)
to help determine the answer.

A unique aspect of REALM is the way that we train this retrieval mechanism.
Instead of relying on a pre-existing document retrieval system,
we train a **neural document retriever** using an unsupervised **fill-in-the-blank**
training objective.

The intuition behind the objective is simple. First, we blank out a few
words in a passage of text. The model then experiments with retrieving different
documents that it suspects are relevant. The documents that improve prediction
accuracy for the missing words are upweighted, while the rest are downweighted.

Once the neural network has been pre-trained in this fashion, the entire network
(including the retriever) can then be fine-tuned in an end-to-end manner for
downstream tasks such as open-domain question answering.

For details, please check out our paper:

> **[REALM: Retrieval-Augmented Language Model Pre-Training](https://arxiv.org/abs/2002.08909)** (ICML 2020)\
> Kelvin Guu*, Kenton Lee*, Zora Tung, Panupong Pasupat, Ming-Wei Chang.

If you find the paper or code useful, please consider citing:

```
@article{guu2020realm,
  title={{REALM}: Retrieval-augmented language model pre-training},
  author={Guu, Kelvin and Lee, Kenton and Tung, Zora and Pasupat, Panupong and Chang, Ming-Wei},
  journal={arXiv preprint arXiv:2002.08909},
  year={2020}
}
```

## What's in this release

### Code
This directory includes all code needed to perform the pre-training step of REALM.
Code for fine-tuning REALM to be an open-domain QA system resides in the
[ORQA](https://github.com/google-research/language/tree/master/language/orqa)
codebase. See [instructions below](#fine-tuning-on-open-domain-question-answering)
for passing the output of REALM as input to ORQA.

We have also designed the code to be readily modified/extended for other
applications that require:

- Large-scale retrieval on every training step, using Maximum Inner Product Search (MIPS)<sup>[1](#MIPS_footnote)</sup>.
- A MIPS index that is continuously updated over the course of training.

<sub><a name="MIPS_footnote">1</a>: Our original experiments used the
[ScaNN](https://github.com/google-research/google-research/tree/master/scann)
library for MIPS. However, as that was not open-sourced until recently, we
switched to using brute-force matrix multiplication in this release and found
that it was also sufficiently fast to reproduce our original results. In
this setting, the MIPS index simply amounts to a matrix of all document
embeddings. Note that refreshing this index is still a key part of the REALM
recipe, as document embeddings must still be re-computed after gradient steps
have been taken on the document embedder.</sub>

### Pre-trained model checkpoints

- REALM pre-trained with CC-News as the target corpus and Wikipedia as the
  knowledge corpus is available at `gs://realm-data/cc_news_pretrained`
- REALM fine-tuned to perform open-domain QA:
  - on WebQuestions: `gs://realm-data/orqa_wq_model_from_realm`
  - on NaturalQuestions: `gs://realm-data/orqa_nq_model_from_realm`

You can use [gsutil](https://cloud.google.com/storage/docs/gsutil) to download
from links with the `gs://` prefix. You can also browse such links by replacing
`gs://` with `https://console.cloud.google.com/storage/browser/` (requires
signing in with any Google account).

### Data
We are in the process of releasing the full pre-training corpus and retrieval
corpus (Wikipedia) needed for pre-training REALM. At this time, we provide a
small subset of the full data that can just be used to verify that the code
runs (see [below](#pre-training-on-a-single-machine) for details).

### Compute
We've provided instructions for pre-training REALM on a single machine. However,
in practice a single machine typically lacks the necessary computational
resources (80 TPUs and numerous CPUs), so we are now working on instructions
for distributing REALM pre-training across multiple machines.

## Pre-training on a single machine

For the sole purpose of understanding the code and debugging, we provide
instructions for pre-training REALM on a single machine using a scaled down model
architecture and a smaller dataset. Please note that good performance cannot be
expected from this setup (the dataset is far too small).

### Setup

1. **Set up the environment.** We recommend creating a conda environment:

    ```sh
    # Note that we use TF 1.15. This is because we use the tf.contrib package,
    # which was removed from TF 2.0.
    conda create --name realm python=3.7 tensorflow=1.15
    conda activate realm
    # We use TensorFlow Hub to download BERT, which is used to initialize REALM.
    pip install tensorflow-hub bert-tensorflow
    ```

2. **Make sure the "language" package is in PYTHONPATH.** Either run the code while at the root of this repository, or set the following environment variable:

    ```sh
    # Note that it is the root of the `language` repository, not the `realm` subdirectory.
    export PYTHONPATH="/absolute/path/to/language/:${PYTHONPATH}"
    ```

3. (Optional) **Change the directories in the launch script.**

    The default data directory (`DATA_DIR`) is `gs://realm-data/realm-data-small`
    (loads a small subset of the REALM pre-training data from Google Cloud Storage)
    and the default output directory (`MODEL_DIR`) is `./out`. These directories
    can be changed in `language/realm/local_launcher.sh`.

### Running the code

The pre-training process involves 3 systems that work together:

- Trainer
- Example generators
- Document index refresher

Launch these systems in the following order (in separate terminal windows):

1. The **document index refresher** embeds all documents in the retrieval corpus and constructs the retrieval index.

    It reads the document corpus from `DATA_DIR`, and starts
    embedding them as soon as a new model is available in `$MODEL_DIR/export/tf_hub_best`.
    The model and the document embeddings are then moved into `$MODEL_DIR/export/encoded`.
    Launch the index refresher like so:

    ```sh
    sh language/realm/local_launcher.sh refresh
    ```

2. The **example generators** are RPC servers that generate examples for the main trainer.

    To generate an example, we read an input sentence from the pre-training corpus
    (`$DATA_DIR/pretrain_corpus_small`), blank out a few words, embed it as a query
    vector, retrieve relevant documents from the retrieval corpus, and then
    package everything into a TensorFlow Example. Retrieval is done using Maximum
    Inner Product Search against the document embeddings produced by the index
    refresher.

    Launch example generators for training and evaluation data as follows:

    ```sh
    sh language/realm/local_launcher.sh gen-train
    sh language/realm/local_launcher.sh gen-eval
    ```

3. The main **trainer** should be run last.

    It fetches TensorFlow Examples from the example generators via RPC calls,
    and performs gradient descent on these examples. Best performing models are
    periodically exported as TensorFlow Hub modules to `$MODEL_DIR/export/tf_hub_best`,
    which are then picked up by the **document index refresher** and **example
    generators**. Launch the trainer like so:

    ```sh
    sh language/realm/local_launcher.sh train
    ```

Instead of running in separate windows, one can also launch all jobs in parallel using a single command:

```sh
sh language/realm/local_launcher.sh all
```

The training process can be monitored in the log files (`$MODEL_DIR/log`) or via TensorBoard:

```sh
tensorboard --logdir=$MODEL_DIR/
```

## Fine-tuning on open domain question answering

See the [ORQA](https://github.com/google-research/language/tree/master/language/orqa) codebase for details.
To fine-tune the ORQA model with REALM pre-training, set the flags for
`language/orqa/experiments/orqa_experiment.py` to the following values:

```sh
--retriever_module_path=gs://realm-data/cc_news_pretrained/embedder
--reader_module_path=gs://realm-data/cc_news_pretrained/bert
```
