# CANINE Baseline for TyDi QA Primary Tasks

This directory contains code for training and evaluating a CANINE baseline model
on [TyDi QA](https://ai.google.com/research/tydiqa).

This README and most of the code in this directory is copied from the
[TyDi QA baseline system](https://github.com/google-research-datasets/tydiqa/tree/master/baseline),
and is minimally modified to support CANINE in order to make it more useful as a
reference implementation.

**These instructions assume that you are in the main directory of the
[`google-research/language`](https://github.com/google-research/language)
repo, or that you have put it in your module path such that the python `-m` flag
will be able to find modules in this repo.**

The approach is nearly identical to the BERT baseline for the
[Natural Questions dataset](https://www.aclweb.org/anthology/Q19-1026.pdf)
described in [this paper](https://arxiv.org/abs/1901.08634). Initial
quality measurements for the BERT baseline system on TyDi QA are given in the
TyDi QA [TACL article](https://www.aclweb.org/anthology/2020.tacl-1.30/).

## Hardware Requirements

This baseline fine-tunes multilingual CANINE and so has similar compute and
memory requirements. Unlike BERT-base, CANINE requires 16 GB of GPU RAM.

## Install

This code runs on Python 3. You'll also need the following libraries---you can
skip the `pip install` steps below if you already have these on your system:

```sh
sudo apt install python3-dev python3-pip
pip3 install --upgrade tensorflow-gpu
pip3 install absl-py
pip3 install tf-slim
```

You'll probably also want a good GPU (or TPU) to efficiently run the model
computations.

Finally, download the latest multilingual CANINE checkpoint, which will serve as
a starting point for fine-tuning. See [../README.md](../README.md#checkpoints)
for download links.

## Get the Data

To get the data, see the instructions in the
[TyDi QA repo](https://github.com/google-research-datasets/tydiqa#download-the-dataset).

## Prepare Data

To run the TensorFlow baseline, we first have to process the data from its
original JSONL format into tfrecord format. These steps use only CPU (no GPU).

*You may wish to develop on a much smaller subset of the data. Because these are
JSONL files, standard shell commands such as `head` will work fine for this
purpose.*

First, process the smaller dev set to make sure everything is working properly:

```sh
python3 -m language.canine.tydiqa.prepare_tydi_data \
  --input_jsonl=tydiqa-v1.0-dev.jsonl.gz \
  --output_tfrecord=dev_samples/dev.tfrecord \
  --max_seq_length=2048 \
  --doc_stride=512 \
  --max_question_length=256 \
  --is_training=false
```

The output of this step will be about 3.0GB. You'll see some fairly detailed
debug logging (from [`debug.py`](debug.py)) for the first few examples.

Next, prepare the training samples:

```sh
python3 -m language.canine.tydiqa.prepare_tydi_data \
  --input_jsonl=tydiqa-v1.0-train.jsonl.gz \
  --output_tfrecord=train_samples.tfrecord \
  --record_count_file=train_samples_record_count.txt \
  --max_seq_length=2048 \
  --doc_stride=512 \
  --max_question_length=256 \
  --include_unknowns=0.1 \
  --is_training=true
```

The output of this step will be about 1.7GB. Note that this is smaller than the
dev set since we subsample negative examples during training, but must do
inference on entire articles for the dev set. This process will take
significantly longer since we process entire Wikipedia articles. This can take
around 10 hours for the training set on a single process. A bit of extra effort
of splitting this into multiple shards, running on many cores, and then
combining record counts may save you a significant amount of wall time if you
plan to run this multiple times. Otherwise, if you plan to run this once and
modify only the modeling, then running this overnight on a workstation should be
fairly painless.

## Train (Fine-tuning CANINE)

Next, we fine-tune on the TyDi QA training data starting from the multilingual
CANINE checkpoint, preferably on GPU:

```sh
python3 -m langauge.canine.tydiqa.run_tydi \
  --model_config_file=canine_dir/model_config.json \
  --init_checkpoint=canine_dir/canine_model.ckpt \
  --train_records_file=train_samples/*.tfrecord \
  --record_count_file=train_samples_record_count.txt \
  --do_train \
  --max_seq_length=2048 \
  --train_batch_size=512 \
  --learning_rate=5e-5 \
  --num_train_epochs=10 \
  --warmup_proportion=0.1 \
  --output_dir=~/tydiqa_baseline_model
```

## Predict

Once the model is trained, we run inference on the dev set:

```sh
python3 -m language.canine.tydiqa.run_tydi \
  --model_config_file=canine_dir/canine_config.json \
  --init_checkpoint=~/tydiqa_baseline_model \
  --predict_file=tydiqa-v1.0-dev.jsonl.gz \
  --precomputed_predict_file=dev_samples/*.tfrecord \
  --do_predict \
  --max_seq_length=2048 \
  --max_answer_length=100 \
  --candidate_beam=30 \
  --predict_batch_size=128 \
  --output_dir=~/tydiqa_baseline_model/predict \
  --output_prediction_file=~/tydiqa_baseline_model/predict/pred.jsonl
```

NOTE: Make sure you correctly set the `--init_checkpoint` to point to your
fine-tuned weights in this step rather than the original pretrained multilingual
CANINE checkpoint.

## Evaluate

For evaluation, see the instructions in the
[TyDi QA repo](https://github.com/google-research-datasets/tydiqa#evaluation).

We encourage you to fine-tune using multiple random seeds and average the
results over these replicas to avoid reading too much into optimization noise.

## Modify and Repeat

Once you've successfully run the baseline system, you'll likely want to improve
on it and measure the effect of your improvements.

To help you get started in modifying the baseline system to incorporate your new
idea---or incorporating parts of the baseline system's code into your own
system---we provide an overview of how the code is organized:

1.  [`data.py`](data.py) -- Responsible for deserializing the JSON and creating
    Pythonic data structures. *Usable by any ML framework / no TF dependencies.*

2.  [`preproc.py`](preproc.py) -- Splits input strings into codepoints and
    munges JSON into a format usable by the model. *Usable by any ML framework /
    no TF dependencies.*

3.  [`tf_io.py`](tf_io.py) -- Tensorflow-specific IO code (reads `tf.Example`s
    from TF records). *If you'd like to use your own favorite DL framework,
    you'd need to modify this; it's only about 200 lines.*

4.  [`tydi_modeling.py`](tydi_modeling.py) -- The core TensorFlow model code.
    **If you want to replace CANINE with your own latest and greatest, start
    here!** For example, you can directly subclass `TyDiModelBuilder` and
    override `create_encoder_model` to return a class with a different encoder
    architecture. *Similarly, if you'd like to use your own favorite DL
    framework, this would be the only file that should require heavy
    modification; it's only about 200 lines.*

5.  [`postproc.py`](postproc.py) -- Does postprocessing to find the answer, etc.
    Relevant only for inference (not used in training). *Usable by any ML
    framework with minimal edits. Has minimal tf dependencies (e.g. a few tensor
    post-processing functions).*

6.  [`run_tydi.py`](run_tydi.py) -- The main driver script that uses all of the
    above and calls Tensorflow to do the main training and inference loops.

# Citation

The citation for [TyDi QA](https://www.aclweb.org/anthology/2020.tacl-1.30/) is:

```tex
@article{tydiqa,
  title   = {{TyDi QA}: A Benchmark for Information-Seeking Question Answering in Typologically Diverse Languages},
  author  = {Jonathan H. Clark and Eunsol Choi and Michael Collins and Dan Garrette and Tom Kwiatkowski and Vitaly Nikolaev and Jennimaria Palomaki}
  year    = {2020},
  journal = {Transactions of the Association for Computational Linguistics}
}
```
