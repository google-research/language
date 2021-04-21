# CANINE: Pre-training an Efficient Tokenization-Free Encoder for Language Representation

[**Core Model**](#core-model) | [**Pre-trained Checkpoints**](#checkpoints)
[**TyDi QA System**](#tydi-qa-system) | [**Tips and Tricks**](#tips-and-tricks)
| [**Paper**](https://arxiv.org/abs/2103.06874)

This repository contains a reference implementation of CANINE, download links
for pre-trained checkpoints, and code for fine-tuning and evaluating on the TyDi
QA dataset.

Want to keep up to date on updates and new releases? Join our low-traffic
[announcement email list](https://groups.google.com/g/canine-announce).

## Introduction

CANINE is a tokenization-free vocabulary-free deep language encoder. Please see
the [paper](https://arxiv.org/abs/2103.06874) for details.

## Core Model

The core of the CANINE model implementation is in [`modeling.py`](modeling.py).

Typical usage is similar to BERT:

```python
config = CanineModelConfig.from_json_file(model_config_file)
model = CanineModel(config, input_ids, input_mask, segment_ids, is_training)
pooled_repr = model.get_pooled_output()  # For classificaton.
seq_repr = model.get_sequence_output()  # For tagging.
```

Input processing is trivial and can typically be accomplished as: `input_ids =
[ord(char) for char in text]`, yielding a list of UTF-32 Unicode codepoints.

## Pre-training Code (Coming later)

We've prioritized releasing the pre-trained checkpoints, modeling code, and TyDi
QA evaluation code since we hope this will cover the most common use cases. The
implementation of pre-training will be released in this repo in the future. If
this is blocking you, feel free to send us a friendly ping to let us know that
this is important to you.

## Checkpoints

*   **[CANINE-S (~500 MB)](https://storage.googleapis.com/caninemodels/canine-s.zip)**:
    Pre-trained with autoregressive character loss, 12-layer, 768-hidden,
    12-heads, 121M parameters.
*   **[CANINE-C (~500 MB)](https://storage.googleapis.com/caninemodels/canine-c.zip)**:
    Pre-trained with subword loss, 12-layer, 768-hidden, 12-heads, 121M
    parameters.

For both models, the checkpoints contain 133M parameters. 13M of these are in
the 16k position embeddings. While only 2k of these are pre-trained (due to our
maximum sequence length of 2048), we reserve 8X this many to make it easier to
conduct experiments with longer inputs. We do not include these unestimated 11M
parameters in the counts above.

## TyDi QA System

We evaluate on the TyDi QA Primary Tasks (TyDiQA-SelectP and TyDiQA-MinSpan).
Note that these tasks are different than the GoldP task, often seen in the
XTREME meta-benchmark. See the TyDi QA
[website](https://ai.google.com/research/tydiqa) and
[paper](https://www.aclweb.org/anthology/2020.tacl-1.30/) for more information.

For details on running the TyDi QA systems, see [`tydiqa/README.md`](tydiqa).

## Tips and Tricks

When adapting your existing code from BERT-like models to a tokenization-free
CANINE model, there are a few potential pitfalls to be aware of:

*   Carefully inspect your sequence length hyperparameters. For example, BERT
    models often use a maximum sequence length of 512 subwords, which would
    typically become 2048 in CANINE. `grep` works well for this.
*   Carefully inspect hyperparameters *derived* from length. For example, in
    TyDi QA, this includes parameters such as question length, answer length,
    document stride, etc.
*   Check that you've disabled your legacy tokenizers in all parts of your input
    preparation pipeline. Converting already-tokenized strings to codepoints
    will probably still work fine, but you will get the best performance by
    feeding natural text---without any punctuation splitting, etc.---to
    CANINE.
*   Check your datasets to see if they have fossilized tokenization artifacts
    inside them. This is especially common for tagging tasks where punctuation
    may be split and spaces may be removed/replaced with a special charcacter.
    In these cases, you should expect better quality by
    detokenizing/unnormalizing the data back to a more natural form.

## Citation

Please cite the [CANINE ArXiV Paper](https://arxiv.org/abs/2103.06874) as:

```tex
@misc{canine,
  title={{CANINE}: Pre-training an Efficient Tokenization-Free Encoder for Language Representation},
  author={Jonathan H. Clark and Dan Garrette and Iulia Turc and John Wieting},
  year={2021},
  eprint={2103.06874},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}
```

## Contact us

If you have a technical question regarding the dataset, code, or publication,
please send us email (see paper).
