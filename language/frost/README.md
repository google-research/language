# FROST and Composition Sampling

This repository contains our pretrained and fine-tuned model checkpoints
(compatible with [Pegasus](https://github.com/google-research/pegasus)) and
their predictions from our paper
[FROST: Planning with Learned Entity Prompts for Abstractive Summarization](https://aclanthology.org/2021.tacl-1.88.pdf)
and
[A Well-Composed Text is Half Done! Composition Sampling for Diverse Conditional Generation](https://github.com/google-research/language/tree/master/language/frost).
We also include spaCy code for FROST-style annotation of targets.

This is not an official Google product. Please cite our papers if you use our
data or models.

```
@article{frost,
    title = "Planning with Learned Entity Prompts for Abstractive Summarization",
    author = "Narayan, Shashi  and  Zhao, Yao  and Maynez, Joshua  and
    Sim{\~o}es, Gon{\c{c}}alo  and Nikolaev, Vitaly  and McDonald, Ryan",
    journal = "Transactions of the Association for Computational Linguistics",
    volume = "9",
    year = "2021",
    publisher = "MIT Press",
    pages = "1475--1492",
}

@inproceedings{composition-sampling,
    title = "A Well-Composed Text is Half Done! Composition Sampling for Diverse
    Conditional Generation",
    author = "Narayan, Shashi  and  Sim{\~o}es, Gon{\c{c}}alo and Zhao, Yao  and
    Maynez, Joshua  and Das, Dipanjan and Collins, Michael and Lapata, Mirella",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for
    Computational Linguistics",
    year = "2022",
    publisher = "Association for Computational Linguistics",
}
```

## Introduction

We developed **FROST**, a simple but flexible mechanism to learn an intermediate
plan to ground the generation of abstractive outputs. Specifically, we prepend
(or **prompt**) target outputs with entity chains -- ordered sequences of
entities mentioned in the output. Transformer-based sequence-to-sequence models
are then trained to generate the entity chain and then continue generating the
output conditioned on the entity chain and the input.

Building on FROST, we developed **Composition Sampling**, a simple but effective
method to generate diverse outputs for conditional generation of higher quality
compared to previous stochastic decoding strategies. Our approach avoids text
degeneration by first sampling a composition in the form of an entity chain and
then using beam search to generate the best possible text grounded to this
entity chain.

The grounded generation with the planning objective in FROST and Composition
Sampling provides mechanism to control hallucinations and induce diversity, in
generated outputs. We achieved state-of-the-art results on Text Summarization
and Question Generation. We believe that NLG researchers will find our
repository as a valuable resource to train and study planning-based generation
models for more controllable, faithful and diverse text generation.


## Checkpoints and Predictions

Download FROST and Composition Sampling checkpoints and predictions from
[Google Cloud](https://console.cloud.google.com/storage/browser/frost-composition-sampling-data).

Alternatively in terminal, follow the instructions and install
[gsutil](https://cloud.google.com/storage/docs/gsutil_install). Then

```
mkdir frost-composition-sampling-data
gsutil cp -r gs://frost-composition-sampling-data/ frost-composition-sampling-data/
```

### FROST Pretrained Checkpoint

We modified GAP-sentence objective in
[Pegasus](https://arxiv.org/abs/1912.08777) to pretrain FROST models for
entity-level content planning and summary generation. We select n important
sentences using Self-Rouge from an input document, the selected sentences work
as a proxy for a human-authored abstractive summaries for the rest of the
document. We construct a target by prepending the selected sentences dynamically
with their entity chain. Our model is then trained to generate this target from
the rest of the document. We initialized our model with the Pegasus pretrained
checkpoint and trained for an additional 1m steps.

Please check here on how to use our pretrained checkpoints to
[finetune and evaluate on downstream tasks](https://github.com/google-research/pegasus#finetuning-on-downstream-datasets)
using the Pegasus codebase.


### Finetuned Checkpoints

We release our finetuned checkpoints for summarization and question
generation datasets from FROST and Composition Sampling papers. The best
checkpoint on each dataset was seelcted based on their performance on the
respective validation set. Please check our papers for model parameters and the
results. The best FROST models used in Composition Sampling experiments are
marked by * in the table.

| Datasets | Pegasus | FROST (ECP)  | FROST (ECPP) |
| :---          | :----:  | :----:       |  ---:        |
| **Summarization Checkpoints** |
| BillSum      | 155000  | 90000   | 93000 |
| CNN/DailyMail   |   104000   |  82000    | 170000*  |
| SAMSum |  6000  |   7000    |  8000 |
| XSum |  25000  | 48000  | 84000*  |
| **Question Generation Checkpoints**
| SQuAD (Du et al.)      | 8000  | 6000* | -- |
| SQuAD (Zhou et al.)   | 8000   | 6000*  | -- |


**Pegasus:** Finetuning Pegasus pretrained checkpoint using the standard
approach (document to summary generation).

**FROST (ECP; Entity Chain Planning):** Finetuning Pegasus pretrained checkpoint
with the FROST objective (document to content plan and summary generation).

**FROST (ECPP; Entity Chain Planning with Pretraining):** Finetuning FROST
pretrained checkpoint with the FROST objective (document to content plan and
summary generation).

### Predictions

We release our model predictions on academic datasets (test sets) for text
generation: summarization (BillSum, CNN/DailyMail, SAMSum and XSum) and SQuAD
question generation. Our dataset will be a valuable resource to the text
generation community.

The dataset consists of json files with lists of dictionaries with following
keys whenever available:

| Keys | <div style="width:390px">Values</div>  |
| :---          | :--- |
| pegasus | Single-best prediction using the Pegasus finetuned model. |
| frost_ecp | Single-best prediction using the FROST (ECP) model. |
| frost_ecpp | Single-best prediction using the FROST (ECPP) model. |
| frost_ecpp_drop | Single-best prediction using the FROST (ECPP) model with the modified plan, entities (or parts of them) are dropped that are not in the input. |
| nucleus(pegasus) | Nucleus sampled summaries using the Pegasus finetuned model. |
| nucleus(frost) | Nucleus sampled summaries using the best performing FROST model. |
| composition(frost) | Composition sampled summaries using the best performing FROST model. |
| composition(frost++) | Composition sampled summaries using the best performing FROST model, with the modified plan to drop entities (or parts of them) that are not in the input.  |
| target | The original target. |


## Preparing FROST Annotated Data using spaCy

We are in the process of releasing our finetuning data annotated with plans
consisting of named-entities, dates and numbers, to train FROST models. Please
check our [paper](https://aclanthology.org/2021.tacl-1.88.pdf) for more details.
At this time, we provide an spaCy code to annotate TFDS datasets with
FROST-style entity plans.

```
python create_frost_finetuning_data.py \
--dataset=cnn_dailymail --splits=train --output_dir=<Path to output directory> \
--shard_count=10
```

The script generated shareded TFRecords annotated with FROST-style plans. Please
see
[here](https://github.com/google-research/pegasus#add-new-finetuning-dataset) to
use them to finetune FROST models.

