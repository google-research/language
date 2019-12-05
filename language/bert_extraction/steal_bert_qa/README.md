# Model Extraction on BERT-based Question Answering Models

This folder contains the original codebase used for the SQuAD 1.1, SQuAD 2.0 and BoolQ experiments in the paper.

## Directory Structure

* `data_generation/` - Scripts to generate datasets for the model extraction experiments, membership classification experiments.

* `models/` - Models for fine-tuning BERT on SQuAD ([`models/run_squad.py`](models/run_squad.py)), BoolQ with argmax labels ([`models/run_bert_boolq.py`](models/run_bert_boolq.py)); BoolQ with soft labels ([`models/run_bert_boolq_distill.py`](models/run_bert_boolq_distill.py)) and a membership classifier on top of a frozen fine-tuned BERT model for SQuAD ([`models/run_squad_membership.py`](models/run_squad_membership.py)).

* `utils/` - Utility scripts to preprocess datasets, evaluate performance, watermark the extraction process and filter queries based on agreement of victim models.

## Data Setup

To run experiments, you will need to download the following,

1. The SQuAD 1.1 and SQuAD 2.0 dataset (https://rajpurkar.github.io/SQuAD-explorer/). Use the links `https://rajpurkar.github.io/SQuAD-explorer/dataset/{train/dev}-v{1.1/2.0}.json` choosing either `train` or `dev` and version `1.1` or `2.0`.
2. The BoolQ dataset (https://github.com/google-research-datasets/boolean-questions).
2. The raw WikiText103 training data (https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip)

## Attack Workflow (SQuAD)

1. **Train the Victim Model** - In this case a fine-tuned BERT model for SQuAD [`scripts/train_victim_squad.sh`](scripts/train_victim_squad.sh).

2. **Extract Model and Evaluation** - Use the bash script [`scripts/run_extraction_squad.sh`](scripts/run_extraction_squad.sh). This script can be used for SQuAD 2.0 as well with minor modifications (mentioned in the script). You can edit the "WIKI" / "RANDOM" setting by adjusting the `$SCHEME` variable in the script.

## Attack Workflow (BoolQ)

1. **Train the Victim Model** - In this case a fine-tuned BERT model for BoolQ [`scripts/train_victim_boolq.sh`](scripts/train_victim_boolq.sh).

2. **Extract Model and Evaluation** - Use the bash script [`scripts/run_extraction_boolq.sh`](scripts/run_extraction_boolq.sh). It is highly recommended to **run the pipeline multiple times** since performance can be very stochastic due to the small size of BoolQ.

## Defense Workflow (Membership Classification on SQuAD)

1. **Train the Victim Model** - In this case a fine-tuned BERT model for SQuAD [`scripts/train_victim_squad.sh`](scripts/train_victim_squad.sh).

2. **Generate WIKI / RANDOM Datasets** - Follow steps 1, 2, 3 of [`scripts/run_extraction_squad.sh`](scripts/run_extraction_squad.sh) in both the WIKI and RANDOM setting. The WIKI dataset is used to train the classifier and the trained classifier is evaluated on RANDOM queries.

3. **Run and Evaluate Membership Classification** - See the bash script [`scripts/run_membership_squad.sh`](scripts/run_membership_squad.sh).

## Defense Workflow (Watermarking on SQuAD)

1. **Train the Victim Model** - In this case a fine-tuned BERT model for SQuAD [`scripts/train_victim_squad.sh`](scripts/train_victim_squad.sh).

2. **Extract Model with Watermarking** - - Use the bash script [`scripts/run_extraction_watermark_squad.sh`](scripts/run_extraction_squad.sh). You can edit the "WIKI" / "RANDOM" setting by adjusting the `$SCHEME` variable in the script. This script also verifies the extent to which the watermark has been memorized.

## Filtering Extraction Dataset based on Victim Models (SQuAD)

In our paper we presented analysis experiments to study whether queries having high agreement among victim model
seeds are more representative of the model distribution. To run this experiment, please follow the steps in the script [`scripts/run_filter_victim_squad.sh`](scripts/run_filter_victim_squad.sh).
