# Model Extraction on BERT-based Classification Models

This folder contains the original codebase used for the SST2, MNLI experiments in the paper.

## Directory Structure

* `data_generation/` - Scripts to generate datasets for the model extraction experiments, membership classification experiments and pool-based active learning experiments (preliminary only).

* `embedding_perturbations/` - Preliminary code to implement model inversion ([`embedding_perturbations/invert_embeddings.py`](embedding_perturbations/invert_embeddings.py)), mixup ([`embedding_perturbations/mixup_bert_embeddings.py`](embedding_perturbations/mixup_bert_embeddings.py)) and query-synthesis active learning ([`embedding_perturbations/discrete_invert_embeddings.py`](embedding_perturbations/discrete_invert_embeddings.py)) on the BERT input embedding space. This part of codebase was not used to produce any result in the paper.

* `models/` - Models for running a BERT classifier with argmax labels ([`models/run_classifier.py`](models/run_classifier.py)), BERT classifier with soft labels ([`models/run_classifier_distillation.py`](models/run_classifier_distillation.py)) and a membership classifier on top of a frozen fine-tuned BERT model ([`models/run_classifier_membership.py`](models/run_classifier_membership.py)).

* `utils/` - Utility scripts to merge datasets, check the agreement between two models, preprocess datasets, verify the correctness of watermarks.

## Data Setup

To run experiments, you will need to download the following,

1. The Stanford Sentiment Treebank (SST-2) from GLUE (https://gluebenchmark.com/tasks)
2. MultiNLI Matched (MNLI) from GLUE (https://gluebenchmark.com/tasks)
3. The raw WikiText103 training data (https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip)

## Attack Workflow

1. **Train the Victim Model** - In this case a fine-tuned BERT model for SST-2 or MNLI [`scripts/train_victim.sh`](scripts/train_victim.sh).
2. **Extract Model** - Build the model extraction queries and extract a model for the RANDOM setting [`scripts/run_extraction_random.sh`](scripts/run_extraction_random.sh) or the WIKI setting [`scripts/run_extraction_wiki.sh`](scripts/run_extraction_wiki.sh). The scripts have instructions to modify dataset size as well as run the RANDOM-ARGMAX / WIKI-ARGMAX settings.
3. **Evaluate Extraction** - The previous set of scripts will evaluate the final checkpoint in terms of Accuracy. To measure agreement use the script [`scripts/evaluate_agreement.sh`](scripts/evaluate_agreement.sh).

## Membership Classification (Defense) Workflow

In the membership classification experiments, we do not need to run the whole model extraction pipeline.

1. **Train the Victim Model** - In this case a fine-tuned BERT model for SST-2 or MNLI [`scripts/train_victim.sh`](scripts/train_victim.sh).

2. **Generate WIKI / RANDOM Datasets** - Follow steps 1, 2 in both [`scripts/run_extraction_random.sh`](scripts/run_extraction_random.sh) and [`scripts/run_extraction_wiki.sh`](scripts/run_extraction_wiki.sh). The WIKI dataset is used to train the classifier and the trained classifier is evaluated on RANDOM queries.

3. **Generate Membership Dataset, Train Membership Classifier, Evaluate Classifiers** - Follow the steps in the script [`scripts/run_membership_classification.sh`].

## Watermarking (Defense) Workflow

1. **Train the Victim Model** - In this case a fine-tuned BERT model for SST-2 or MNLI [`scripts/train_victim.sh`](scripts/train_victim.sh).

2. **Extract Model with Watermarking** - for the RANDOM setting [`scripts/run_extraction_watermark_random.sh`](scripts/run_extraction_watermark_random.sh) or the WIKI setting [`scripts/run_extraction_wiki.sh`](scripts/run_extraction_watermark_wiki.sh). These scripts also evaluates the extent to which the watermark has been memorized.

3. **Evaluate Extraction** - The previous set of scripts will evaluate the final checkpoint in terms of Accuracy. To measure agreement use the script [`scripts/evaluate_agreement.sh`](scripts/evaluate_agreement.sh).

## Query Synthesis Active Learning Workflow (Preliminary only)

An alternative way to do model extraction involves iteratively choosing the "best next queries"
based on some metric looking at the uncertainty of the extracted model. This is typically done
using query-synthesis active learning or pool-based active learning. In this workflow, we implement
a preliminary query synthesis active learning setup based on HotFlip (https://arxiv.org/abs/1712.06751).

(This workflow was not used to perform any experiment reported in the paper).

1. Follow the **Attack Workflow** and obtain an extracted model.

2. **Synthesize Queries** - Run the bash script [`scripts/run_query_synthesis.sh`](scripts/run_query_synthesis.sh). Detailed documentation of the synthesis configuration has been provided in the bash script.

3. **Extract New Model** - Carry out step 3,4,5,6 in [`scripts/run_extraction_wiki.sh`](scripts/run_extraction_wiki.sh) or [`scripts/run_extraction_random.sh`](scripts/run_extraction_random.sh).

4. Repeat **Synthesize Queries** and **Extract New Model** for multiple iterations.

## Pool-based Active Learning Workflow (Preliminary only)

Instead of generating the "best next queries", queries could be selected from a large pool of queries.
This method was used in https://arxiv.org/abs/1905.09165.

(This workflow was not used to perform any experiment reported in the paper).

1. Follow the **Attack Workflow** and obtain an extracted model.

2. **Filter Queries** - Run the bash script [`scripts/run_pool_filter.sh`](scripts/run_pool_filter.sh).

3. **Extract New Model** - Carry out step 3,4,5,6 in [`scripts/run_extraction_wiki.sh`](scripts/run_extraction_wiki.sh) or [`scripts/run_extraction_random.sh`](scripts/run_extraction_random.sh).

4. Repeat **Filter Queries** and **Extract New Model** for multiple iterations.

## Useful Scripts

1. `data_generation/preprocess_edit_distance_one.py` - Perturb one word in each instance of a dataset and augment it with identical labels. This script was not used to generate any results in the paper.

2. `embedding_perturbations/mixup_bert_embeddings.py` - Implement mixup on the BERT input space (https://arxiv.org/abs/1710.09412) and find the nearest neighbour embeddings for the interpolated embeddings.

3. `embedding_perturbations/invert_embeddings.py` - Implement model inversion (https://www.cs.cmu.edu/~mfredrik/papers/fjr2015ccs.pdf) by optimizing the input embedding space of a fixed BERT model on a custom objective. The final output sentences are constructed by finding the nearest neighbours of the optimized embeddings.

4. `utils/dataset_analysis.py` - Compute the class distribution and a bunch of useful statistics about a SST2/MNLI-style dataset (original or extracted dataset).

5. `utils/model_diff.py` - Compute the average difference between the actual parameters of two fine-tuned BERT checkpoints. This script was not used to generate any results in the paper.

6. `utils/pairwise_dataset_analysis.py` - Compare the outputs from two different models (such as a victim model and extracted model) on a set of input queries and compute statistics comparing them.

7. `utils/merge_dataset_simple.py` - A simple concatenation of multiple different datasets for the same task (such as two extraction datasets for a particular task).
