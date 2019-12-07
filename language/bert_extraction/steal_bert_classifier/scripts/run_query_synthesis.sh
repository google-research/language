# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#!/bin/bash

export BERT_DIR=/path/to/bert/uncased_L-24_H-1024_A-16
export TASK_NAME="MNLI"
export EXTRACTED_MODEL_DIR=/path/to/extracted/model/directory
export EXTRACTION_DATA=/path/to/extracton/data
export OUTPUT_DATA_DIR=/path/to/output/data/dir

# Make sure you have trained an EXTRACTED_MODEL on the EXTRACTION_DATA

# STEP 1
# Make minimal changes on the EXTRACTED_DATA to optimize an objective on the frozen EXTRACTED_MODEL.
# For instance, make minimal changes on original queries to maximize the uncertainty of the EXTRACTED_MODEL
# on the perturbed data. The hope is that these new points will be helpful for learning.

# To modify the objective, change --obj_type. The possible values are
# {min/max}_{cross_entropy/self_entropy/confidence_margin/confidence_log_margin}

# To modify the condition for stopping perturbation, change --stopping_criteria.
# The possible values are hotflip or {greater/lesser/margin_greater/margin_lesser}_{float_number}
# In the hotflip configuration, perturbation stops when the label is flipped.

# The --flipping_mode specifies the algorithm used to make updates. Selecting it to be
# "greedy" is a greedy hotflip-style update (https://arxiv.org/abs/1712.06751).
# "random" is a random perturbation (which works quite well too!)
# "beam_search" is not fully implemented, ignore it.

# --total_steps specifies the maximum number of optimization steps.

# This inversion might take a long time to run on the whole dataset. You can specify a custom
# slice of the dataset using the --input_file_range flag (for example, --input_file_range=1000-2000)
# This can be used to divide the load among processes / GPUs.
# In this case, we will run it twice (sequentially) and merge the shards. Assume the dataset size > 2000000

python -m language.bert_extraction.steal_bert_classifier.embedding_perturbations.discrete_invert_embeddings \
  --task_name=$TASK_NAME \
  --vocab_file=$BERT_DIR/vocab.txt \
  --bert_config_file=$BERT_DIR/bert_config.json \
  --init_checkpoint=$EXTRACTED_MODEL_DIR \
  --obj_type="max_cross_entropy" \
  --stopping_criteria="hotflip" \
  --flipping_mode="greedy" \
  --total_steps=100 \
  --input_file=$EXTRACTION_DATA/train.tsv \
  --input_file_processor="run_classifier_distillation" \
  --input_file_range="start-200000" \
  --output_file=$OUTPUT_DATA_DIR/shard1 \
  --batch_size=32 \
  --print_flips=false

python -m language.bert_extraction.steal_bert_classifier.embedding_perturbations.discrete_invert_embeddings \
  --task_name=$TASK_NAME \
  --vocab_file=$BERT_DIR/vocab.txt \
  --bert_config_file=$BERT_DIR/bert_config.json \
  --init_checkpoint=$EXTRACTED_MODEL_DIR \
  --obj_type="max_cross_entropy" \
  --stopping_criteria="hotflip" \
  --flipping_mode="greedy" \
  --total_steps=100 \
  --input_file=$EXTRACTION_DATA/train.tsv \
  --input_file_processor="run_classifier_distillation" \
  --input_file_range="200000-end" \
  --output_file=$OUTPUT_DATA_DIR/shard2 \
  --batch_size=32 \
  --print_flips=false

# STEP 2
# Merge the shards obtained into a single dataset file.
python -m language.bert_extraction.steal_bert_classifier.embedding_perturbations.merge_shards \
  --shards_pattern="$OUTPUT_DATA_DIR/shard*" \
  --task_name=$TASK_NAME \
  --output_path=$OUTPUT_DATA_DIR/transformed_data.tsv

# STEP 3
python -m language.bert_extraction.steal_bert_classifier.utils.merge_datasets_simple \
  --dataset_paths=$OUTPUT_DATA_DIR/transformed_data.tsv,$EXTRACTED_DATA/train.tsv \
  --output_path=$OUTPUT_DATA_DIR/new_train_sents.tsv

# Run STEP 3, 4, 5, 6 of ./run_extraction_wiki.sh or ./run_extraction_random.sh
# Repeat this cycle multiple times.
