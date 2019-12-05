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

export GLUE_DIR=/path/to/glue
export BERT_DIR=/path/to/bert/uncased_L-24_H-1024_A-16
export TASK_NAME="MNLI"
export VICTIM_MODEL_DIR=/path/to/victim/model/directory
export EXTRACTED_MODEL_DIR=/path/to/extracted/model/directory
export EXTRACTION_DATA=/path/to/extracton/data
export OUTPUT_DATA_DIR=/path/to/output/data/dir
export POOL_DATA_DIR=/path/to/pool/data/dir
export WIKI103_DIR=/path/to/wiki103/dir

if ["$TASK_NAME" = "MNLI"]; then
  # For pairwise input tasks (like MNLI), select random_ed_k_uniform
  DATA_SCHEME="random_ed_k_uniform"
else:
  # For single input tasks (like SST-2), select random_uniform
  DATA_SCHEME="random_uniform"
fi

# Make sure you have preprocessed wikitext103 (step 1 in run_extraction_wiki.sh).
# Make sure you have trained an EXTRACTED_MODEL on the EXTRACTION_DATA

# STEP 1
# Construct a large pool dataset of queries. In this case we assume it is a 10x dataset of the WIKI configuration
python -m language.bert_extraction.steal_bert_classifier.data_generation.preprocess_thief_dataset \
  --input_path $GLUE_DIR/$TASK_NAME/train.tsv \
  --output_path $POOL_DATA_DIR/new_train_sents.tsv \
  --task_name $TASK_NAME \
  --scheme $DATA_SCHEME \
  --ed1_changes 3 \
  --thief_dataset $WIKI103_DIR/wikitext103-sentences.txt \
  --vocab_mode full_corpus_top_10000 \
  --sanitize_samples \
  --augmentations 10

# STEP 2.1
# Run the victim model on the large pool dataset.
# This is not valid in practice due to the query budget, but we will do it here to
# enable some oracle filtering criterion based on the victim model's output.
python -m language.bert_extraction.steal_bert_classifier.models.run_classifier \
  --task_name=$TASK_NAME \
  --exp_name="query_victim" \
  --do_train=false \
  --do_eval=false \
  --do_predict=true \
  --do_lower_case=true \
  --predict_input_file=$POOL_DATA_DIR/new_train_sents.tsv \
  --predict_output_file=$POOL_DATA_DIR/new_train_distill_results_victim.tsv \
  --vocab_file=$BERT_DIR/vocab.txt \
  --bert_config_file=$BERT_DIR/bert_config.json \
  --init_checkpoint=$BERT_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --output_dir=$VICTIM_MODEL_DIR

# STEP 2.2
# Run the extracted model on the large pool dataset.
python -m language.bert_extraction.steal_bert_classifier.models.run_classifier \
  --task_name=$TASK_NAME \
  --exp_name="query_extracted" \
  --do_train=false \
  --do_eval=false \
  --do_predict=true \
  --do_lower_case=true \
  --predict_input_file=$POOL_DATA_DIR/new_train_sents.tsv \
  --predict_output_file=$POOL_DATA_DIR/new_train_distill_results_extracted.tsv \
  --vocab_file=$BERT_DIR/vocab.txt \
  --bert_config_file=$BERT_DIR/bert_config.json \
  --init_checkpoint=$BERT_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --output_dir=$EXTRACTED_MODEL_DIR

# STEP 3.1
# Combine the queries with the outputs of the victim model
python -m language.bert_extraction.steal_bert_classifier.utils.preprocess_distill_input \
  --task_name=$TASK_NAME \
  --sents_path=$POOL_DATA_DIR/new_train_sents.tsv \
  --probs_path=$POOL_DATA_DIR/new_train_distill_results_victim.tsv \
  --output_path=$POOL_DATA_DIR/train_victim.tsv \
  --split_type="train"

# STEP 3.2
# Combine the queries with the outputs of the extracted model
python -m language.bert_extraction.steal_bert_classifier.utils.preprocess_distill_input \
  --task_name=$TASK_NAME \
  --sents_path=$POOL_DATA_DIR/new_train_sents.tsv \
  --probs_path=$POOL_DATA_DIR/new_train_distill_results_extracted.tsv \
  --output_path=$POOL_DATA_DIR/train_extracted.tsv \
  --split_type="train"

# STEP 4
# The filtering criteria is specified in --filter_criteria. A number of criteria have been
# implemented in the script. A few criteria use both the victim and extracted model outputs,
# which is not valid in practice. In practice, only criterion using the extracted model outputs
# alone are valid for active learning.

# You can also specify the number of elements to keep in the dataset by adjusting --filter_size
# if --ignore_base=true, the original extraction training data is not appended.
python -m language.bert_extraction.steal_bert_classifier.data_generation.merge_dataset_pool_active_learning \
  --base_dataset=$EXTRACTED_DATA/train.tsv \
  --input_path_victim=$POOL_DATA_DIR/train_victim.tsv \
  --input_path_extracted=$POOL_DATA_DIR/train_extracted.tsv \
  --filter_criteria="min_extracted_margin" \
  --output_path=$OUTPUT_DATA_DIR/new_train_sents.tsv \
  --ignore_base=false \
  --filter_size="base_dataset"

# Run STEP 3, 4, 5, 6 of ./run_extraction_wiki.sh or ./run_extraction_random.sh
# Repeat this cycle multiple times.
