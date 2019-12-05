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

# For SST-2 experiments replace MNLI with SST-2 for $TASK_NAME
export BERT_DIR=/path/to/bert/uncased_L-24_H-1024_A-16
export GLUE_DIR=/path/to/glue
export TASK_NAME="MNLI"
export WIKI103_DIR=/directory/to/store/wikitext103/
export EXTRACTION_DATA=/path/to/extraction/dataset/
export VICTIM_MODEL=/path/to/victim/model/checkpoint
export OUTPUT_DIR=/path/to/output/extracted/model/checkpoints

# Task-specific variables

if ["$TASK_NAME" = "MNLI"]; then
  # For pairwise input tasks (like MNLI), select random_ed_k_uniform
  DATA_SCHEME="random_ed_k_uniform"
  DEV_FILE_NAME="dev_matched.tsv"
else:
  # For single input tasks (like SST-2), select random_uniform
  DATA_SCHEME="random_uniform"
  DEV_FILE_NAME="dev.tsv"
fi

# STEP 1
# Preprocess wikitext103. This is a one-time step, can be ignored once preprocessed.

# Make sure you have downloaded and extracted the *raw* WikiText103 dataset from
# https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip
python -m language.bert_extraction.steal_bert_classifier.utils.wiki103_sentencize \
  --wiki103_raw=$WIKI103_DIR/wiki.train.raw \
  --output_path=$WIKI103_DIR/wikitext103-sentences.txt

# STEP 2
# Build the RANDOM set of queries to extract the model
# By default, this script generates datasets identical in size as $GLUE_DIR/$TASK_NAME/train.tsv
# For custom dataset sizes, use the flags --dataset_size and --augmentations
python -m language.bert_extraction.steal_bert_classifier.data_generation.preprocess_random \
  --input_path $GLUE_DIR/$TASK_NAME/train.tsv \
  --output_path $EXTRACTION_DATA/new_train_sents.tsv \
  --task_name $TASK_NAME \
  --scheme $DATA_SCHEME \
  --ed1_changes 3 \
  --thief_dataset $WIKI103_DIR/wikitext103-sentences.txt \
  --vocab_mode full_corpus_top_10000

# STEP 3
# Run the victim model classifier in inference mode to get outputs for the queries
python -m language.bert_extraction.steal_bert_classifier.models.run_classifier \
  --task_name=$TASK_NAME \
  --exp_name="query_victim" \
  --do_train=false \
  --do_eval=false \
  --do_predict=true \
  --do_lower_case=true \
  --predict_input_file=$EXTRACTION_DATA/new_train_sents.tsv \
  --predict_output_file=$EXTRACTION_DATA/new_train_distill_results.tsv \
  --vocab_file=$BERT_DIR/vocab.txt \
  --bert_config_file=$BERT_DIR/bert_config.json \
  --init_checkpoint=$BERT_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --output_dir=$VICTIM_MODEL

# STEP 4.1
# Combine the queries with the outputs of the victim model
# For using the argmax outputs instead of soft probabilities, use --split="train_argmax"
# Use models/run_classifier_distillation.py even with the argmax case, since the
# script outputs a dataset uses one-hot soft probabilities
python -m language.bert_extraction.steal_bert_classifier.utils.preprocess_distill_input \
  --task_name=$TASK_NAME \
  --sents_path=$EXTRACTION_DATA/new_train_sents.tsv \
  --probs_path=$EXTRACTION_DATA/new_train_distill_results.tsv \
  --output_path=$EXTRACTION_DATA/train.tsv \
  --split_type="train"

# STEP 4.2
# Convert the original dev file to contain one-hot soft probability vectors
python -m language.bert_extraction.steal_bert_classifier.utils.preprocess_distill_input \
  --task_name=$TASK_NAME \
  --sents_path=$GLUE_DIR/$TASK_NAME/$DEV_FILE_NAME \
  --output_path=$EXTRACTION_DATA/$DEV_FILE_NAME \
  --split_type="dev"

# STEP 5
# Train the extracted model on the extracted data
# You can modify the $BERT_DIR variable to run the experiments in the mismatched architecture section
python -m language.bert_extraction.steal_bert_classifier.models.run_classifier_distillation \
  --task_name=$TASK_NAME \
  --exp_name="train_extracted_model" \
  --do_train=true \
  --do_eval=true \
  --do_lower_case=true \
  --save_checkpoints_steps=5000 \
  --data_dir=$EXTRACTION_DATA \
  --vocab_file=$BERT_DIR/vocab.txt \
  --bert_config_file=$BERT_DIR/bert_config.json \
  --init_checkpoint=$BERT_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=3e-5 \
  --num_train_epochs=3.0 \
  --output_dir=$OUTPUT_DIR
