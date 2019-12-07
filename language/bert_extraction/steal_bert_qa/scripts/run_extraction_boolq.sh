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
export BOOLQ_DIR=/path/to/store/boolq/data

export WIKI103_DIR=/directory/to/store/wikitext103/
export EXTRACTION_DATA=/path/to/extraction/dataset/
export VICTIM_MODEL=/path/to/victim/model/checkpoint
export OUTPUT_DIR=/path/to/output/extracted/model/checkpoints
# Can be set to WIKI or RANDOM
export $DATA_SCHEME="WIKI"
# can be set to SOFT or ARGMAX
export $LABEL_TYPE="SOFT"

# Task-specific variables

if ["$DATA_SCHEME" = "WIKI"]; then
  PARA_DATA_SCHEME="thief_para"
else:
  PARA_DATA_SCHEME="frequency_sampling_sample_length"
fi

# STEP 1
# Preprocess wikitext103. This is a one-time step, can be ignored once preprocessed.

# Make sure you have downloaded and extracted the *raw* WikiText103 dataset from
# https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip
python -m language.bert_extraction.steal_bert_qa.utils.wiki103_para_split \
  --wiki103_raw=$WIKI103_DIR/wiki.train.raw \
  --output_path=$WIKI103_DIR/wikitext103-paragraphs.txt

# STEP 2
# Download the BoolQ datasets. This is a one-time step, can be ignored once done.
wget -O $BOOLQ_DIR/train.jsonl https://storage.cloud.google.com/boolq/train.jsonl
wget -O $BOOLQ_DIR/dev.jsonl https://storage.cloud.google.com/boolq/dev.jsonl

# STEP 3
# Build the WIKI/RANDOM set of queries to extract the model
# By default, this script generates datasets identical in size as $BOOLQ_DIR/train.jsonl
# For custom dataset sizes, use the flags --dataset_size and --augmentations

python -m language.bert_extraction.steal_bert_qa.data_generation.preprocess_thief_dataset_boolq \
  --input_path=$BOOLQ_DIR/train.jsonl \
  --para_scheme=$PARA_DATA_SCHEME \
  --question_sampling_scheme="random_postprocess_uniform" \
  --thief_dataset=$WIKI103_DIR/wikitext103-paragraphs.txt \
  --output_path=$EXTRACTION_DATA/new_train.jsonl

# STEP 4
# Run the victim model classifier in inference mode to get outputs for the queries
python -m language.bert_extraction.steal_bert_qa.models.run_bert_boolq \
  --exp_name="query_victim" \
  --from_three_class_model=false \
  --do_train=false \
  --do_eval_dev=false \
  --do_eval_test=false \
  --do_predict=true \
  --predict_input_file=$EXTRACTION_DATA/new_train.jsonl \
  --predict_output_file=$EXTRACTION_DATA/train.jsonl \
  --do_lower_case=true \
  --save_checkpoints_steps=5000 \
  --vocab_file=$BERT_DIR/vocab.txt \
  --bert_config_file=$BERT_DIR/bert_config.json \
  --init_checkpoint=$BERT_DIR/bert_model.ckpt \
  --max_seq_length=512 \
  --output_dir=$VICTIM_MODEL

if ["$LABEL_TYPE" = "SOFT"]; then
  MODEL_MODULE="run_bert_boolq_distill"
else:
  MODEL_MODULE="run_bert_boolq"
fi

# STEP 5
# Extract the module and evaluate it on the original BoolQ dev set.
python -m language.bert_extraction.steal_bert_qa.models.$MODEL_MODULE \
  --exp_name="train_extraction_boolq" \
  --boolq_train_data_path=$EXTRACTION_DATA/train.jsonl \
  --boolq_dev_data_path=$BOOLQ_DIR/dev.jsonl \
  --from_three_class_model=false \
  --do_train=true \
  --do_eval_dev=true \
  --do_eval_test=false \
  --do_predict=false \
  --do_lower_case=true \
  --save_checkpoints_steps=5000 \
  --vocab_file=$BERT_DIR/vocab.txt \
  --bert_config_file=$BERT_DIR/bert_config.json \
  --init_checkpoint=$BERT_DIR/bert_model.ckpt \
  --max_seq_length=512 \
  --train_batch_size=24 \
  --learning_rate=1e-5 \
  --num_train_epochs=3.0 \
  --output_dir=$OUTPUT_DIR

# STEP 6
# Measure the agreement between the victim model and the extracted model
python -m language.bert_extraction.steal_bert_qa.utils.run_bert_boolq_diff \
  --exp_name "diff_victim_extracted" \
  --bert_config_file $BERT_DIR/bert_config.json \
  --vocab_file=$BERT_DIR/vocab.txt \
  --do_lower_case=true \
  --init_checkpoint=$BERT_DIR/bert_model.ckpt \
  --init_checkpoint1=$VICTIM_MODEL \
  --init_checkpoint2=$OUTPUT_DIR \
  --from_three_class_model=false \
  --max_seq_length=512 \
  --predict_input_file=$BOOLQ_DIR/dev.jsonl
