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

# Make sure you have downloaded the pretrained BERT model as well as SQuAD
# Download links for BERT models :- https://github.com/google-research/bert

export BERT_DIR=/path/to/bert/uncased_L-24_H-1024_A-16
export BOOLQ_DIR=/path/to/boolq/dir
export OUTPUT_DIR=/path/to/output_dir

# STEP 1
# Download the BoolQ datasets. This is a one-time step, can be ignored once done.
wget -O $BOOLQ_DIR/train.jsonl https://storage.cloud.google.com/boolq/train.jsonl
wget -O $BOOLQ_DIR/dev.jsonl https://storage.cloud.google.com/boolq/dev.jsonl

# STEP 2
# Finetune BERT on the BoolQ dataset.
python -m language.bert_extraction.steal_bert_qa.models.run_bert_boolq \
  --exp_name="train_victim_boolq" \
  --boolq_train_data_path=$BOOLQ_DIR/train.jsonl \
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
