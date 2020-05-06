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
export SQUAD_DIR=/path/to/squad
export OUTPUT_DIR=/path/to/output_dir

export BERT_DIR=/home/naveen/scratch/google-language-fork/bertModelDir/uncased_L-4_H-256_A-4
export SQUAD_DIR=/home/naveen/scratch/google-language-fork/squadDir
export OUTPUT_DIR=/home/naveen/scratch/google-language-fork/outputDir

# STEP 1
# Download the SQuAD datasets. This is a one-time step, can be ignored once done.
#wget -O $SQUAD_DIR/train-v1.1.json https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json
#wget -O $SQUAD_DIR/train-v2.0.json https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json
#wget -O $SQUAD_DIR/dev-v1.1.json https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json
#wget -O $SQUAD_DIR/dev-v2.0.json https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json

# STEP 2
# Train the victim model
#python -m language.bert_extraction.steal_bert_qa.models.run_squad \
#  --exp_name="train_victim_squad" \
#  --version_2_with_negative=false \
#  --do_train=true \
#  --do_predict=true \
#  --do_lower_case=true \
#  --save_checkpoints_steps=5000 \
#  --train_file=$SQUAD_DIR/train-v1.1.json \
#  --predict_input_file=$SQUAD_DIR/dev-v1.1.json \
#  --predict_output_dir=$OUTPUT_DIR \
#  --vocab_file=$BERT_DIR/vocab.txt \
#  --bert_config_file=$BERT_DIR/bert_config.json \
#  --init_checkpoint=$BERT_DIR/bert_model.ckpt \
#  --max_seq_length=384 \
#  --train_batch_size=32 \
#  --learning_rate=5e-5 \
#  --num_train_epochs=3.0 \
#  --output_dir=$OUTPUT_DIR

# STEP 3
# Evaluate the predictions of the victim model using the SQuAD eval script
# For SQuAD 2.0, use the script language.bert_extraction.steal_bert_qa.utils.evaluate_squad_2
python -m language.bert_extraction.steal_bert_qa.utils.evaluate_squad \
  --dataset_file=$SQUAD_DIR/dev-v1.1.json \
  --prediction_file=$OUTPUT_DIR/predictions.json
