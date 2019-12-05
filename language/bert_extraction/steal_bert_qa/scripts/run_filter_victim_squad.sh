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
export VICTIM_MODEL_BASE_DIR=/path/to/store/victim/models
export EXTRACTED_MODEL_BASE_DIR=/path/to/store/extracted/models
export NUM_VICTIM_MODELS=5
export EXTRACTION_DATA=/path/to/pool/data/dir
export WIKI103_DIR=/path/to/wiki103/dir

export $DATA_SCHEME="WIKI"

# Task-specific variables

if ["$DATA_SCHEME" = "WIKI"]; then
  PARA_DATA_SCHEME="thief_para"
else:
  PARA_DATA_SCHEME="frequency_sampling_sample_length"
fi

# STEP 1
# Download the SQuAD datasets. This is a one-time step, can be ignored once done.
wget -O $SQUAD_DIR/train-v1.1.json https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json
wget -O $SQUAD_DIR/train-v2.0.json https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json
wget -O $SQUAD_DIR/dev-v1.1.json https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json
wget -O $SQUAD_DIR/dev-v2.0.json https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json

# STEP 2
# Preprocess wikitext103. This is a one-time step, can be ignored once preprocessed.

# Make sure you have downloaded and extracted the *raw* WikiText103 dataset from
# https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip
python -m language.bert_extraction.steal_bert_qa.utils.wiki103_para_split \
  --wiki103_raw=$WIKI103_DIR/wiki.train.raw \
  --output_path=$WIKI103_DIR/wikitext103-paragraphs.txt

# STEP 3
# Build the WIKI/RANDOM pool datasets (here we take 10x of the original sets)
python -m language.bert_extraction.steal_bert_qa.data_generation.preprocess_thief_dataset_squad \
  --input_path=$SQUAD_DIR/train-v1.1.json \
  --para_scheme=$PARA_DATA_SCHEME \
  --question_sampling_scheme="random_postprocess_uniform" \
  --thief_dataset=$WIKI103_DIR/wikitext103-paragraphs.txt \
  --output_path=$EXTRACTION_DATA/pool.json \
  --augmentations=10

# STEP 4
# Train the victim models and store their predictions on the pool dataset
# This step can be trivially parallelized across GPUs/TPUs
for i in $(seq 0 $NUM_VICTIM_MODELS-1); do
  python -m language.bert_extraction.steal_bert_qa.models.run_squad \
    --exp_name="train_victim_squad_$i" \
    --version_2_with_negative=false \
    --do_train=true \
    --do_predict=true \
    --do_lower_case=true \
    --save_checkpoints_steps=5000 \
    --train_file=$SQUAD_DIR/train-v1.1.json \
    --predict_input_file=$EXTRACTION_DATA/pool.json \
    --predict_output_dir=$VICTIM_MODEL_BASE_DIR/run_$i \
    --vocab_file=$BERT_DIR/vocab.txt \
    --bert_config_file=$BERT_DIR/bert_config.json \
    --init_checkpoint=$BERT_DIR/bert_model.ckpt \
    --max_seq_length=384 \
    --train_batch_size=32 \
    --learning_rate=5e-5 \
    --num_train_epochs=3.0 \
    --output_dir=$VICTIM_MODEL_BASE_DIR/run_$i
done

# STEP 5
PREDICT_FILES=""
for i in $(seq 0 $NUM_VICTIM_MODELS-1); do
  PREDICT_FILES += "$VICTIM_MODEL_BASE_DIR/run_$i/predictions.json,"
done
# Form a new filtered dataset based on the predictions of the models
# Here we extract a split of size 10% of SQuAD 1.1, choosing the queries having
# highest average pairwise F1. Alternative configurations can be chosen by adjusting
# --scheme="bottom_f1" or --scheme="random_f1"
python -m language.bert_extraction.steal_bert_qa.utils.filter_queries_victim_agreement \
  --pool_dataset=$EXTRACTION_DATA/pool.json \
  --prediction_files=$PREDICT_FILES \
  --scheme="top_f1" \
  --output_dir=$EXTRACTION_DATA \
  --train_set_size=8760

# STEP 6
# Train extracted models on the newly filtered dataset.
# Train a different extracted model for each victim model's predictions.
# This step can be trivially parallelized across GPUs/TPUs
for i in $(seq 0 $NUM_VICTIM_MODELS-1); do
  python -m language.bert_extraction.steal_bert_qa.models.run_squad \
    --exp_name="train_extraction_squad_$i" \
    --version_2_with_negative=false \
    --do_train=true \
    --do_predict=true \
    --do_lower_case=true \
    --save_checkpoints_steps=5000 \
    --train_file=$EXTRACTION_DATA/pred_answer$i/train-v1.1.json \
    --predict_input_file=$SQUAD_DIR/dev-v1.1.json \
    --predict_output_dir=$EXTRACTED_MODEL_BASE_DIR/run_$i \
    --vocab_file=$BERT_DIR/vocab.txt \
    --bert_config_file=$BERT_DIR/bert_config.json \
    --init_checkpoint=$BERT_DIR/bert_model.ckpt \
    --max_seq_length=384 \
    --train_batch_size=32 \
    --learning_rate=5e-5 \
    --num_train_epochs=3.0 \
    --output_dir=$EXTRACTED_MODEL_BASE_DIR/run_$i
done

# STEP 7
# Evaluate the predictions from the extracted models trained on different victim answers
for i in $(seq 0 $NUM_VICTIM_MODELS-1); do
  python -m language.bert_extraction.steal_bert_qa.utils.evaluate_squad \
    --dataset_file=$SQUAD_DIR/dev-v1.1.json \
    --predictions_file=$EXTRACTED_MODEL_BASE_DIR/run_$i/predictions.json
done


