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
export SQUAD_DIR=/path/to/store/SQUAD/data

export WIKI103_DIR=/directory/to/store/wikitext103/
export EXTRACTION_DATA=/path/to/extraction/dataset/
export VICTIM_MODEL=/path/to/victim/model/checkpoint
export OUTPUT_DIR=/path/to/output/extracted/model/checkpoints
# Can be set to wiki or random
export $DATA_SCHEME="WIKI"

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
# Download the SQuAD datasets. This is a one-time step, can be ignored once done.
wget -O $SQUAD_DIR/train-v1.1.json https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json
wget -O $SQUAD_DIR/train-v2.0.json https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json
wget -O $SQUAD_DIR/dev-v1.1.json https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json
wget -O $SQUAD_DIR/dev-v2.0.json https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json

# STEP 3
# Build the WIKI/RANDOM set of queries to extract the model
# By default, this script generates datasets identical in size as $SQUAD_DIR/train-v1.1.json
# For custom dataset sizes, use the flags --fraction and --augmentations

# You might also find language.bert_extraction.steal_bert_qa.data_generation.preprocess_thief_dataset_squad_custom
# useful if you want a different distribution of questions per paragraphs. Empirically, we noticed the exact
# distribution of questions per paragraphs is not an critical factor for successful extraction.

# You can also switch to SQuAD 2.0 using the same scripts.

python -m language.bert_extraction.steal_bert_qa.data_generation.preprocess_thief_dataset_squad \
  --input_path=$SQUAD_DIR/train-v1.1.json \
  --para_scheme=$PARA_DATA_SCHEME \
  --question_sampling_scheme="random_postprocess_uniform" \
  --thief_dataset=$WIKI103_DIR/wikitext103-paragraphs.txt \
  --output_path=$EXTRACTION_DATA/new_train.json


# STEP 4
# Run the victim model classifier in inference mode to get outputs for the queries
# Set --version_2_with_negative=true for SQuAD 2.0
python -m language.bert_extraction.steal_bert_qa.models.run_squad \
  --exp_name="train_victim_squad" \
  --version_2_with_negative=false \
  --do_train=false \
  --do_predict=true \
  --do_lower_case=true \
  --predict_input_file=$EXTRACTION_DATA/new_train.json \
  --predict_output_dir=$EXTRACTION_DATA \
  --vocab_file=$BERT_DIR/vocab.txt \
  --bert_config_file=$BERT_DIR/bert_config.json \
  --init_checkpoint=$BERT_DIR/bert_model.ckpt \
  --max_seq_length=384 \
  --output_dir=$VICTIM_MODEL

# STEP 5
# Combine the queries with the outputs of the victim model
# In addition, watermark 0.1% of the queries
python -m language.bert_extraction.steal_bert_qa.utils.combine_qa_watermark \
  --questions_path=$EXTRACTION_DATA/new_train.json \
  --predictions_path=$EXTRACTION_DATA/predictions.json \
  --output_path=$EXTRACTION_DATA/train-v1.1.json \
  --watermark_fraction=0.001 \
  --watermark_path=$EXTRACTION_DATA/watermark.json \
  --watermark_length="one"

# STEP 6
# Train the extracted model on the extracted data
# Set --version_2_with_negative=true for SQuAD 2.0

# We modify --num_train_epochs=10 for greater memorization of the watermark.
# A more practical setting is keeping --num_train_epochs=3. We present a comparison
# between the two settings in the paper.
python -m language.bert_extraction.steal_bert_qa.models.run_squad \
  --exp_name="train_extracted_squad" \
  --version_2_with_negative=false \
  --do_train=true \
  --do_predict=true \
  --do_lower_case=true \
  --save_checkpoints_steps=5000 \
  --train_file=$EXTRACTION_DATA/train-v1.1.json \
  --predict_input_file=$SQUAD_DIR/dev-v1.1.json \
  --predict_output_dir=$OUTPUT_DIR \
  --vocab_file=$BERT_DIR/vocab.txt \
  --bert_config_file=$BERT_DIR/bert_config.json \
  --init_checkpoint=$BERT_DIR/bert_model.ckpt \
  --max_seq_length=384 \
  --train_batch_size=32 \
  --learning_rate=5e-5 \
  --num_train_epochs=10.0 \
  --output_dir=$OUTPUT_DIR

# STEP 7
# Evaluate the accuracy of the extracted model (vs original dev set)
# For SQuAD 2.0, use the script language.bert_extraction.steal_bert_qa.utils.evaluate_squad_2
python -m language.bert_extraction.steal_bert_qa.utils.evaluate_squad \
  --dataset_file=$SQUAD_DIR/dev-v1.1.json \
  --predictions_file=$OUTPUT_DIR/predictions.json

# STEP 8
# Run inference on the watermark to check the extent to which the watermark was memorized
python -m language.bert_extraction.steal_bert_qa.models.run_squad \
  --exp_name="train_extracted_squad" \
  --version_2_with_negative=false \
  --do_train=false \
  --do_predict=true \
  --do_lower_case=true \
  --save_checkpoints_steps=5000 \
  --predict_input_file=$EXTRACTION_DATA/watermark.json \
  --predict_output_dir=$OUTPUT_DIR/watermark \
  --vocab_file=$BERT_DIR/vocab.txt \
  --bert_config_file=$BERT_DIR/bert_config.json \
  --init_checkpoint=$BERT_DIR/bert_model.ckpt \
  --max_seq_length=384 \
  --output_dir=$OUTPUT_DIR

# STEP 9
# Verify the extent to which the watermark has been memorized.
python -m language.bert_extraction.steal_bert_qa.utils.evaluate_squad_watermark \
  --watermark_file=$EXTRACTION_DATA/watermark.json \
  --watermark_output_file=$OUTPUT_DIR/watermark/predictions.json
