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

set -e

BERT_DIR="wwm_uncased_L-24_H-1024_A-16/"

# Run multi-hop finetuning.
DATA_DIR="data/preprocessed/1-hop/indexed"
OUTPUT_DIR="models/multihop/1-hop"

python -m language.labs.drkit.run_multihop_follow \
  --vocab_file $BERT_DIR/vocab.txt \
  --bert_config_file $BERT_DIR/bert_config.json \
  --output_dir $OUTPUT \
  --train_file $DATA_DIR/train.json \
  --predict_file $DATA_DIR/dev.json \
  --test_file $DATA_DIR/test.json \
  --init_checkpoint $DATA_DIR/bert_init \
  --train_data_dir $DATA_DIR \
  --test_data_dir $DATA_DIR \
  --do_train=True \
  --do_predict=False \
  --do_test=True \
  --num_train_epochs 10.0 \
  --data_type wikimovie \
  --model_type wikimovie \
  --train_batch_size 40 \
  --predict_batch_size 40 \
  --entity_score_threshold 1e-4 \
  --sparse_strategy dense_first \
  --num_mips_neighbors 5000 \
  --softmax_temperature 4.0 \
  --num_hops 1 \
  --logtostderr

DATA_DIR="data/preprocessed/2-hop/indexed"
OUTPUT_DIR="models/multihop/2-hop"

python -m language.labs.drkit.run_multihop_follow \
  --vocab_file $BERT_DIR/vocab.txt \
  --bert_config_file $BERT_DIR/bert_config.json \
  --output_dir $OUTPUT \
  --train_file $DATA_DIR/train.json \
  --predict_file $DATA_DIR/dev.json \
  --test_file $DATA_DIR/test.json \
  --init_checkpoint $DATA_DIR/bert_init \
  --train_data_dir $DATA_DIR \
  --test_data_dir $DATA_DIR \
  --do_train=True \
  --do_predict=False \
  --do_test=True \
  --num_train_epochs 20.0 \
  --data_type wikimovie-2hop \
  --model_type wikimovie-2hop \
  --train_batch_size 40 \
  --predict_batch_size 40 \
  --entity_score_threshold 1e-4 \
  --sparse_strategy dense_first \
  --num_mips_neighbors 20000 \
  --softmax_temperature 3.0 \
  --num_hops 2 \
  --logtostderr

DATA_DIR="data/preprocessed/3-hop/indexed"
OUTPUT_DIR="models/multihop/3-hop"

python -m language.labs.drkit.run_multihop_follow \
  --vocab_file $BERT_DIR/vocab.txt \
  --bert_config_file $BERT_DIR/bert_config.json \
  --output_dir $OUTPUT \
  --train_file $DATA_DIR/train.json \
  --predict_file $DATA_DIR/dev.json \
  --test_file $DATA_DIR/test.json \
  --init_checkpoint $DATA_DIR/bert_init \
  --train_data_dir $DATA_DIR \
  --test_data_dir $DATA_DIR \
  --do_train=True \
  --do_predict=False \
  --do_test=True \
  --num_train_epochs 20.0 \
  --data_type wikimovie-3hop \
  --model_type wikimovie-3hop \
  --train_batch_size 40 \
  --predict_batch_size 40 \
  --entity_score_threshold 1e-4 \
  --sparse_strategy dense_first \
  --num_mips_neighbors 15000 \
  --softmax_temperature 2.0 \
  --num_hops 3 \
  --logtostderr
