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
DATA_DIR="data/onehop"
OUTPUT_DIR="models/multihop/onehop"

python -m language.labs.drkit.run_multihop_follow \
  --vocab_file $BERT_DIR/vocab.txt \
  --bert_config_file $BERT_DIR/bert_config.json \
  --output_dir $OUTPUT \
  --train_file $DATA_DIR/train/indexed/train_qrys.json \
  --predict_file $DATA_DIR/train/indexed/dev_qrys.json \
  --test_file $DATA_DIR/test/indexed/dev_qrys.json \
  --init_checkpoint $DATA_DIR/train/indexed/bert_init \
  --train_data_dir $DATA_DIR/train/indexed \
  --test_data_dir $DATA_DIR/train/indexed \
  --do_train=True \
  --do_predict=False \
  --do_test=True \
  --num_train_epochs 20.0 \
  --data_type onehop \
  --model_type onehop \
  --train_batch_size 40 \
  --entity_score_threshold 1e-4 \
  --sparse_strategy dense_first \
  --num_mips_neighbors 500 \
  --softmax_temperature 1.0 \
  --num_hops 1 \
  --logtostderr

DATA_DIR="data/twohop"
OUTPUT_DIR="models/multihop/twohop"

python -m language.labs.drkit.run_multihop_follow \
  --vocab_file $BERT_DIR/vocab.txt \
  --bert_config_file $BERT_DIR/bert_config.json \
  --output_dir $OUTPUT \
  --train_file $DATA_DIR/train/indexed/train_qrys.json \
  --predict_file $DATA_DIR/train/indexed/dev_qrys.json \
  --test_file $DATA_DIR/test/indexed/dev_qrys.json \
  --init_checkpoint $DATA_DIR/train/indexed/bert_init \
  --train_data_dir $DATA_DIR/train/indexed \
  --test_data_dir $DATA_DIR/train/indexed \
  --do_train=True \
  --do_predict=False \
  --do_test=True \
  --num_train_epochs 20.0 \
  --data_type twohop \
  --model_type twohop \
  --train_batch_size 40 \
  --entity_score_threshold 1e-4 \
  --sparse_strategy dense_first \
  --num_mips_neighbors 10000 \
  --softmax_temperature 3.0 \
  --num_hops 2 \
  --logtostderr

DATA_DIR="data/threehop"
OUTPUT_DIR="models/multihop/threehop"

python -m language.labs.drkit.run_multihop_follow \
  --vocab_file $BERT_DIR/vocab.txt \
  --bert_config_file $BERT_DIR/bert_config.json \
  --output_dir $OUTPUT \
  --train_file $DATA_DIR/train/indexed/train_qrys.json \
  --predict_file $DATA_DIR/train/indexed/dev_qrys.json \
  --test_file $DATA_DIR/test/indexed/dev_qrys.json \
  --init_checkpoint $DATA_DIR/train/indexed/bert_init \
  --train_data_dir $DATA_DIR/train/indexed \
  --test_data_dir $DATA_DIR/train/indexed \
  --do_train=True \
  --do_predict=False \
  --do_test=True \
  --num_train_epochs 20.0 \
  --max_query_length 35 \
  --data_type threehop \
  --model_type threehop \
  --train_batch_size 32 \
  --entity_score_threshold 1e-4 \
  --sparse_strategy dense_first \
  --num_mips_neighbors 5000 \
  --softmax_temperature 4.0 \
  --num_hops 3 \
  --logtostderr
