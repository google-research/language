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
INIT_CKPT="../wikidata/models/pretraining/model.ckpt-?????"
DATA_DIR="data/pretraining/2-hop"
OUTPUT_DIR="models/pretraining/2-hop"

python -m language.labs.drkit.run_dualencoder_lsf \
  --vocab_file=$BERT_DIR/vocab.txt \
  --bert_config_file=$BERT_DIR/bert_config.json \
  --init_checkpoint=$INIT_CKPT \
  --do_train=True \
  --train_file=$DATA_DIR/train.json \
  --do_predict=False \
  --do_test=True \
  --test_file=$DATA_DIR/dev.json \
  --output_dir=$OUTPUT_DIR \
  --projection_dim=200 \
  --train_batch_size 48 \
  --num_train_epochs 12.0 \
  --max_seq_length 256 \
  --logtostderr
