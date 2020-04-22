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

BERT_DIR="wwm_uncased_L-24_H-1024_A-16/"
TEST_DIR="data/tiny-preprocessed-corpus"
DRKIT_DIR="models/multihop"
BERT_CKPT="models/answer"
PASSAGES="data/tiny-wiki.json"
OUTPUT="/tmp/demo"
WEB="language/labs/drkit/hotpotqa/web"

python language.labs.drkit.hotpotqa.demo \
  --vocab_file $BERT_DIR/vocab.txt \
  --bert_config_file $BERT_DIR/bert_config.json \
  --output_dir $OUTPUT \
  --init_checkpoint $DRKIT_DIR \
  --hotpot_init_checkpoint $BERT_CKPT \
  --raw_passages $PASSAGES \
  --train_data_dir $TEST_DIR \
  --model_type "hotpotqa" \
  --sparse_strategy "sparse_first" \
  --num_hops 2 \
  --port 8888 \
  --web_path $WEB \
  --logtostderr
