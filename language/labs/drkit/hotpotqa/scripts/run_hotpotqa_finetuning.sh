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
HOTPOT_DIR="hotpot"
DATA_DIR="data/tiny-preprocessed-corpus"
OUTPUT="models/multihop"

# Entity link questions.
## Train
python -m language.labs.drkit.hotpotqa.preprocessing.link_questions \
  --hotpotqa_file $HOTPOT_DIR/hotpot_train_v1.1.json \
  --entity_dir $DATA_DIR \
  --vocab_file $BERT_DIR/vocab.txt \
  --output_file $DATA_DIR/hotpot_train_tfidf_entities.json \
  --logtostderr
## Dev
python -m language.labs.drkit.hotpotqa.preprocessing.link_questions \
  --hotpotqa_file $HOTPOT_DIR/hotpot_dev_distractor_v1.json \
  --entity_dir $DATA_DIR \
  --vocab_file $BERT_DIR/vocab.txt \
  --output_file $DATA_DIR/hotpot_dev_tfidf_entities.json \
  --logtostderr

# Run multi-hop finetuning.
python -m language.labs.drkit.run_multihop_follow \
  --vocab_file $BERT_DIR/vocab.txt \
  --bert_config_file $BERT_DIR/bert_config.json \
  --output_dir $OUTPUT \
  --train_file $DATA_DIR/hotpot_train_tfidf_entities.json \
  --predict_file $DATA_DIR/hotpot_dev_tfidf_entities.json \
  --init_checkpoint $DATA_DIR/bert_init \
  --train_data_dir $DATA_DIR \
  --test_data_dir $DATA_DIR \
  --do_train=True \
  --do_predict=True \
  --num_train_epochs 30.0 \
  --data_type hotpotqa \
  --model_type hotpotqa \
  --sparse_strategy sparse_first \
  --num_hops 2 \
  --logtostderr
