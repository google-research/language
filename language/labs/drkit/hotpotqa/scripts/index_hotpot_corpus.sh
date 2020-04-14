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

BERT_BASE_DIR="wwm_uncased_L-24_H-1024_A-16/"
PRETRAIN_DIR="models/pretraining"
WIKI_DIR="hotpot/wiki"
WIKI_FILE="data/tiny-wiki.json"
OUTPUT="data/tiny-preprocessed-corpus"
NUM_SHARDS="4"

python -m language.labs.drkit.hotpotqa.preprocessing.parse_wiki \
  --base_dir $WIKI_DIR/enwiki-20171001-pages-meta-current-withlinks-abstracts \
  --output_file $WIKI_FILE \
  --debug=True \
  --logtostderr

python -m language.labs.drkit.hotpotqa.index \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --wiki_file=$WIKI_FILE \
  --multihop_output_dir=$OUTPUT \
  --do_preprocess=True \
  --do_copy=True \
  --do_embed=False \
  --do_combine=False \
  --output_dir=$PRETRAIN_DIR \
  --pretrain_dir=$PRETRAIN_DIR \
  --num_shards=$NUM_SHARDS \
  --logtostderr

for i in $(seq 1 $(($NUM_SHARDS+1))); do
  echo $(($i-1))
  python -m language.labs.drkit.hotpotqa.index \
    --vocab_file=$BERT_BASE_DIR/vocab.txt \
    --bert_config_file=$BERT_BASE_DIR/bert_config.json \
    --wiki_file=$WIKI_FILE \
    --multihop_output_dir=$OUTPUT \
    --do_preprocess=False \
    --do_copy=False \
    --do_embed=True \
    --do_combine=False \
    --output_dir=$PRETRAIN_DIR \
    --pretrain_dir=$PRETRAIN_DIR \
    --num_shards=$NUM_SHARDS \
    --my_shard=$(($i-1)) \
    --logtostderr >> /tmp/$i.log 2>&1 &
done

wait

python -m language.labs.drkit.hotpotqa.index \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --wiki_file=$WIKI_FILE \
  --multihop_output_dir=$OUTPUT \
  --do_preprocess=False \
  --do_copy=False \
  --do_embed=False \
  --do_combine=True \
  --output_dir=$PRETRAIN_DIR \
  --pretrain_dir=$PRETRAIN_DIR \
  --num_shards=$NUM_SHARDS \
  --logtostderr

