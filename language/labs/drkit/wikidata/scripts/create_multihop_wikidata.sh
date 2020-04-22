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

BASE="sling/local/data"
PRETRAIN_SPLIT="0"
PRETRAIN_OUT_DIR="$BASE/multihop/pretraining"
ONEHOP_DIR="$BASE/multihop/onehop"
TWOHOP_DIR="$BASE/multihop/twohop"
THREEHOP_DIR="$BASE/multihop/threehop"

mkdir -p $BASE/distant

# Preprocess wiki and extract distantly supervised facts.
python3 preprocessing/distantly_supervise.py --data $BASE

# Add negatives to for BERT pretraining.
python -m language.labs.drkit.wikidata.preprocessing.add_negatives \
  --input_pattern "$BASE/distant/facts-0000${PRETRAIN_SPLIT}-of-00010.json" \
  --output_prefix $PRETRAIN_OUT_DIR \
  --logtostderr

# Create 1-hop data.
## This is just a smaller set of the pretraining data.
## Training from split-0.
mkdir -p $ONEHOP_DIR/train
python preprocessing/distantly_supervise.py \
  --data $BASE \
  --output multihop/onehop/train \
  --wiki_split 0 \
  --max_n 10000
## Testing from split-1.
mkdir -p $ONEHOP_DIR/test
python preprocessing/distantly_supervise.py \
  --data $BASE \
  --output multihop/onehop/test \
  --wiki_split 1 \
  --max_n 10000

# Create 2-hop data.
## Take wiki splits and identify 2-hop queries within them.
## Train from split-0.
mkdir -p $TWOHOP_DIR/train
python -m language.labs.drkit.wikidata.preprocessing.create_follow_queries \
  --paragraphs_file $BASE/distant/paragraphs-00000-of-00010.json \
  --queries_file $BASE/distant/queries-00000-of-00010.json \
  --output_paragraphs_file $TWOHOP_DIR/train/paragraphs.json \
  --output_queries_file $TWOHOP_DIR/train/queries.json \
  --max_paragraphs 120000 \
  --logtostderr
## Test from split-1.
mkdir -p $TWOHOP_DIR/train
python -m language.labs.drkit.wikidata.preprocessing.create_follow_queries \
  --paragraphs_file $BASE/distant/paragraphs-00001-of-00010.json \
  --queries_file $BASE/distant/queries-00001-of-00010.json \
  --output_paragraphs_file $TWOHOP_DIR/test/paragraphs.json \
  --output_queries_file $TWOHOP_DIR/test/queries.json \
  --max_paragraphs 120000 \
  --logtostderr

# Create 3-hop data.
## Take wiki splits and identify 3-hop queries within them.
## Train from split-0.
mkdir -p $THREEHOP_DIR/train
python -m language.labs.drkit.wikidata.preprocessing.create_3hop_queries \
  --paragraphs_file $BASE/distant/paragraphs-00000-of-00010.json \
  --queries_file $BASE/distant/queries-00000-of-00010.json \
  --output_paragraphs_file $THREEHOP_DIR/train/paragraphs.json \
  --output_queries_file $THREEHOP_DIR/train/queries.json \
  --max_paragraphs 120000 \
  --logtostderr
## Test from split-1.
mkdir -p $THREEHOP_DIR/train
python -m language.labs.drkit.wikidata.preprocessing.create_3hop_queries \
  --paragraphs_file $BASE/distant/paragraphs-00001-of-00010.json \
  --queries_file $BASE/distant/queries-00001-of-00010.json \
  --output_paragraphs_file $THREEHOP_DIR/test/paragraphs.json \
  --output_queries_file $THREEHOP_DIR/test/queries.json \
  --max_paragraphs 120000 \
  --logtostderr
