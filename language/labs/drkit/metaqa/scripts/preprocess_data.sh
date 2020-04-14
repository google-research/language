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

WIKIMOVIE_DIR="movieqa"
METAQA_DIR="metaqa"
PREPROCESS_DIR="data/preprocessed"
PRETRAIN_DATA_DIR="data/pretraining"

mkdir -p $PREPROCESS_DIR

# Run preprocessing on METAQA data.
python -m language.labs.drkit.metaqa.preprocessing.metaqa_preprocess \
  --metaqa_dir $METAQA_DIR \
  --output_dir $PREPROCESS_DIR \
  --logtostderr

# Preprocess wiki corpus.
python -m language.labs.drkit.metaqa.preprocessing.process_wiki \
  --wiki_file $WIKIMOVIE_DIR/knowledge_source/wiki.txt \
  --entity_file $PREPROCESS_DIR/entities.txt \
  --output_file $PREPROCESS_DIR/processed_wiki.json \
  --logtostderr

# Create distant supervision data for pretraining.
for HOP in "1" "2" "3"; do
  DATA_DIR="$PREPROCESS_DIR/$HOP-hop"
  OUT_DIR="$PRETRAIN_DATA_DIR/$HOP-hop"
  mkdir -p $OUT_DIR
  python -m language.labs.drkit.metaqa.preprocessing.distantly_supervise \
    --paragraphs_file $PREPROCESS_DIR/processed_wiki.json \
    --kb_file $METAQA_DIR/kb.txt \
    --entity_file $PREPROCESS_DIR/entities.txt \
    --train_file $DATA_DIR/train.json \
    --test_file $DATA_DIR/test.json \
    --output_path $OUT_DIR \
    --logtostderr
done
