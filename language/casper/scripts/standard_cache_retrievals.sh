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
# Cache the top 100 retrievals for each example and dump them to JSONL files.
# Note that the input query itself is excluded from the retrieval results.
# The retrieval index is always the entire training data.
set -e
set -u

[[ -z "${DATA_DIR}" ]] && echo "Error: DATA_DIR must be set." && exit 1

# Retrieve with USE-large

python -m language.casper.retrieve.cache_query_retrievals \
  --alsologtostderr \
  --retriever=use --embedder_size=large --neighbor_filter=simple \
  --index_files="${DATA_DIR}/raw/train.no_exemplars.jsonl" \
  --example_files="${DATA_DIR}/raw/train.no_exemplars.jsonl",\
"${DATA_DIR}/raw/dev.no_exemplars.jsonl",\
"${DATA_DIR}/raw/test.no_exemplars.jsonl" \
  --output_files="${DATA_DIR}/raw/train.use-large.jsonl",\
"${DATA_DIR}/raw/dev.use-large.jsonl",\
"${DATA_DIR}/raw/test.use-large.jsonl"

# Retrieve with BERT-base and BERT-large (ablation)

python -m language.casper.retrieve.cache_query_retrievals \
  --alsologtostderr \
  --retriever=bert_cls --embedder_size=base --neighbor_filter=simple \
  --index_files="${DATA_DIR}/raw/train.no_exemplars.jsonl" \
  --example_files="${DATA_DIR}/raw/train.no_exemplars.jsonl",\
"${DATA_DIR}/raw/dev.no_exemplars.jsonl" \
  --output_files="${DATA_DIR}/raw/train.bert-base.jsonl",\
"${DATA_DIR}/raw/dev.bert-base.jsonl"

python -m language.casper.retrieve.cache_query_retrievals \
  --alsologtostderr \
  --retriever=bert_cls --embedder_size=large --neighbor_filter=simple \
  --index_files="${DATA_DIR}/raw/train.no_exemplars.jsonl" \
  --example_files="${DATA_DIR}/raw/train.no_exemplars.jsonl",\
"${DATA_DIR}/raw/dev.no_exemplars.jsonl" \
  --output_files="${DATA_DIR}/raw/train.bert-large.jsonl",\
"${DATA_DIR}/raw/dev.bert-large.jsonl"

# Oracle retriever that only retrieves exemplars with the same template as the
# input query (ablation)

python -m language.casper.retrieve.cache_query_retrievals \
  --alsologtostderr \
  --retriever=use --embedder_size=large --neighbor_filter=match_frame \
  --index_files="${DATA_DIR}/raw/train.no_exemplars.jsonl" \
  --example_files="${DATA_DIR}/raw/train.no_exemplars.jsonl",\
"${DATA_DIR}/raw/dev.no_exemplars.jsonl" \
  --output_files="${DATA_DIR}/raw/train.use-large.match-frame.jsonl",\
"${DATA_DIR}/raw/dev.use-large.match-frame.jsonl"
