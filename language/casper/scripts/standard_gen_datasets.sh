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
# Generate retrieval-augmented datasets for the standard setup.
set -e
set -u

[[ -z "${DATA_DIR}" ]] && echo "Error: DATA_DIR must be set." && exit 1

function gen() {
  split="${1}"
  outdir="${2}"
  retriever="${3}"
  converter="${4}"
  converter_kwargs="${5}"
  formatter_kwargs="${6}"
  python -m language.casper.augment.cached_retrieval_to_dataset \
    --alsologtostderr \
    --"${split}"_data_paths="${DATA_DIR}/raw/${split}.${retriever}.jsonl" \
    --index_paths="${DATA_DIR}/raw/train.no_exemplars.jsonl" \
    --output_dir="${DATA_DIR}/t5/${outdir}" \
    --file_format="tsv" \
    --example_converter="${converter}" \
    --converter_kwargs="${converter_kwargs}" \
    --formatter_kwargs="${formatter_kwargs}"
}

# Baseline (no augmentation)
gen train "noaug" "use-large" "query_only" '{}' '{}'
gen dev "noaug" "use-large" "query_only" '{}' '{}'
gen test "noaug" "use-large" "query_only" '{}' '{}'

# Augment sampled K exemplars. Used for training data.
gen train "samp5x20" "use-large" "add_samp" '{"k":5,"n":20}' '{}'

# Augment top K exemplars. Used for test data.
gen dev "top5" "use-large" "add_top" '{"k":5}' '{}'
gen test "top5" "use-large" "add_top" '{"k":5}' '{}'

# Anonymization.
# The final model mixes anonymized training examples with the original ones.
gen train "samp5x20.anon" "use-large" "add_samp" '{"k":5,"n":20}' \
  '{"anonymize":true}'
gen dev "top5.anon" "use-large" "add_top" '{"k":5}' '{"anonymize":true}'
gen test "top5.anon" "use-large" "add_top" '{"k":5}' '{"anonymize":true}'

# Ablation: Retriever
gen train "samp5x20.bert-base" "bert-base" "add_samp" \
  '{"k":5,"n":20}' '{}'
gen dev "top5.bert-base" "bert-base" "add_top" '{"k":5}' '{}'
gen train "samp5x20.bert-large" "bert-large" "add_samp" \
  '{"k":5,"n":20}' '{}'
gen dev "top5.bert-large" "bert-large" "add_top" '{"k":5}' '{}'
gen train "samp5x20.oracle" "use-large.match-frame" "add_samp" \
  '{"k":5,"n":20}' '{}'
gen dev "top5.oracle" "use-large.match-frame" "add_top" '{"k":5}' '{}'

# Ablation: Number of exemplars, or using top-K exemplars at training time
gen train "samp1x1" "use-large" "add_samp" '{"k":1,"n":1}' '{}'
gen train "samp1x20" "use-large" "add_samp" '{"k":1,"n":20}' '{}'
gen train "top1" "use-large" "add_top" '{"k":1}' '{}'
gen dev "top1" "use-large" "add_top" '{"k":1}' '{}'

gen train "samp3x20" "use-large" "add_samp" '{"k":3,"n":20}' '{}'
gen dev "top3" "use-large" "add_top" '{"k":3}' '{}'

gen train "samp5x1" "use-large" "add_samp" '{"k":5,"n":1}' '{}'
gen train "top5" "use-large" "add_top" '{"k":5}' '{}'

gen train "samp10x20" "use-large" "add_samp" '{"k":10,"n":20}' '{}'
gen dev "top10" "use-large" "add_top" '{"k":10}' '{}'
