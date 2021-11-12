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
# Generate retrieval-augmented datasets for the parse guiding experiments.
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

# Oracle training data.
# Augment oracle exemplars (exemplars with the same frame as the input).
# The ones with "plat" also add guiding tags.
gen train "ora5x20" "use-large" "add_oracle" '{"k":5,"n":20}' '{}'
gen train "ora5x20.anon" "use-large" "add_oracle" '{"k":5,"n":20}' \
  '{"anonymize":true}'
gen train "ora5x20.plat" "use-large" "add_oracle" '{"k":5,"n":20}' \
  '{"presets":["punc","plat"]}'
gen train "ora5x20.anon.plat" "use-large" "add_oracle" '{"k":5,"n":20}' \
  '{"presets":["punc","plat"],"anonymize":true}'

# Add guiding tags to non-oracle dev examples.
# This should cause degradation since the parser tries to follow the exemplars.
gen dev "top5.plat" "use-large" "add_top" '{"k":5}' \
  '{"presets":["punc","plat"]}'

# Oracle dev examples, without and with guiding tags.
gen dev "ora5xT" "use-large" "add_oracle" '{"k":5,"n":1,"p":1}' '{}'
gen dev "ora5xT.plat" "use-large" "add_oracle" '{"k":5,"n":1,"p":1}' \
  '{"presets":["punc","plat"]}'

# Augment adversarial exemplars (all exemplars have the same frame but different
# from the input). Used for Figure 5.
gen dev "adv5xT" "use-large" "add_adversarial" '{"k":5,"n":1,"p":1}' '{}'
gen dev "adv5xT.plat" "use-large" "add_adversarial" '{"k":5,"n":1,"p":1}' \
  '{"presets":["punc","plat"]}'
