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
# Generate retrieval-augmented datasets for the schema refactoring experiments.
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

# Simulate label split backwards by merging multiple labels into the same name.
LABEL_MAP='{
"IN:GET_REMINDER": "IN:GET_EVENT",
"SL:TYPE_CONTENT": "SL:TYPE_RELATION",
"IN:GET_TODO": "IN:GET_MESSAGE",
"SL:MUSIC_PROVIDER_NAME": "SL:MUSIC_PLAYLIST_TITLE",
"SL:RECIPES_COOKING_METHOD": "SL:RECIPES_SOURCE",
"IN:UPDATE_CALL": "IN:SWITCH_CALL",
"IN:GET_LOCATION": "IN:GET_CONTACT",
"IN:GET_AVAILABILITY": "IN:SET_AVAILABLE",
"SL:EMPLOYER": "SL:SCHOOL",
"SL:NEWS_CATEGORY": "SL:NEWS_TOPIC"
}'
AFFECTED_LABELS='IN:GET_REMINDER,IN:GET_EVENT,SL:TYPE_CONTENT,'\
'SL:TYPE_RELATION,IN:GET_TODO,IN:GET_MESSAGE,SL:MUSIC_PROVIDER_NAME,'\
'SL:MUSIC_PLAYLIST_TITLE,SL:RECIPES_COOKING_METHOD,SL:RECIPES_SOURCE,'\
'IN:UPDATE_CALL,IN:SWITCH_CALL,IN:GET_LOCATION,IN:GET_CONTACT,'\
'IN:GET_AVAILABILITY,IN:SET_AVAILABLE,SL:EMPLOYER,SL:SCHOOL,'\
'SL:NEWS_CATEGORY,SL:NEWS_TOPIC'

# Baseline (no augmentation)
gen train "noaug.label-merge" "use-large" "query_only" '{}' \
  '{"rename_labels":'"${LABEL_MAP}"'}'
gen dev "noaug.label-merge" "use-large" "query_only" '{}' \
  '{"rename_labels":'"${LABEL_MAP}"'}'

# Augment sampled K exemplars, optionally with anonymization + guiding tags.
gen train "samp5.label-merge" "use-large" "add_samp" '{"k":5,"n":20}' \
  '{"rename_labels":'"${LABEL_MAP}"'}'
gen train "samp5.label-merge.anon" "use-large" "add_samp" '{"k":5,"n":20}' \
  '{"rename_labels":'"${LABEL_MAP}"',"anonymize":true}'
gen train "ora5.label-merge.plat" "use-large" "add_oracle" '{"k":5,"n":20}' \
  '{"presets":["punc","plat"],"rename_labels":'"${LABEL_MAP}"'}'
gen train "ora5.label-merge.plat.anon" "use-large" "add_oracle" \
  '{"k":5,"n":20}' \
  '{"presets":["punc","plat"],"rename_labels":'"${LABEL_MAP}"',"anonymize":true}'

# Augment top K exemplars at test time.
gen dev "top5.label-merge" "use-large" "add_top" '{"k":5}' \
  '{"rename_labels":'"${LABEL_MAP}"'}'

# For models with guiding tags, add the guiding tags to affected dev examples.
# Please run standard_gen_datasets.sh first.
python -m language.casper.augment.patch_guiding_tag \
  --alsologtostderr \
  --src_pattern="${DATA_DIR}/t5/top5/dev.tsv*" \
  --tgt_dir="${DATA_DIR}/t5/top5.label-split.plat" \
  --file_format="tsv" \
  --affected_labels="${AFFECTED_LABELS}"
