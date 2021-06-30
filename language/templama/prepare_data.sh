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

SLING_BASE="$1"
OUT_BASE="$2"

WIKIOUT="$OUT_BASE/wikidata/"
TEMPOUT="$OUT_BASE/templama/"
SLING_KB="$SLING_BASE/data/e/kb/kb.sling"
SLING_WIKI="$SLING_BASE/data/e/wiki/en/mapping.sling"

python3 sling2facts.py \
  --basedir=$WIKIOUT \
  --sling_kb_file=$SLING_KB \
  --action="make_kb" \
  --quick_test=False \
  --keep_only_numeric_slots=False \
  --keep_only_date_slots=True \
  --skip_empty_objects=True \
  --skip_nonentity_objects=False \
  --skip_nonentity_qualifiers=False \
  --skip_qualifiers=False \
  --skip_nonenglish=False \
  --show_names=False \
  --close_types_with_inheritance=False \
  --close_locations_with_containment=False \
  --frame_to_echo="Q1079" \
  --inherit_props_from_entity="P580,P582,P585" \
  --logtostderr

python3 templama.py \
  --out_dir=$TEMPOUT \
  --facts_file=$WIKIOUT/kb.cfacts \
  --sling_kb_file=$SLING_KB \
  --sling_wiki_mapping_file=$SLING_WIKI \
  --logtostderr
