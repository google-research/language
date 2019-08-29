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
#!/usr/bin/env bash

set -e

# Parse cmd arguments.
SCRIPTS_DIR="$( dirname "${BASH_SOURCE[0]}" )"
source "${SCRIPTS_DIR}/parse-args.sh"

echo "Writing to ${EXP_DATASET_DIR}."
mkdir -p ${EXP_DATASET_DIR}

if [ ! -f "${EXP_DATASET_DIR}/de-en.tgz" ]; then
  echo "Downloading Europarl v7 DE-EN. This may take a while..."
  curl "https://www.statmt.org/europarl/v7/de-en.tgz" \
    -o "${EXP_DATASET_DIR}/de-en.tgz" \
    --compressed
fi
if [ ! -f "${EXP_DATASET_DIR}/fr-en.tgz" ]; then
  echo "Downloading Europarl v7 FR-EN. This may take a while..."
  curl "https://www.statmt.org/europarl/v7/fr-en.tgz" \
    -o "${EXP_DATASET_DIR}/fr-en.tgz" \
    --compressed
fi
if [ ! -f "${EXP_DATASET_DIR}/es-en.tgz" ]; then
  echo "Downloading Europarl v7 ES-EN. This may take a while..."
  curl "https://www.statmt.org/europarl/v7/es-en.tgz" \
    -o "${EXP_DATASET_DIR}/es-en.tgz" \
    --compressed
fi
if [ ! -f "${EXP_DATASET_DIR}/devsets.tgz" ]; then
  echo
  echo "Downloading Europarl v7 devsets. This may take a while..."
  curl "https://www.statmt.org/wmt07/devsets.tgz" \
    -o "${EXP_DATASET_DIR}/devsets.tgz" \
    --compressed
fi

# Extract everything
if [ ! -d "${EXP_DATASET_DIR}/original" ]; then
  echo "Extracting all files..."
  mkdir -p "${EXP_DATASET_DIR}/original/parallel"
  tar -xvzf "${EXP_DATASET_DIR}/de-en.tgz" -C "${EXP_DATASET_DIR}/original/parallel"
  ln -s "${EXP_DATASET_DIR}/original/parallel/europarl-v7.de-en.de" "${EXP_DATASET_DIR}/original/parallel/europarl-v7.en-de.de"
  ln -s "${EXP_DATASET_DIR}/original/parallel/europarl-v7.de-en.en" "${EXP_DATASET_DIR}/original/parallel/europarl-v7.en-de.en"
  tar -xvzf "${EXP_DATASET_DIR}/fr-en.tgz" -C "${EXP_DATASET_DIR}/original/parallel"
  ln -s "${EXP_DATASET_DIR}/original/parallel/europarl-v7.fr-en.fr" "${EXP_DATASET_DIR}/original/parallel/europarl-v7.en-fr.fr"
  ln -s "${EXP_DATASET_DIR}/original/parallel/europarl-v7.fr-en.en" "${EXP_DATASET_DIR}/original/parallel/europarl-v7.en-fr.en"
  tar -xvzf "${EXP_DATASET_DIR}/es-en.tgz" -C "${EXP_DATASET_DIR}/original/parallel"
  ln -s "${EXP_DATASET_DIR}/original/parallel/europarl-v7.es-en.es" "${EXP_DATASET_DIR}/original/parallel/europarl-v7.en-es.es"
  ln -s "${EXP_DATASET_DIR}/original/parallel/europarl-v7.es-en.en" "${EXP_DATASET_DIR}/original/parallel/europarl-v7.en-es.en"
  tar -xvzf "${EXP_DATASET_DIR}/devsets.tgz" -C "${EXP_DATASET_DIR}/original"
fi

# Extract overlapping sentences from Europarl training corpora.
if [ ! -d "${EXP_DATASET_DIR}/overlap" ]; then
  echo "Extracting overlapping sentences..."
  mkdir -p "${EXP_DATASET_DIR}/overlap"
  python "${SCRIPTS_DIR}/identify_overlap_europarl.py" \
    --input_data_dir "${EXP_DATASET_DIR}/original/" \
    --output_data_dir "${EXP_DATASET_DIR}/overlap/"
fi

echo "All done."
