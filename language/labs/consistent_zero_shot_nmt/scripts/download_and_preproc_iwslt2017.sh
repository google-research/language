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

# Download IWSLT17 dataset.
if [ ! -f "${EXP_DATASET_DIR}/DeEnItNlRo-DeEnItNl.tgz" ]; then
  echo "Downloading IWSLT17. This may take a while..."
  curl "https://wit3.fbk.eu/archive/2017-01-trnmted//texts/DeEnItNlRo/DeEnItNlRo/DeEnItNlRo-DeEnItNlRo.tgz" \
    -o "${EXP_DATASET_DIR}/DeEnItNlRo-DeEnItNl.tgz" \
    --compressed
fi

# Extract everything.
if [ ! -d "${EXP_DATASET_DIR}/original" ]; then
  echo "Extracting all files..."
  mkdir -p "${EXP_DATASET_DIR}/original"
  tar -xvzf "${EXP_DATASET_DIR}/DeEnItNlRo-DeEnItNl.tgz" -C "${EXP_DATASET_DIR}/original" --strip=1
fi

# Extract overlapping sentences from IWSLT17 training corpora.
if [ ! -d "${EXP_DATASET_DIR}/overlap" ]; then
  echo "Extracting overlapping sentences..."
  mkdir -p "${EXP_DATASET_DIR}/overlap"
  python "${SCRIPTS_DIR}/identify_overlap_iwslt17.py" \
    --input_data_dir "${EXP_DATASET_DIR}/original/" \
    --output_data_dir "${EXP_DATASET_DIR}/overlap/"
fi

echo "All done."
