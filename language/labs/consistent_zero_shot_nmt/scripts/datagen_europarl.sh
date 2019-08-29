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

ORIG_DATA_PATH="${EXP_DATASET_DIR}/original"
OVERLAP_DATA_PATH="${EXP_DATASET_DIR}/overlap"
TFRECORD_DATA_PATH="${EXP_DATASET_DIR}/tfrecords"
TMP_DIR="${EXP_DATASET_DIR}/tmp"

mkdir -p $TFRECORD_DATA_PATH $TMP_DIR

python -m language.labs.consistent_zero_shot_nmt.bin.t2t_datagen \
  --data_dir=${TFRECORD_DATA_PATH} \
  --europarl_orig_data_path=${ORIG_DATA_PATH} \
  --europarl_overlap_data_path=${OVERLAP_DATA_PATH} \
  --problem=${EXP_PROBLEM_NAME} \
  --tmp_dir=${TMP_DIR} \
  --alsologtostderr
