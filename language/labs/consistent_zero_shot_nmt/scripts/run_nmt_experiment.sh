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

rm -rf ${EXP_OUTPUT_DIR}

# Additional parameters.
EXP_HPARAMS=""
EXP_TRAIN_STEPS=1000000
EXP_LOCAL_EVAL_FREQ=500

python -m language.labs.consistent_zero_shot_nmt.bin.t2t_trainer \
  --problem=${EXP_PROBLEM_NAME} \
  --model=${EXP_MODEL_NAME} \
  --hparams=${EXP_HPARAMS} \
  --hparams_set=${EXP_CONF_NAME} \
  --data_dir=${EXP_DATASET_DIR}/tfrecords \
  --train_steps=${EXP_TRAIN_STEPS} \
  --output_dir=${EXP_OUTPUT_DIR} \
  --local_eval_frequency=${EXP_LOCAL_EVAL_FREQ} \
  --schedule=train_and_evaluate \
  --alsologtostderr
