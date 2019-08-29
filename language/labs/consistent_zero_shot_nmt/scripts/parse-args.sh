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

# Set defaults.
EXP_DATA_DIR="/tmp/data"
EXP_RESULTS_DIR="/tmp/results"
EXP_DATASET_NAME="iwslt17-official"
EXP_MODEL_NAME="basic_multilingual_nmt"
EXP_PROBLEM_NAME="translate_iwslt17_nonoverlap"

# Parse arguments and override defaults.
while [[ $# -gt 0 ]]; do
  case $1 in
    --data-dir=*)
      EXP_DATA_DIR=${1#*=}
      ;;
    --output-dir=*)
      EXP_RESULTS_DIR=${1#*=}
      ;;
    --dataset-name=*)
      EXP_DATASET_NAME=${1#*=}
      ;;
    --model-name=*)
      EXP_MODEL_NAME=${1#*=}
      ;;
    --problem-name=*)
      EXP_PROBLEM_NAME=${1#*=}
      ;;
    # Help.
    -h|--help)
      echo "Available arguments: --data-dir, --output-dir, --dataset-name, --model-name, --problem-name."
      exit 1
      ;;
    # Unsupported arguments.
    *)
      echo "Error: Unsupported argument $1" >&2
      exit 1
      ;;
  esac
  shift
done

# Set derived parameters.
EXP_DATASET_DIR="${EXP_DATA_DIR}/${EXP_DATASET_NAME}"
EXP_OUTPUT_DIR="${EXP_RESULTS_DIR}/${EXP_PROBLEM_NAME}/${EXP_MODEL_NAME}"

# Print arguments.
EXP_VARS=$(set | grep -e "^EXP_")
echo "-----------------"
echo "Parsed arguments:"
echo "-----------------"
echo ${EXP_VARS} | tr " " "\n"
echo "-----------------"
