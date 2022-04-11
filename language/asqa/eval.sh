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

RESULTS_PATH=$1 #path to result json file
EXP_NAME=$2 #name of your experiment
OUTPUT_DIR=./results/${EXP_NAME}

mkdir -p ${OUTPUT_DIR}
python convert_to_roberta_format.py  \
  --asqa ./dataset/ASQA.json \
  --predictions $RESULTS_PATH  \
  --split dev \
  --output_path ${OUTPUT_DIR}

python transformers/examples/pytorch/question-answering/run_qa.py \
  --model_name_or_path ./roberta/roberta-squad \
  --validation_file ${OUTPUT_DIR}/qa.json \
  --do_eval \
  --version_2_with_negative \
  --max_seq_length 384 \
  --output_dir ${OUTPUT_DIR} \
  --null_score_diff_threshold 0

python scoring.py \
  --asqa ./dataset/ASQA.json \
  --predictions $RESULTS_PATH \
  --roberta_output ${OUTPUT_DIR}/eval_predictions.json \
  --split dev \
  --out_dir $OUTPUT_DIR
