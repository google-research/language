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

export BERT_DIR=/path/to/bert/uncased_L-24_H-1024_A-16
export GLUE_DIR=/path/to/glue
export TASK_NAME="MNLI"
export VICTIM_MODEL_DIR=/path/to/victim/model/directory
export EXTRACTED_MODEL_DIR=/path/to/extracted/model/directory

if ["$TASK_NAME" = "MNLI"]; then
  DEV_FILE_NAME="dev_matched.tsv"
else:
  DEV_FILE_NAME="dev.tsv"
fi

# take care of the --bert_config_file* flags for mismatched architecture settings
python -m language.bert_extraction.steal_bert_classifier.utils.model_diff_dataset \
  --task_name=$TASK_NAME \
  --bert_config_file1=$BERT_DIR/bert_config.json \
  --bert_config_file2=$BERT_DIR/bert_config.json \
  --vocab_file=$BERT_DIR/vocab.txt \
  --init_checkpoint1=$VICTIM_MODEL_DIR \
  --init_checkpoint2=$EXTRACTED_MODEL_DIR \
  --do_lower_case=true \
  --max_seq_length=128 \
  --use_random=false \
  --diff_input_file=$GLUE_DIR/$TASK_NAME/$DEV_FILE_NAME \
  --exp_name="agreement_victim_extracted"

