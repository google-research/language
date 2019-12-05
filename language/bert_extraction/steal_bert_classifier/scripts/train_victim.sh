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

# Make sure you have downloaded the pretrained BERT model as well as GLUE
# Download links for BERT models :- https://github.com/google-research/bert
# Download links for GLUE :- https://gluebenchmark.com/tasks

# For SST-2 experiments replace MNLI with SST-2 for $TASK_NAME

export BERT_DIR=/path/to/bert/uncased_L-24_H-1024_A-16
export GLUE_DIR=/path/to/glue
export TASK_NAME="MNLI"
export OUTPUT_DIR=/path/to/output_dir

python -m language.bert_extraction.steal_bert_classifier.models.run_classifier \
  --task_name=$TASK_NAME \
  --exp_name="train_victim" \
  --do_train=true \
  --do_eval=true \
  --do_lower_case=true \
  --save_checkpoints_steps=5000 \
  --data_dir=$GLUE_DIR/$TASK_NAME \
  --vocab_file=$BERT_DIR/vocab.txt \
  --bert_config_file=$BERT_DIR/bert_config.json \
  --init_checkpoint=$BERT_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=3e-5 \
  --num_train_epochs=3.0 \
  --output_dir=$OUTPUT_DIR
