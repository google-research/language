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

BERT_DIR="wwm_uncased_L-24_H-1024_A-16/"
HOTPOT_DIR="hotpot"
OUTPUT="models/answer"

# Train answer extraction model.
python -m language.labs.drkit.hotpotqa.answer_extractor \
  --vocab_file $BERT_DIR/vocab.txt \
  --bert_config_file $BERT_DIR/bert_config.json \
  --init_checkpoint $BERT_DIR/bert_model.ckpt \
  --output_dir $OUTPUT \
  --train_file $HOTPOT_DIR/hotpot_train_v1.1.json \
  --predict_file $HOTPOT_DIR/hotpot_dev_distractor_v1.json \
  --do_train=True \
  --do_predict=True \
  --train_batch_size 32 \
  --num_train_epochs 5.0 \
  --use_tpu=False \
  --logtostderr
