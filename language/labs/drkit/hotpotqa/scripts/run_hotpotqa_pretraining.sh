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

set -e

BERT_DIR="wwm_uncased_L-24_H-1024_A-16/"
HOTPOT_DIR="hotpot"
SLING_DIR="sling/local/data"
PREDATA_DIR="data"
PRETRAIN_DIR="models/pretraining"

# Find answer spans.
python -m language.labs.drkit.hotpotqa.preprocessing.convert_hotpot_to_mrqa \
  --in_file $HOTPOT_DIR/hotpot_train_v1.1.json \
  --out_file $PREDATA_DIR/HotpotQA.jsonl.gz \
  --logtostderr

python -m language.labs.drkit.hotpotqa.preprocessing.convert_wikidata_to_mrqa \
  --in_pattern "$SLING_DIR/distant/facts-0000%d-of-00010.json" \
  --out_pattern "$PREDATA_DIR/LSF_%d.jsonl.gz" \
  --logtostderr

# Convert to TFRecords.
## Hotpot
python -m language.labs.drkit.hotpotqa.preprocessing.create_tfrecords \
  --input_file $PREDATA_DIR/HotpotQA.jsonl.gz \
  --output_file $PREDATA_DIR/HotpotQA.tf_record \
  --vocab_file $BERT_DIR/vocab.txt \
  --logtostderr

## LSF
python -m language.labs.drkit.hotpotqa.preprocessing.create_tfrecords \
  --input_file "$PREDATA_DIR/LSF_-1,0,2,3,4.jsonl.gz" \
  --output_file "$PREDATA_DIR/LSF_-1,0,2,3,4.tf_record" \
  --vocab_file $BERT_DIR/vocab.txt \
  --logtostderr

# Launch Pretraining
python -m language.labs.drkit.run_dualencoder_qa \
  --init_checkpoint $BERT_DIR/bert_model.ckpt \
  --bert_config_file $BERT_DIR/bert_config.json \
  --output_dir $PRETRAIN_DIR \
  --pretrain_data_dir $PREDATA_DIR \
  --pretrain_tfrecord_file "LSF_-1,0,2,3,4:HotpotQA" \
  --do_pretrain=True \
  --do_eval=False \
  --pretrain_batch_size 64 \
  --num_pretrain_epochs 3 \
  --use_tpu=False \
  --logtostderr
