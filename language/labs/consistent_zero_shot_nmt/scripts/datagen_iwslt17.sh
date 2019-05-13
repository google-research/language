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

PROBLEM=translate_iwslt17
DATA_DIR=$1
TMP_DIR=$2
IWSLT17_ORIG_DATA_PATH=$3
IWSLT17_OVERLAP_DATA_PATH=$4

mkdir -p $DATA_DIR $TMP_DIR

python -m language.labs.consistent_zero_shot_nmt.bin.t2t_datagen \
  --data_dir=$DATA_DIR \
  --iwslt17_orig_data_path=$IWSLT17_ORIG_DATA_PATH \
  --iwslt17_overlap_data_path=$IWSLT17_OVERLAP_DATA_PATH \
  --problem=$PROBLEM \
  --tmp_dir=$TMP_DIR \
  --alsologtostderr
