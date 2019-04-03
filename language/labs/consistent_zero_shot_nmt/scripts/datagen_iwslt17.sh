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
