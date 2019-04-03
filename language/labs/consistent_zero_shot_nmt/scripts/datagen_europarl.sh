#!/bin/bash

PROBLEM=translate_europarl_nonoverlap
DATA_DIR=$1
TMP_DIR=$2
EUROPARL_ORIG_DATA_PATH=$3
EUROPARL_OVERLAP_DATA_PATH=$4

mkdir -p $DATA_DIR $TMP_DIR

python -m language.labs.consistent_zero_shot_nmt.bin.t2t_datagen \
  --data_dir=$DATA_DIR \
  --europarl_orig_data_path=$EUROPARL_ORIG_DATA_PATH \
  --europarl_overlap_data_path=$EUROPARL_OVERLAP_DATA_PATH \
  --problem=$PROBLEM \
  --tmp_dir=$TMP_DIR \
  --alsologtostderr
