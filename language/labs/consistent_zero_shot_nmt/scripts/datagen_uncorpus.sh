#!/bin/bash

PROBLEM=translate_uncorpus_exp1_lm
DATA_DIR=$1
TMP_DIR=$2
UNCORPUS_ORIG_DATA_EXP1=$3
UNCORPUS_ORIG_DATA_EXP1_LM=$4
UNCORPUS_ORIG_DATA_EXP2=$5
UNCORPUS_ORIG_DATA_EXP2_LM=$6

mkdir -p $DATA_DIR $TMP_DIR

python -m language.labs.consistent_zero_shot_nmt.bin.t2t_datagen \
  --data_dir=$DATA_DIR \
  --uncorpus_orig_data_exp1=$UNCORPUS_ORIG_DATA_EXP1 \
  --uncorpus_orig_data_exp1_lm=$UNCORPUS_ORIG_DATA_EXP1_LM \
  --uncorpus_orig_data_exp2=$UNCORPUS_ORIG_DATA_EXP2 \
  --uncorpus_orig_data_exp2_lm=$UNCORPUS_ORIG_DATA_EXP2_LM \
  --problem=$PROBLEM \
  --tmp_dir=$TMP_DIR \
  --alsologtostderr
