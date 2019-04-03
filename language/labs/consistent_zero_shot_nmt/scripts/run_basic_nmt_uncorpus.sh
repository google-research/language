#!/bin/bash

NAME=basic-multilingual-nmt-uncorpus-exp1
PROBLEM=translate_uncorpus_exp1
MODEL=basic_multilingual_nmt
CONF=basic_gnmt_luong_att_multi

OUT_DIR=$1
DATA_DIR=$2

rm -rf $OUT_DIR

python -m language.labs.consistent_zero_shot_nmt.bin.t2t_trainer \
  --problem=$PROBLEM \
  --model=$MODEL \
  --hparams="" \
  --hparams_set=$CONF \
  --data_dir=$DATA_DIR \
  --train_steps=10000 \
  --output_dir=$OUT_DIR \
  --schedule=train_and_evaluate \
  --local_eval_frequency=1000 \
  --alsologtostderr
