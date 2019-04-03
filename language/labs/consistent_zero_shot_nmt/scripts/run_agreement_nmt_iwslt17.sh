#!/bin/bash

NAME=agreement-multilingual-nmt-iwslt17
PROBLEM=translate_iwslt17_nonoverlap
MODEL=agreement_multilingual_nmt
CONF=ag_gnmt_luong_att

OUT_DIR=$1
DATA_DIR=$2

rm -rf $OUT_DIR

python -m language.labs.consistent_zero_shot_nmt.bin.t2t_trainer \
  --problem=$PROBLEM \
  --model=$MODEL \
  --hparams="decoder_type=basic,decoder_continuous=False,scheduled_training=False" \
  --hparams_set=$CONF \
  --data_dir=$DATA_DIR \
  --train_steps=10000 \
  --output_dir=$OUT_DIR \
  --schedule=train_and_evaluate \
  --local_eval_frequency=500 \
  --alsologtostderr
