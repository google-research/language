#!/bin/bash

PROBLEM=translate_uncorpus_exp2_lm
MODEL=agreement_multilingual_nmt_lm
HPARAMS=ag_gnmt_luong_att_lm
HPARAMS_EXTRA="lr-0.01"
TIMESTAMP="pretrained_lm"

DATA_DIR=$1
MODEL_DIR=$2
OUTPUT_DIR="$MODEL_DIR/$MODEL.$HPARAMS.$PROBLEM.$TIMESTAMP/$HPARAMS_EXTRA"
TEST_DIR=$3

BEAM_SIZE=1
ALPHA=0.6

src_langs=( es fr ru )
tgt_langs=( es fr ru )
for idx in "${!src_langs[@]}"; do
  src=${src_langs[$idx]}
  tgt=${tgt_langs[$idx]}
  python -m language.labs.consistent_zero_shot_nmt.bin.t2t_decoder \
    --data_dir=$DATA_DIR \
    --problem=$PROBLEM \
    --model=$MODEL \
    --hparams_set=$HPARAMS \
    --output_dir=$OUTPUT_DIR \
    --hparams="lm_do_train=True,beam_width=$BEAM_SIZE" \
    --decode_hparams="beam_size=$BEAM_SIZE,alpha=$ALPHA" \
    --decode_from_file="$TEST_DIR/test/UNv1.0.testset.$src-$tgt" \
    --decode_to_file="$OUTPUT_DIR/tmp/results_lm_uncorpus_exp1/translation.$src-$tgt.$tgt" \
    --alsologtostderr
done
