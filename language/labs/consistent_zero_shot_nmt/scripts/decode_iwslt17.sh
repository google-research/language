#!/bin/bash

PROBLEM=translate_iwslt17_nonoverlap
MODEL=agreement_multilingual_nmt
HPARAMS=ag_gnmt_luong_att
HPARAMS_EXTRA="dt-basic.dc-false.dac-0.01.eac-0.0001.st-false.smc-10"
TIMESTAMP="1128_1056"

DATA_DIR=$1
MODEL_DIR=$2
OUTPUT_DIR="$MODEL_DIR/$MODEL.$HPARAMS.$PROBLEM.$TIMESTAMP/$HPARAMS_EXTRA"

BEAM_SIZE=10
ALPHA=0.6

src_langs=( de en nl en it en ro en de it de ro nl it nl ro de nl it ro )
tgt_langs=( en de en nl en it en ro it de ro de it nl ro nl nl de ro it )
for idx in "${!src_langs[@]}"; do
  src=${src_langs[$idx]}
  tgt=${tgt_langs[$idx]}
  python -m language.labs.consistent_zero_shot_nmt.bin.t2t_decoder \
    --data_dir=$DATA_DIR \
    --problem=$PROBLEM \
    --model=$MODEL \
    --hparams_set=$HPARAMS \
    --output_dir=$OUTPUT_DIR \
    --hparams="beam_width=$BEAM_SIZE" \
    --decode_hparams="beam_size=$BEAM_SIZE,alpha=$ALPHA,batch_size=1" \
    --decode_from_file="$DATA_DIR/test/IWSLT17.TED.tst2010.$src-$tgt.$src-$tgt" \
    --decode_to_file="$OUTPUT_DIR/tmp/results_basic_iwslt/translation.$src-$tgt.$tgt" \
    --alsologtostderr
done
