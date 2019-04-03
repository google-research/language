#!/bin/bash

PROBLEM=translate_europarl_nonoverlap
MODEL=agreement_multilingual_nmt
HPARAMS=ag_gnmt_luong_att
HPARAMS_EXTRA="dc-false.eac-0.001.dac-0.0001.dals-true.lmc-0.st-false.smc-10"
TIMESTAMP="1204_2249"

DATA_DIR=$1
MODEL_DIR=$2
OUTPUT_DIR="$MODEL_DIR/$MODEL.$HPARAMS.$PROBLEM.$TIMESTAMP/$HPARAMS_EXTRA"

BEAM_SIZE=10
ALPHA=0.6

src_langs=( fr fr )
tgt_langs=( es de )
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
    --decode_hparams="beam_size=$BEAM_SIZE,alpha=$ALPHA" \
    --decode_from_file="$DATA_DIR/translation/$MODEL.$HPARAMS.$PROBLEM.$TIMESTAMP.$HPARAMS_EXTRA.pivot.translation.$src-en.en-$tgt" \
    --decode_to_file="$DATA_DIR/translation/$MODEL.$HPARAMS.$PROBLEM.$TIMESTAMP.$HPARAMS_EXTRA.pivot.translation.$src-$tgt" \
    --alsologtostderr
done
